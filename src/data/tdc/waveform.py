try:
    import cupy as xp
except (ImportError, ModuleNotFoundError):
    import numpy as xp

from pathlib import Path

import numpy as np
from bidict import bidict
from few.waveform import Pn5AAKWaveform
from ldc.waveform.waveform import HpHc
from pycbc.waveform import get_td_waveform

from .utils.constant import Constant
from .utils.cosmology import Cosmology
from .utils.log_utils import MyLogger, TimerLog

mylogger = MyLogger(__name__)


class GB:
    def __init__(self, use_gpu=False, VGB=True):
        if use_gpu:
            self.xp = xp
        else:
            self.xp = np
        if VGB:
            self.key_idx = bidict(
                {
                    "f": 2,
                    "fdot": 3,
                    "beta": 4,
                    "lambda": 5,
                    "A": 1,
                    "iota": 6,
                    "psi": 7,
                    "phi0": 8,
                }
            )
        else:
            self.key_idx = bidict(
                {
                    "f": 0,
                    "fdot": 1,
                    "beta": 2,
                    "lambda": 3,
                    "A": 4,
                    "iota": 5,
                    "psi": 6,
                    "phi0": 7,
                }
            )
        self.idx_key = self.key_idx.inverse

    def __call__(self, A, f, fdot, iota, phi0, psi, T=1.0, dt=10.0):
        # get the t array
        t = self.xp.arange(0.0, T * Constant.YRSID_SI, dt)
        cos2psi = self.xp.cos(2.0 * psi)
        sin2psi = self.xp.sin(2.0 * psi)
        cosiota = self.xp.cos(iota)

        fddot = 11.0 / 3.0 * fdot**2 / f

        # phi0 is phi(t = 0) not phi(t = t0)
        phase = 2 * np.pi * (f * t + 1.0 / 2.0 * fdot * t**2 + 1.0 / 6.0 * fddot * t**3) - phi0

        hSp = -self.xp.cos(phase) * A * (1.0 + cosiota * cosiota)
        hSc = -self.xp.sin(phase) * 2.0 * A * cosiota

        # from source frame to SSB frame
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        return hp + 1j * hc

    @TimerLog(mylogger.logger, "Reading catalog from file")
    def read_catalog(self, cat_path):
        if cat_path.suffix in [".txt", ".dat"]:
            par = np.loadtxt(cat_path)
        elif cat_path.suffix == ".npy":
            par = np.load(cat_path)
        permute = [
            self.key_idx["A"],
            self.key_idx["f"],
            self.key_idx["fdot"],
            self.key_idx["iota"],
            self.key_idx["phi0"],
            self.key_idx["psi"],
            self.key_idx["lambda"],
            self.key_idx["beta"],
        ]
        self.para_cat = par[:, permute]


class AAK(object):
    def __init__(self, use_gpu=True, n_signal=1):
        self.n_signal = n_signal
        # keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
        inspiral_kwargs = {
            "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
            "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
        }
        # keyword arguments for summation generator (AAKSummation)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is availabel for this type of summation
            "pad_output": False,
        }
        self.Pn5AAK = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

    def __call__(
        self,
        M,
        mu,
        a,
        p0,
        e0,
        Y0,
        qS,
        phiS,
        qK,
        phiK,
        dist,
        Phi_phi0,
        Phi_theta0,
        Phi_r0,
        mich,
        dt=10,
        T=1,
    ):
        # print(dt,T)
        hS = self.Pn5AAK(
            M,
            mu,
            a,
            p0,
            e0,
            Y0,
            dist,
            qS,
            phiS,
            qK,
            phiK,
            Phi_phi0=Phi_phi0,
            Phi_theta0=Phi_theta0,
            Phi_r0=Phi_r0,
            mich=mich,
            dt=dt,
            T=T,
        )

        hSp = hSc = xp.zeros(self.n_signal)
        n = len(hS)
        # print(len(hS), self.n_signal)
        if n <= self.n_signal:
            hSp[:n] = hS.real
            hSc[:n] = hS.imag
        else:
            hSp = hS.real[:n]
            hSc = hS.imag[:n]

        # from source frame to SSB frame
        cosqS = xp.cos(qS)
        sinqS = xp.sin(qS)
        cosqK = xp.cos(qK)
        sinqK = xp.sin(qK)

        up_ldc = cosqS * sinqK * xp.cos(phiS - phiK) - cosqK * sinqS
        dw_ldc = sinqK * xp.sin(phiS - phiK)
        if dw_ldc:
            psi_ldc = xp.arctan2(up_ldc, dw_ldc)
        else:
            psi_ldc = 0.5 * np.pi

        cos2psi = xp.cos(2.0 * psi_ldc)
        sin2psi = xp.sin(2.0 * psi_ldc)
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        # n_signal = int(self.T_buffer * YRSID_SI / dt)
        # self.n_signal

        return hp + 1j * hc


class MBHB(object):
    def __init__(self, f_min, T_buffer, buffer_ind, apx="SEOBNRv4_opt"):
        self.apx = apx
        self.f_min = f_min
        self.T_buffer = T_buffer
        self.buffer_ind = buffer_ind

    def __call__(self, M, q, spin1z, spin2z, coa_phase, distance, iota, psi, t_c, T=1, dt=10):
        ratio = M / 100
        m1, m2 = MBHB.m1_m2_from_M_q(100, q)
        # print(ratio*self.f_min)
        f_lower = np.max([ratio * self.f_min, 3])

        hSp_tmp, hSc_tmp = get_td_waveform(
            approximant=self.apx,
            mass1=m1,
            mass2=m2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=iota,
            coa_phase=coa_phase,
            delta_t=dt / ratio,
            f_lower=f_lower,
        )

        # adjust merge time to t_c
        n_pycbc = len(hSp_tmp)
        n_signal = int(self.T_buffer * Constant.YRSID_SI / dt)
        a = np.where(hSp_tmp.sample_times < dt / ratio)
        merge_idx_pycbc = a[0][-1]
        # t = np.arange(0.0, T * YRSID_SI, dt)
        # t_c : in [yr]
        hSp = hSc = np.zeros(n_signal)
        merge_idx = int(t_c * Constant.YRSID_SI / dt) + self.buffer_ind
        if merge_idx > merge_idx_pycbc:
            bgn_idx_pycbc = 0
            bgn_idx_signal = merge_idx - merge_idx_pycbc
        else:
            bgn_idx_pycbc = merge_idx_pycbc - merge_idx
            bgn_idx_signal = 0
        if (n_signal - merge_idx) > (n_pycbc - merge_idx_pycbc):
            end_idx_pycbc = n_pycbc
            end_idx_signal = merge_idx + n_pycbc - merge_idx_pycbc
        else:
            end_idx_pycbc = merge_idx_pycbc + n_signal - merge_idx
            end_idx_signal = n_signal
        hSp[bgn_idx_signal:end_idx_signal] = hSp_tmp.data[bgn_idx_pycbc:end_idx_pycbc]
        hSc[bgn_idx_signal:end_idx_signal] = hSc_tmp.data[bgn_idx_pycbc:end_idx_pycbc]

        # rescale
        hSp = ratio * hSp
        hSc = ratio * hSc
        # rotation by psi i.e. source frame to SSB frame
        cos2psi = np.cos(2.0 * psi)
        sin2psi = np.sin(2.0 * psi)
        hp = hSp * cos2psi - hSc * sin2psi
        hc = hSp * sin2psi + hSc * cos2psi

        # remove first cycle to avoid tdi bugs
        est_T = ratio / f_lower
        zero_idx = (np.abs(hp[bgn_idx_signal : bgn_idx_signal + int(est_T / dt / 2)])).argmin()
        # ensure same +- sign
        if hp[bgn_idx_signal + zero_idx] * hp[bgn_idx_signal + zero_idx + 1] < 0:
            zero_idx += 1
        hp[bgn_idx_signal : bgn_idx_signal + zero_idx] = np.zeros(zero_idx)
        hc[bgn_idx_signal : bgn_idx_signal + zero_idx] = np.zeros(zero_idx)
        return hp + 1j * hc

    @staticmethod
    def m1_m2_from_M_q(M, q):
        """Compute individual masses from total mass and mass ratio.

        Choose m1 >= m2.

        Arguments:
            M {float} -- total mass
            q {mass ratio} -- mass ratio, 0.0< q <= 1.0

        Returns:
            (float, float) -- (mass_1, mass_2)
        """

        m1 = M / (1.0 + q)
        m2 = q * m1

        return m1, m2

    @staticmethod
    def m1_m2_from_M_Chirp_q(M_Chirp, q):
        q = 1 / q
        eta = q / (1 + q) ** 2
        M = M_Chirp * eta ** (-3 / 5)
        return MBHB.m1_m2_from_M_q(M, 1 / q)


class SOBBH(object):
    def __init__(self, f_min, dt, n_signal):
        self.f_min = f_min
        # self.T_buffer = T_buffer
        # self.buffer_ind = buffer_ind
        self.n_signal = n_signal
        self.data = None
        # self.delta_f = df
        self.delta_t = dt
        self.SOBHB = HpHc.type("my-sobhb", "SBBH", "IMRPhenomD")

    def __call__(self, pSOBBH, beta, lam, T=1, dt=10):
        # pSOBBH = self.get_par(idx)
        t = np.arange(0.0, pSOBBH["ObservationDuration"], dt)
        hp_s, hc_s = self.SOBHB.compute_hphc_td(t, pSOBBH)

        return hp_s + 1j * hc_s

    def get_par(self, idx):
        """
        from LDC code
        =====================================================
        Gal['ind'] = sub_src[:, 1]
        Gal['Redshift'] = sub_src[:, 2]
        Gal['Mass1'] = sub_src[:, 4]
        Gal['Mass2'] = sub_src[:, 5]
        Gal['InitialFrequency'] = sub_src[:, 6]
        Gal['EclipticLongitude'] = sub_src[:, 10]
        Gal['EclipticLatitude'] = 0.5*np.pi - sub_src[:, 9]
        Gal['Inclination'] = sub_src[:, 11]
        Gal['Polarization'] = sub_src[:, 12]
        Gal['Spin1'] = sub_src[:, 13]
        Gal['Spin2'] = sub_src[:, 14]
        Gal['AzimuthalAngleOfSpin1'] = np.zeros(N)
        Gal['AzimuthalAngleOfSpin2'] = np.zeros(N)
        Gal['PolarAngleOfSpin1'] = sub_src[:, 15]
        Gal['PolarAngleOfSpin2'] = sub_src[:, 16]
        Gal['InitialPhase'] = sub_src[:, 18]
        Gal['Approximant'] = 'PhenomD'
        ==================================================
        For ldc package HpHc waveform generator
        """
        return dict(
            {
                "Mass1": self.para_cat[idx, 4],
                "Spin1": self.para_cat[idx, 13],
                "Mass2": self.para_cat[idx, 5],
                "Spin2": self.para_cat[idx, 14],
                "EclipticLatitude": Constant.PI_2 - self.para_cat[idx, 9],
                "EclipticLongitude": self.para_cat[idx, 10],
                "Inclination": self.para_cat[idx, 11],
                "InitialFrequency": max(self.f_min, self.para_cat[idx, 6]),
                "InitialPhase": self.para_cat[idx, 18],
                "Polarization": self.para_cat[idx, 12],
                "Redshift": self.para_cat[idx, 2],
                "Distance": Cosmology.DL(self.para_cat[idx, 2], w=0)[0],
                "Cadence": self.delta_t,
                "ObservationDuration": self.n_signal * self.delta_t,
            }
        )

    @TimerLog(mylogger.logger, "Reading catalog from file")
    def read_catalog(self, cat_path):
        cat_path = Path(cat_path)
        if cat_path.suffix in [".txt", ".dat", ".OUT"]:
            par = np.loadtxt(cat_path)
        elif cat_path.suffix == ".npy":
            par = np.load(cat_path)
        else:
            raise NotImplementedError

        self.para_cat = par


if __name__ == "__main__":
    pass
