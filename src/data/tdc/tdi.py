try:
    import cupy as xp
except (ImportError, ModuleNotFoundError):
    import numpy as xp

import functools

import numpy as np
from fastlisaresponse import ResponseWrapper

from .utils.constant import Constant

# from .utils.cosmology import Cosmology
from .utils.log_utils import MyLogger
from .waveform import AAK, GB, MBHB

# -------------------------------------
# constants
EPS = 1e-8

mylogger = MyLogger(__name__)


class TDIWaveformGen(object):
    def __init__(
        self,
        T=1,
        sample_rate=0.1,
        tdi_gen=2,
        det="Taiji",
        t0=10000,
        use_gpu=True,
        orbit_file="../orbit/taiji-orbit.hdf5",
    ):
        self.use_gpu = use_gpu
        self.tdi_gen = tdi_gen

        assert det in ["LISA", "Taiji"]
        if det == "LISA":
            # L_arm: Arm length [2.5e9]
            # sqSacc_level: Amplitude level of acceleration noise [3e-15]
            # sqSoms_level: Amplitude level of OMS noise [15e-12]
            self.L_arm = 2.5e9
            self.sqSacc = 3e-15
            self.sqSoms = 15e-12
            self.orbit_file = ""
        elif det == "Taiji":
            self.L_arm = 3e9
            self.sqSacc = 3e-15
            self.sqSoms = 8e-12
            self.orbit_file = orbit_file

        self.sampling_rate = sample_rate
        self.t0 = t0
        self.T = T  # [yr]
        self.time_duration = (int((T * Constant.YRSID_SI) / self.delta_t) + int((T * Constant.YRSID_SI) / self.delta_t) % 2) * self.delta_t  # [s]
        self.T_buffer = (self.time_duration + 2 * self.t0) / Constant.YRSID_SI  # [yr]
        self.buffer_ind = int(t0 / self.delta_t)
        self.n_signal = int(self.T_buffer * Constant.YRSID_SI / self.delta_t)
        assert self.Nt == self.n_signal - 2 * self.buffer_ind

        self.f_min = max(3.0e-5, EPS + 1.0 / self.time_duration)

        self.get_psd()

    @property
    def f_max(self):
        """Set the maximum frequency to half the sampling rate."""
        return self.sampling_rate / 2.0

    @f_max.setter
    def f_max(self, f_max):
        self.sampling_rate = 2.0 * f_max

    @property
    def delta_t(self):
        return 1.0 / self.sampling_rate

    @delta_t.setter
    def delta_t(self, delta_t):
        self.sampling_rate = 1.0 / delta_t

    @property
    def delta_f(self):
        return 1.0 / self.time_duration

    @delta_f.setter
    def delta_f(self, delta_f):
        self.time_duration = 1.0 / delta_f

    @property
    def Nt(self):
        return int(self.time_duration * self.sampling_rate)

    @property
    def Nf(self):
        return int(self.f_max / self.delta_f) + 1

    @property
    @functools.lru_cache()
    def sample_times(self):
        """Array of times at which waveforms are sampled."""
        return xp.linspace(0.0, self.time_duration, num=self.Nt, endpoint=False, dtype=xp.float32)

    @property
    @functools.lru_cache()
    def sample_frequencies(self):
        return xp.linspace(0.0, self.f_max, num=self.Nf, endpoint=True, dtype=xp.float32)

    def TDI(
        self,
        wave_gen,
        index_lambda,
        index_beta,
        remove_sky_coords=False,
        is_ecliptic_latitude=False,
        orbit_file=".",
        tdi_gen="2nd generation",
        t0=10000,
        tdi_chan="XYZ",
    ):
        """
        # 1st or 2nd or custom (see docs for custom)
        # tdi_gen = "2nd generation"
        # for GBWave
        # index_lambda = 6
        # index_beta = 7
        """
        # t0 = 10000.0  # time at which signal starts (chops off data at start of waveform where information is not correct)
        # order of the langrangian interpolation
        order = 25
        orbit_kwargs = dict(orbit_file=orbit_file)
        tdi_kwargs = dict(
            orbit_kwargs=orbit_kwargs,
            order=order,
            tdi=tdi_gen,
            tdi_chan=tdi_chan,
        )

        return ResponseWrapper(
            wave_gen,
            self.T_buffer,
            self.delta_t,
            index_lambda,
            index_beta,
            t0=t0,
            flip_hx=False,  # set to True if waveform is h+ - ihx
            use_gpu=self.use_gpu,
            remove_sky_coords=remove_sky_coords,  # True if the waveform generator does not take sky coordinates
            is_ecliptic_latitude=is_ecliptic_latitude,  # False if using polar angle (theta)
            remove_garbage=True,  # removes the beginning of the signal that has bad information
            **tdi_kwargs,
        )

    @staticmethod
    def PSD_Noise_components(fr, sqSnoise):
        [sqSacc_level, sqSoms_level] = sqSnoise
        # sqSacc_level: Amplitude level of acceleration noise [3e-15]
        # sqSoms_level: Amplitude level of OMS noise [15e-12]

        # ## Acceleration noise
        Sa_a = sqSacc_level**2 * (1.0 + (0.4e-3 / fr) ** 2) * (1.0 + (fr / 8e-3) ** 4)
        Sa_d = Sa_a * (2.0 * xp.pi * fr) ** (-4.0)
        Sa_nu = Sa_d * (2.0 * xp.pi * fr / Constant.C_SI) ** 2

        # ## Optical Metrology System
        Soms_d = sqSoms_level**2 * (1.0 + (2.0e-3 / fr) ** 4)
        Soms_nu = Soms_d * (2.0 * xp.pi * fr / Constant.C_SI) ** 2

        return [Sa_nu, Soms_nu]

    @staticmethod
    def PSD_Noise_X15(fr, sqSnoise, L_arm):
        [Sa_nu, Soms_nu] = TDIWaveformGen.PSD_Noise_components(fr, sqSnoise)
        phiL = 2 * xp.pi * fr * L_arm / Constant.C_SI
        return 16 * (xp.sin(phiL)) ** 2 * (Soms_nu + Sa_nu * (3 + xp.cos(2 * phiL)))

    @staticmethod
    def PSD_Noise_XY15(fr, sqSnoise, L_arm):
        [Sa_nu, Soms_nu] = TDIWaveformGen.PSD_Noise_components(fr, sqSnoise)
        phiL = 2 * xp.pi * fr * L_arm / Constant.C_SI
        return -8 * (xp.sin(phiL)) ** 2 * xp.cos(phiL) * (Soms_nu + 4 * Sa_nu)

    @staticmethod
    def PSD_Noise_X20(fr, sqSnoise, L_arm):
        [Sa_nu, Soms_nu] = TDIWaveformGen.PSD_Noise_components(fr, sqSnoise)
        phiL = 2 * xp.pi * fr * L_arm / Constant.C_SI
        return 64 * (xp.sin(phiL)) ** 2 * (xp.sin(2 * phiL)) ** 2 * (Soms_nu + Sa_nu * (3 + xp.cos(2 * phiL)))

    def get_psd(
        self,
    ):
        fr = xp.linspace(0.0, self.f_max, num=self.Nf, endpoint=True)
        psd = xp.zeros(self.Nf)
        if self.tdi_gen == 1:
            psd[1:] = self.PSD_Noise_X15(fr[1:], [self.sqSacc, self.sqSoms], self.L_arm)
        else:
            psd[1:] = self.PSD_Noise_X20(fr[1:], [self.sqSacc, self.sqSoms], self.L_arm)
        self.psd = psd

    def gen_noise(self):
        """
        Generates noise from a psd
        """
        T_obs = self.time_duration
        psd = self.psd
        N = self.Nt  # the total number of time samples
        df = self.delta_f

        amp = xp.sqrt(0.25 * T_obs * psd)
        idx = xp.argwhere(psd == 0.0)
        amp[idx] = 0.0
        re = amp * xp.random.normal(0, 1, self.Nf)
        im = amp * xp.random.normal(0, 1, self.Nf)
        re[0] = 0.0
        im[0] = 0.0
        x = N * xp.fft.irfft(re + 1j * im) * df
        return x

    def init_EMRI(
        self,
    ):
        """
        tdi_wave = aak_AET(M, mu, a, p0, e0,
                            Y0, qS, phiS, qK, phiK, dist,
                            Phi_phi0, Phi_theta0, Phi_r0, mich)
        """
        mylogger.logger.info("Init AAK")
        self.aak = AAK(self.use_gpu, self.n_signal)
        self.aak_TDI = self.TDI(
            self.aak,
            index_lambda=7,
            index_beta=6,
            orbit_file=self.orbit_file,
            is_ecliptic_latitude=False,
            remove_sky_coords=False,
            t0=self.t0,
        )

    def init_GB(self, VGB=True):
        mylogger.logger.info("Init GB")
        self.gb = GB(self.use_gpu, VGB=VGB)
        self.gb_TDI = self.TDI(
            self.gb,
            index_lambda=6,
            index_beta=7,
            orbit_file=self.orbit_file,
            is_ecliptic_latitude=True,
            remove_sky_coords=True,
            t0=self.t0,
        )

    def init_MBHB(self):
        mylogger.logger.info("Init MBHB")
        self.mbhb = MBHB(self.f_min, self.T_buffer, self.buffer_ind)
        self.mbhb_TDI = self.TDI(
            self.mbhb,
            index_lambda=9,
            index_beta=10,
            orbit_file=self.orbit_file,
            is_ecliptic_latitude=True,
            remove_sky_coords=True,
            t0=self.t0,
        )
