import sys

sys.path.append("..")
try:
    import cupy as xp
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

import h5py
import numpy as np

from tdc.tdi import TDIWaveformGen
from tdc.utils.constant import Constant
from tdc.utils.cosmology import Cosmology


def power_law_sgwb_Omega_gw(wg, a, b=2 / 3, f_ref=1e-3):
    h2_Omega_gw = 10 ** (a) * (wg.sample_frequencies / f_ref) ** (b)
    return h2_Omega_gw / (Constant.H0 / 100) ** 2


def Resp(f, L_arm):
    Loc = L_arm / Constant.C_SI
    tmp = 2 * xp.pi * f * Loc
    R = 16 * (xp.sin(tmp)) ** 2
    R = R * 4 * xp.sin(2 * tmp) ** 2
    R = (0.15 * R) / (1 + 0.6 * (tmp**2))
    R = R * tmp**2
    return R


def get_sgwb_psd(wg, Omega_gw):
    f = wg.sample_frequencies
    psd = xp.zeros(f.shape, dtype=xp.float64)
    h = Constant.H0 * 1e-2
    h2_Omega_gw = h * h * Omega_gw
    # h2_Omega_gw = 10**(alpha) * (self.sample_frequencies/f_star)**(n_t)
    a = (3 / (4 * xp.pi**2)) * ((0.1 / Constant.PC_SI) ** 2)

    fidx1 = int(wg.f_min / wg.delta_f)
    fidx2 = int(wg.f_max / wg.delta_f)
    psd[fidx1:fidx2] = (h2_Omega_gw[fidx1:fidx2]) / (f[fidx1:fidx2] ** 3)
    psd_tmp = a / h**2 * psd
    psd = psd_tmp * Resp(f, wg.L_arm)
    return psd


def gen_sgwb(wg, psd):
    """
    Generates SGWB from a psd
    """
    T_obs = wg.time_duration
    N = wg.Nt
    amp = xp.sqrt(0.25 * T_obs * psd)
    idx = xp.argwhere(psd == 0.0)
    amp[idx] = 0.0
    re = amp * xp.random.normal(0, 1, wg.Nf)
    im = amp * xp.random.normal(0, 1, wg.Nf)
    re[0] = 0.0
    im[0] = 0.0
    x = N * xp.fft.irfft(re + 1j * im) * wg.delta_f
    return x


def main():
    use_gpu = False
    orbit_file = "../../orbit/taiji-orbit.hdf5"

    tdiwg = TDIWaveformGen(T=2.0, use_gpu=use_gpu, det="Taiji", orbit_file=orbit_file)

    pl_Omega_gw_12 = power_law_sgwb_Omega_gw(tdiwg, -12)
    pl_sgwb_psd = get_sgwb_psd(tdiwg, pl_Omega_gw_12)
    pl_sgwb_signal = gen_sgwb(tdiwg, pl_sgwb_psd)

    nX = tdiwg.gen_noise()
    nY = tdiwg.gen_noise()
    nZ = tdiwg.gen_noise()

    dataX = pl_sgwb_signal + nX
    dataY = pl_sgwb_signal + nY
    dataZ = pl_sgwb_signal + nZ

    data = np.vstack([tdiwg.sample_times, dataX, dataY, dataZ])

    with h5py.File("test_SGWB.hdf5", "w") as f:
        f.create_dataset("TDIdata", data=data)


if __name__ == "__main__":
    main()
