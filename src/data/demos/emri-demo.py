import sys

sys.path.append("..")
try:
    import cupy as xp
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

import h5py
import numpy as np

from tdc.tdi import TDIWaveformGen


def main():
    use_gpu = False
    orbit_file = "../tdc/orbit/taiji-orbit.hdf5"

    tdiwg = TDIWaveformGen(T=2.0, use_gpu=use_gpu, det="Taiji", orbit_file=orbit_file)
    tdiwg.init_EMRI()

    M = 1e6
    mu = 10
    a = 0.3
    p0 = 15
    e0 = 0.6
    iota0 = 0.7
    Y0 = np.cos(iota0)

    Phi_phi0 = 0.0
    Phi_theta0 = 0.0
    Phi_r0 = 0

    qS = np.pi / 4
    phiS = np.pi / 2

    # SMBH spin direction
    qK = 1e-6
    phiK = 1e-6

    # D_L
    dist = 1
    # ...............
    mich = False

    emri_xyz = tdiwg.aak_TDI(M, mu, a, p0, e0, Y0, qS, phiS, qK, phiK, dist, Phi_phi0, Phi_theta0, Phi_r0, mich)

    nX = tdiwg.gen_noise()
    nY = tdiwg.gen_noise()
    nZ = tdiwg.gen_noise()

    dataX = emri_xyz[0] + nX
    dataY = emri_xyz[1] + nY
    dataZ = emri_xyz[2] + nZ

    data = np.vstack([tdiwg.sample_times, dataX, dataY, dataZ])
    with h5py.File("test-EMRI.hdf5", "w") as f:
        f.create_dataset("TDIdata", data=data)


if __name__ == "__main__":
    main()
