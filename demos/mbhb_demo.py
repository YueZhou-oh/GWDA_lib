from pathlib import Path
import sys

sys.path.append("..")
try:
    import cupy as xp
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

import h5py
import numpy as np
from src.data.tdi import TDIWaveformGen
from utils.lisa.cosmology import Cosmology


def main():
    use_gpu = False
    
    p = Path.cwd().parent
    orbit_file = p / "src/data/orbit/taiji-orbit.hdf5"

    tdiwg = TDIWaveformGen(T=2.0, use_gpu=use_gpu, det="Taiji", orbit_file=orbit_file)
    tdiwg.init_MBHB()

    M_tot = 6e5
    q = 0.5
    a1 = 0.2
    a2 = 0.4
    iota = np.pi / 3.0
    beta = np.pi / 4.0  # ecliptic latitude
    lam = np.pi / 5.0  # ecliptic longitude
    psi = np.pi / 3
    phi0 = 0.5
    distance = Cosmology.DL(5.5, w=0)[0]
    t_c = 1
    # --------

    bbh_xyz = tdiwg.mbhb_TDI(M_tot, q, a1, a2, phi0, distance, iota, psi, t_c, lam, beta)

    nX = tdiwg.gen_noise()
    nY = tdiwg.gen_noise()
    nZ = tdiwg.gen_noise()

    dataX = bbh_xyz[0] + nX
    dataY = bbh_xyz[1] + nY
    dataZ = bbh_xyz[2] + nZ

    data = np.vstack([tdiwg.sample_times, dataX, dataY, dataZ])
    with h5py.File("test_MBHB.hdf5", "w") as f:
        f.create_dataset("TDIdata", data=data)


if __name__ == "__main__":
    main()
