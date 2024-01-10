import sys

sys.path.append("..")

try:
    import cupy as xp
except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from tdc.tdi import TDIWaveformGen
from tdc.utils.log_utils import MyLogger, TimerLog

mylogger = MyLogger(__name__)


@TimerLog(mylogger.logger, "Generating data.......")
def gen_sobbh_data(tdiwg, cat_path, h5_path):
    cat_path = Path(cat_path)
    tdiwg.sobbh.read_catalog(cat_path)
    tdiwg.sobbh.data = xp.zeros([3, tdiwg.Nt])
    for i in tqdm(range(tdiwg.sobbh.para_cat.shape[0])):
        para = tdiwg.sobbh.get_par(i)
        wave = tdiwg.sobbh_TDI(para, para["EclipticLatitude"], para["EclipticLongitude"])
        wave = xp.array(wave)
        tdiwg.sobbh.data += wave

    data = xp.vstack([tdiwg.sample_times, tdiwg.sobbh.data])
    mylogger.logger.info("Writing to file----> {}".format(h5_path))
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("TDIdata", data=data)


def main():
    use_gpu = False
    orbit_file = "../tdc/orbit/taiji-orbit.hdf5"
    sobbh_cat = "../catalog/test-SOBBH-catalog.dat"
    tdiwg = TDIWaveformGen(use_gpu=use_gpu, T=1.0, det="Taiji", orbit_file=orbit_file)
    tdiwg.init_SOBBH()

    gen_sobbh_data(tdiwg, sobbh_cat, "test-SOBBH.hdf5")


if __name__ == "__main__":
    main()
