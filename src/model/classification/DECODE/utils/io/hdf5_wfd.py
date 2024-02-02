# type: ignore
from pathlib import Path

import h5py

# mylogger = MyLogger(__name__)


def save_waveform(data=None, DIR=".", data_fn="waveform_dataset.hdf5"):
    """Save waveform dataset to hdf5 file.

    Parameters
    ----------
    data : list, optional
        Dataset to save, [waveform_dataset, waveform_params]
    DIR : str, optional
        Specified directory to save waveform dataset, by default '.'
    data_fn : str, optional
        Specified file name to save waveform dataset, by default 'waveform_dataset.hdf5'
    """
    if data is None:
        print("No data to save!")
        return
    wfd, wfp = data
    p = Path(DIR)
    p.mkdir(parents=True, exist_ok=True)
    # mylogger.logger.info("Saving waveforms...")
    f_data = h5py.File(p / data_fn, "w")

    data_name = "0"
    for i in wfd.keys():
        for j in wfd[i].keys():
            data_name = i + "_" + j
            f_data.create_dataset(
                data_name,
                data=wfd[i][j],
                compression="gzip",
                compression_opts=9,
            )

    for i in wfp.keys():
        data_name = i + "_" "par"
        f_data.create_dataset(data_name, data=wfp[i], compression="gzip", compression_opts=9)
    f_data.close()


def load_waveform(data=None, DIR=".", data_fn="waveform_dataset.hdf5"):
    """load waveform dataset from hdf5 file.

    Parameters
    ----------
    DIR : str, optional
        Specified directory to the waveform dataset, by default '.'
    data_fn : str, optional
        Specified file name to the waveform dataset, by default 'waveform_dataset.hdf5'
    """
    # Load data from HDF5 file
    if data is None:
        print("No data to save!")
        return
    wfd, wfp = data
    p = Path(DIR)
    f_data = h5py.File(p / data_fn, "r")
    # mylogger.logger.info("Loading waveforms...")
    data_name = "0"
    # Load parameters
    for i in wfp.keys():
        data_name = i + "_" "par"
        try:
            wfp[i] = f_data[data_name][()]
        except KeyError:
            print("Could not find dataset with name %s" % data_name)
    # Load waveform data
    for i in wfd.keys():
        for j in wfd[i].keys():
            data_name = i + "_" + j
            try:
                wfd[i][j] = f_data[data_name][()]
            except KeyError:
                print("Could not find dataset with name %s" % data_name)
    f_data.close()
