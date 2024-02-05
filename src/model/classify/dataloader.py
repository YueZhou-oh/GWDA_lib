# mypy: ignore-errors
# type: ignore
import logging

import numpy as np
import torch

# import h5py
from torch.utils.data import DataLoader, Dataset

from ...utils.io.hdf5_wfd import load_waveform, save_waveform

log = logging.getLogger(__name__)

class TinyEMRIDataset(object):
    def __init__(self, DIR, fn):
        super().__init__()
        # self.train = train
        self.data = {
            "train": {"signal": [], "noise": []},
            "test": {"signal": [], "noise": []},
        }
        self.label = {"train": [], "test": []}
        self.params = {"train": [], "test": []}

        log.info("Loading data from {}/{}".format(DIR, fn))
        load_waveform(data=[self.data, self.params], DIR=DIR, data_fn=fn)
    
    def save(self, DIR, fn):
        save_waveform(data=[self.data, self.params], DIR=DIR, data_fn=fn)


class EMRIDatasetTorch(Dataset):
    def __init__(self, wfd, train=True):
        super().__init__()
        self.wfd = wfd
        self.train = train
        self.type_str = "train" if self.train else "test"
        self.length = self.wfd.data[self.type_str]["signal"].shape[-1]
        self.n_signal = self.wfd.data[self.type_str]["signal"].shape[0]

    def __len__(self):
        return (
            self.wfd.data[self.type_str]["noise"].shape[0]
            + self.wfd.data[self.type_str]["signal"].shape[0]
        )

    def __getitem__(self, idx):
        if idx < self.n_signal:
            data = (
                self.wfd.data[self.type_str]["signal"][idx][:, ::4]
            )
            label = 1
        else:
            data = self.wfd.data[self.type_str]["noise"][
                idx - self.n_signal
            ][:, ::4]
            label = 0
        if np.isnan(data).any():
            raise ValueError("NaN in data")

        return (
            torch.tensor(idx, dtype=torch.long),
            torch.from_numpy(data).float(),
            torch.tensor(label, dtype=torch.long),
        )
