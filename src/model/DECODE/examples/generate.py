import sys
sys.path.append("..")
import os
from pathlib import Path
import hydra
import numpy as np
import cupy as xp

from rich import print
from scipy.signal import welch
from DECODE.emridataset import EMRIDataset
from emridetection.utils.io.hdf5_wfd import save_waveform
from emridetection.data.dataloader import TinyEMRIDataset
from tqdm import tqdm




def gen_amp_ratio_asd(config, ratios=[0.5, 1]):
    ed = EMRIDataset(config)
    data_dir = "/workspace/zhty/EMRI_Detection/emridetection/datasets"
    data_fn = "emri_asd_v0.5_50_120_1yr.hdf5"
    wfd = TinyEMRIDataset(data_dir, data_fn)
    test_params_array = wfd.params["test"]
    wfd_all = [TinyEMRIDataset(data_dir, data_fn) for _ in range(len(ratios))]
    for i in tqdm(range(test_params_array.shape[0])):
        p = list(test_params_array[i])
        sig = ed.aak_tdi(*p)
        # rho = xp.sqrt(ed.get_snr(sig["A"]) ** 2 + ed.get_snr(sig["E"]) ** 2)
        ratio = [1, 1]
        for j in range(ed.n_channels):
            sig_j = sig[ed.tdi_chan[j]]
            # sig_j = (
            #     sig_j * test_params_array[i][ed.param_idx["snr"]] / rho
            # )
            f, sig_j_psd = welch(sig_j.get(), fs=ed.sampling_rate, nperseg=ed.Nt / 8, noverlap=0.2)
            psd_f = np.interp(f, ed.sample_frequencies.get(), ed.psd.get())
            ratio[j] = max(sig_j_psd[1:] / psd_f[1:])

            noise = ed.gen_noise()
            # data = sig + noise
            # data_w = self.whiten_data(data)
            noise_w = ed.whiten_data(noise)
            sig_j_w = ed.whiten_data(sig_j)
            noise_asd = ed.gen_log_asd(data=noise_w, ovlp=0.2)
            sig_j_asd = ed.gen_log_asd(data=sig_j_w, ovlp=0.2)
            for k, r in enumerate(ratios):
                asd = (10**noise_asd) + (10**sig_j_asd) * (r / ratio[j])
                log_asd = np.log10(asd)
                # log_asd = ed.gen_log_asd(data=data_w_k, ovlp=0.2)
                log_asd = ed.standardize_data(log_asd)
                wfd_all[k].data["test"]["test"][i, j] = log_asd

    data_all_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets/amp_ratio")
    data_all_dir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(ratios):
        fn_k = f"emri_asd_v0.5_50_120_1yr_amp_ratio_{r}.hdf5"
        wfd_all[k].save(data_all_dir, fn_k)


def gen_snr_asd(config, snrs=[50, 60]):
    ed = EMRIDataset(config)
    data_dir = "/workspace/zhty/EMRI_Detection/emridetection/datasets"
    data_fn = "emri_asd_v0.5_50_120_1yr.hdf5"
    wfd = TinyEMRIDataset(data_dir, data_fn)
    test_params_array = wfd.params["test"]
    wfd_all = [TinyEMRIDataset(data_dir, data_fn) for _ in range(len(snrs))]
    for i in tqdm(range(test_params_array.shape[0])):
        p = list(test_params_array[i])
        sig = ed.aak_tdi(*p)
        rho = xp.sqrt(ed.get_snr(sig["A"]) ** 2 + ed.get_snr(sig["E"]) ** 2)
        # ratio = [1, 1]
        for j in range(ed.n_channels):
            sig_j = sig[ed.tdi_chan[j]]
            noise = ed.gen_noise()
            # data = sig + noise
            # data_w = self.whiten_data(data)
            noise_w = ed.whiten_data(noise)
            sig_j_w = ed.whiten_data(sig_j)
            noise_asd = ed.gen_log_asd(data=noise_w, ovlp=0.2)
            sig_j_asd = ed.gen_log_asd(data=sig_j_w, ovlp=0.2)
            for k, r in enumerate(snrs):
                # sig_j = (
                #     sig_j * r / rho
                # )
                asd = (10**noise_asd) + (10**sig_j_asd) * (r / rho)
                log_asd = np.log10(asd)
                # log_asd = ed.gen_log_asd(data=data_w_k, ovlp=0.2)
                log_asd = ed.standardize_data(log_asd)
                wfd_all[k].data["test"]["test"][i, j] = log_asd

    data_all_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets/amp_ratio")
    data_all_dir.mkdir(parents=True, exist_ok=True)
    for k, r in enumerate(snrs):
        fn_k = f"emri_asd_v0.5_1yr_snr_{r}.hdf5"
        wfd_all[k].save(data_all_dir, fn_k)


def calac_amp_ratio(ed, test_params_array):
    amp_ratio = np.zeros(1000)
    for i in tqdm(range(1000)):
        p = list(test_params_array[i])
        # print(p[0])
        # break
        sig = ed.aak_tdi(*p)
        rho = xp.sqrt(ed.get_snr(sig["A"]) ** 2 + ed.get_snr(sig["E"]) ** 2)
        ratio = [1, 1]
        for j in range(ed.n_channels):
            sig_j = sig[ed.tdi_chan[j]]
            sig_j = sig_j * test_params_array[i][ed.param_idx["snr"]] / rho
            f, sig_j_psd = welch(sig_j.get(), fs=ed.sampling_rate, nperseg=ed.Nt / 8, noverlap=0.2)
            psd_f = np.interp(f, ed.sample_frequencies.get(), ed.psd.get())
            # print(f[:5])
            # print(psd_f[:5])
            ratio[j] = max(sig_j_psd[1:] / psd_f[1:])
        amp_ratio[i] = max(ratio)
    return np.sqrt(amp_ratio)


@hydra.main(version_base="1.2", config_path="../configs", config_name="dataset")
def main2(config):
    ed = EMRIDataset(config)
    data_fn = "emri_asd_v0.5_50_120_1yr.hdf5"
    wfd = TinyEMRIDataset("/workspace/zhty/EMRI_Detection/emridetection/datasets", data_fn)
    test_params_array = wfd.params["test"]
    amp_ratio = calac_amp_ratio(ed, test_params_array)
    np.save("emri_asd_v0.5_50_120_1yr_amp_ratio.npy", amp_ratio)


@hydra.main(version_base="1.2", config_path="../configs", config_name="dataset")
def main(config):
    gen_amp_ratio_asd(config, ratios=[0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])


if __name__ == "__main__":
    main()
