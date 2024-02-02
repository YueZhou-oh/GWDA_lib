# mypy: ignore-errors
# type: ignore
import time

import numpy as np
import torch

try:
    import cupy as xp

    # import cupyx.scipy.signal as xpx_signal
    # from cupyx.scipy.signal import welch as xpx_welch
    # import cupyx.scipy.interpolate as xpx_interpolate
except (ImportError, ModuleNotFoundError):
    import numpy as xp

# import functools
import itertools
import logging
import os

import fast_matched_filter as fmf

# import h5py
from bidict import bidict
from rich import print
from scipy.signal import welch
from tqdm import tqdm

# from emridetection.gw.psd import PSD
from emridetection.data.gwdataset import GWDataset
from emridetection.gw.waveform import AAK, AAK_TDI

# from emridetection.utils.io.hdf5_wfd import load_waveform, save_waveform
from emridetection.utils.lisa.constant import Constant
from emridetection.utils.lisa.cosmology import Cosmology
from emridetection.utils.logger import Timer

# from pathlib import Path


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


log = logging.getLogger(__name__)

# EPS = 1e-8


class EMRIDataset(GWDataset):
    def __init__(
        self,
        config=None,
        T=1,
        dt=10,
        tdi_gen="2nd generation",
        tdi_chan="AE",
        det="LISA",
        t0=10000,
        use_gpu=True,
        orbit_file="/workspace/zhaoty/emridetection/emridetection/gw/orbit/h5/orbit.hdf5",
    ):
        super().__init__(config=config)
        # parameter range
        # Pn5AAK(M, mu, a, p0, e0, Y0, dist,
        # qS, phiS, qK, phiK,
        # Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0,
        # mich=mich, dt=dt, T=T)

        param_idx = bidict(
            M=0,
            mu=1,
            a=2,
            p0=3,
            e0=4,
            Y0=5,
            snr=6,
            qS=7,
            phiS=8,
            qK=9,
            phiK=10,
            Phi_phi0=11,
            Phi_theta0=12,
            Phi_r0=13,
        )

        nparams = 14
        self.param_idx = param_idx
        self.nparams = nparams
        self.idx_param = param_idx.inverse
        self.par_range = dict(
            M=[5, 8],  # log10(M) [Msun]
            mu=[10, 50],
            a=[1e-3, 0.99],
            p0=[20, 30],
            e0=[1e-3, 0.7],
            Y0=[-0.98, 0.99],
            snr=[50, 120],
            qS=[0, Constant.PI],
            phiS=[0, 2 * Constant.PI],
            qK=[0, Constant.PI],
            phiK=[0, 2 * Constant.PI],
            Phi_phi0=[0, 2 * Constant.PI],
            Phi_theta0=[0, Constant.PI],
            Phi_r0=[0, 2 * Constant.PI],
        )

        if config:
            self.update_par_range()
        else:
            self.use_gpu = use_gpu
            self.tdi_gen = tdi_gen
            self.tdi_chan = tdi_chan
            self.delta_t = dt
            self.t0 = t0
            self.T = T  # [yr]
            self.det = det
            self.orbit_file = orbit_file

        self.aak = AAK(self.use_gpu, self.n_signal)
        self.aak_tdi = AAK_TDI(
            self.aak,
            self.T_buffer,
            self.delta_t,
            self.param_idx["phiS"],
            self.param_idx["qS"],
            remove_sky_coords=False,
            is_ecliptic_latitude=False,
            orbit_file=self.orbit_file,
            order=25,
            tdi_gen=self.tdi_gen,
            t0=self.t0,
            tdi_chan=self.tdi_chan,
        )

    def mfcnn_template(self, dims=[0, 4], n_grid=[8, 8], N=16384):
        assert len(dims) == len(n_grid)

        n = np.prod(n_grid)

        default_params = [
            1e6,  # M
            10,  # mu
            0.5,  # a
            20,  # p0
            0.5,  # e0
            0.5,  # Y0
            1,  # D_L (snr_opt)
            1e-3,  # qS
            1e-3,  # phiS
            1e-3,  # qK
            1e-3,  # phiK
            0,  # Phi_phi0
            0,  # Phi_theta0
            0,  # Phi_r0
        ]

        param_array = np.stack([default_params] * n)

        # Use list comprehension to generate starts and ends
        starts = [self.par_range[self.idx_param[i]][0] for i in dims]
        ends = [self.par_range[self.idx_param[i]][1] for i in dims]

        # Use list comprehension to generate the grid_1d
        grid_1d = [
            np.linspace(start, end, num)
            for start, end, num in zip(starts, ends, n_grid)
        ]

        self.waveform_dataset["train"]["signal"] = np.zeros(
            (n, self.n_channels, self.Nt)
        )

        # Use list comprehension instead of for loop to generate p_grid and then update param_array
        log.info("Generating parameter grid...")
        for i, par_idx in tqdm(
            enumerate(itertools.product(*grid_1d)), total=n, disable=True
        ):
            new_params = default_params.copy()
            for j, dim in enumerate(dims):
                new_params[dim] = par_idx[j]
            param_array[i] = new_params
        # 10^M
        param_array[:, self.param_idx["M"]] = (
            10 ** param_array[:, self.param_idx["M"]]
        )
        self.waveform_params["train"] = param_array

        # Reduce the repetitive calls to self.waveform_dataset and self.waveform_params
        waveform_dataset_train_signal = self.waveform_dataset["train"][
            "signal"
        ]
        waveform_params_train = self.waveform_params["train"]
        waveform_params_snr = np.zeros((n, self.n_channels))

        log.info("Generating signal")
        for i in tqdm(range(waveform_dataset_train_signal.shape[0])):
            p = list(waveform_params_train[i])
            try:
                emri_aet = self.aak_tdi(*p)
            except Exception as e:
                print(p)
                raise e

            for j in range(self.n_channels):
                sig = emri_aet[self.tdi_chan[j]]
                rho = self.get_snr(sig)
                sig_w = self.whiten_data(sig)
                # sig_w = self.standardize_data(sig_w)
                # rho = self.get_inner_product_w(sig_w, sig_w)
                sig_w /= rho
                waveform_params_snr[i, j] = xp.sqrt(rho)
                # waveform_params_snr[i, j] = rho
                waveform_dataset_train_signal[i, j] = sig_w.get()
        # stack snr to waveform_params
        self.waveform_params["train"] = np.hstack(
            (waveform_params_train, waveform_params_snr)
        )
        # cut waveform to 16384 * 8
        cut_win = int((self.Nt - N) / 2 + 0.5)
        # print(cut_win)
        # exit()
        if cut_win > 0:
            self.waveform_dataset["train"][
                "signal"
            ] = waveform_dataset_train_signal[:, :, cut_win:-cut_win]
        else:
            self.waveform_dataset["train"][
                "signal"
            ] = waveform_dataset_train_signal

        # print("Here")

    def grid_M_snr(self, dims=[0, 6], n_grid=[30, 30]):
        assert len(dims) == len(n_grid)

        n = np.prod(n_grid)

        default_params = [
            1e6,  # M
            10,  # mu
            0.5,  # a
            20,  # p0
            0.5,  # e0
            0.5,  # Y0
            1,  # D_L (snr_opt)
            1e-3,  # qS
            1e-3,  # phiS
            1e-3,  # qK
            1e-3,  # phiK
            0,  # Phi_phi0
            0,  # Phi_theta0
            0,  # Phi_r0
        ]

        param_array = np.stack([default_params] * n)

        # Use list comprehension to generate starts and ends
        starts = [self.par_range[self.idx_param[i]][0] for i in dims]
        ends = [self.par_range[self.idx_param[i]][1] for i in dims]

        # Use list comprehension to generate the grid_1d
        grid_1d = [
            np.linspace(start, end, num)
            for start, end, num in zip(starts, ends, n_grid)
        ]

        self.waveform_dataset["train"]["signal"] = np.zeros(
            (n, self.n_channels, self.Nt)
        )

        # Use list comprehension instead of for loop to generate p_grid and then update param_array
        log.info("Generating parameter grid...")
        for i, par_idx in tqdm(
            enumerate(itertools.product(*grid_1d)), total=n, disable=True
        ):
            new_params = default_params.copy()
            for j, dim in enumerate(dims):
                new_params[dim] = par_idx[j]
            param_array[i] = new_params
        # 10^M
        param_array[:, self.param_idx["M"]] = (
            10 ** param_array[:, self.param_idx["M"]]
        )
        return param_array

    def sampling_parameters(self, n_train, n_test):
        """uniform sampling waveform parameters by self.par_ramge

        Args:
            n_train (int): number of samples
            n_test (int): number of samples
        """
        log.info("Sampling parameters")
        # ramdom param_array
        # malloc memory on cpu
        param_array = np.random.uniform(
            0, 1, size=(n_train + n_test, self.nparams)
        )
        # rescale parameters
        for pkey, idx in self.param_idx.items():
            param_array[:, idx] = self.par_range[pkey][0] + param_array[
                :, idx
            ] * (self.par_range[pkey][1] - self.par_range[pkey][0])
        # 10^M
        param_array[:, self.param_idx["M"]] = (
            10 ** param_array[:, self.param_idx["M"]]
        )
        # fix mu to 30
        param_array[:, self.param_idx["mu"]] = 10.0
        self.waveform_params["train"] = param_array[:n_train]
        self.waveform_params["test"] = param_array[n_train:]

    def sampling_one_parameter(
        self,
    ):
        """uniform sampling waveform parameters by self.par_ramge

        Args:
            n_train (int): number of samples
            n_test (int): number of samples
        """
        param = np.random.uniform(0, 1, size=self.nparams)
        # rescale parameters
        for pkey, idx in self.param_idx.items():
            param[idx] = self.par_range[pkey][0] + param[idx] * (
                self.par_range[pkey][1] - self.par_range[pkey][0]
            )
        # 10^M
        param[self.param_idx["M"]] = 10 ** param[self.param_idx["M"]]
        return param

    def update_par_range(
        self,
    ):
        for key in self.config["parameters"].keys():
            self.par_range[key] = self.config["parameters"][key]

    def gen_log_asd(
        self,
        data,
        nperseg=None,
        ovlp=0,
    ):
        """generate log asd for each channel"""
        if nperseg is None:
            nperseg = self.Nt / 8
        if isinstance(data, xp.ndarray):
            data_in = data.copy().get()
        else:
            data_in = data.copy()
        f, psd_data = welch(
            data_in,
            fs=self.sampling_rate,
            window="tukey",
            noverlap=ovlp * nperseg,
            nperseg=nperseg,
        )
        # log uniform sampling
        log_psd = np.interp(self.sample_log_frequencies.get(), f, psd_data)
        return np.log10(np.sqrt(log_psd))

    @Timer()
    def generate_sig_dataset(self, n_train, n_test):
        """Generate dataset by sampling parameters
        n_train, n_test are the number of samples contain signal
        total number of samples are 2 * (n_train + n_test)

        Args:
            n_train (int): number of samples
            n_test (int): number of samples
        """

        self.sampling_parameters(n_train, n_test)

        # malloc memory on cpu
        self.waveform_dataset["train"]["signal"] = np.zeros(
            (n_train, self.n_channels, self.Nt)
        )
        self.waveform_dataset["test"]["signal"] = np.zeros(
            (n_test, self.n_channels, self.Nt)
        )
        self.waveform_dataset["train"]["noise"] = np.zeros(
            (n_train, self.n_channels, self.Nt)
        )
        self.waveform_dataset["test"]["noise"] = np.zeros(
            (n_test, self.n_channels, self.Nt)
        )

        log.info("Generating noise")
        for key in self.waveform_dataset.keys():
            for i in self.waveform_dataset[key].keys():
                for j in tqdm(range(self.waveform_dataset[key][i].shape[0])):
                    for k in range(self.n_channels):
                        noise = self.gen_noise()
                        self.waveform_dataset[key][i][j, k] = noise.get()

        log.info("Generating signal")
        for key in self.waveform_dataset.keys():
            for i in tqdm(
                range(self.waveform_dataset[key]["signal"].shape[0])
            ):
                # args, kwargs = self.format_params(self.waveform_params[key][i])
                # emri_aet = self.aak_tdi(*args, **kwargs)
                p = list(self.waveform_params[key][i])
                try:
                    emri_aet = self.aak_tdi(*p)
                except Exception as e:
                    # print(dict.fromkeys(self.param_idx.keys(), p))
                    print(p)
                    raise e

                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    rho = self.get_snr(sig)
                    sig = sig * p[self.param_idx["snr"]] / rho
                    # calculate on gpu then get()
                    self.waveform_dataset[key]["signal"][i, j] = sig.get()

    @Timer()
    def generate_clsf_dataset(self, n_train, n_test):
        """Generate dataset by sampling parameters
        n_train, n_test are the number of samples contain signal
        total number of samples are 2 * (n_train + n_test)

        Args:
            n_train (int): number of samples
            n_test (int): number of samples
        """

        self.sampling_parameters(n_train, n_test)

        # malloc memory on cpu
        self.waveform_dataset["train"]["signal"] = np.zeros(
            (n_train, self.n_channels, self.Nt)
        )
        self.waveform_dataset["test"]["signal"] = np.zeros(
            (n_test, self.n_channels, self.Nt)
        )
        self.waveform_dataset["train"]["noise"] = np.zeros(
            (n_train, self.n_channels, self.Nt)
        )
        self.waveform_dataset["test"]["noise"] = np.zeros(
            (n_test, self.n_channels, self.Nt)
        )

        log.info("Generating noise")
        for key in self.waveform_dataset.keys():
            for i in self.waveform_dataset[key].keys():
                for j in tqdm(range(self.waveform_dataset[key][i].shape[0])):
                    for k in range(self.n_channels):
                        noise = self.gen_noise()
                        self.waveform_dataset[key][i][j, k] = noise.get()

        log.info("Generating signal")
        for key in self.waveform_dataset.keys():
            for i in tqdm(
                range(self.waveform_dataset[key]["signal"].shape[0])
            ):
                # args, kwargs = self.format_params(self.waveform_params[key][i])
                # emri_aet = self.aak_tdi(*args, **kwargs)
                p = list(self.waveform_params[key][i])
                try:
                    emri_aet = self.aak_tdi(*p)
                except Exception as e:
                    # print(dict.fromkeys(self.param_idx.keys(), p))
                    print(p)
                    raise

                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    rho = self.get_snr(sig)
                    sig = sig * p[self.param_idx["snr"]] / rho
                    # calculate on gpu then get()
                    self.waveform_dataset[key]["signal"][i, j] += sig.get()

    @Timer()
    def generate_clsf_dataset_v2(
        self, n_train, n_test, trg_len=2103876, whitten=False
    ):
        assert trg_len < self.Nt
        n_samples = n_train + n_test
        cut_win = int((self.Nt - trg_len) / 2 + 0.5)
        assert self.Nt - 2 * cut_win == trg_len
        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        snr_factor = xp.sqrt(self.T / 2)
        # snr_factor = 1.0

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nt), dtype=np.float32
        )
        noise_dataset = np.zeros(
            (n_samples * 2, self.n_channels, self.Nt), dtype=np.float32
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    rho = self.get_snr(sig)
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    sig_dataset[i, j] = sig.get()
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples * 2)):
            for j in range(self.n_channels):
                noise_dataset[i, j] = self.gen_noise().get()

        if whitten:
            log.info("Whitenning data")
            for i in tqdm(range(n_samples)):
                for j in range(self.n_channels):
                    sig_dataset[i, j] = self.whiten_data(
                        xp.asarray(sig_dataset[i, j])
                    ).get()
                    noise_dataset[i, j] = self.whiten_data(
                        xp.asarray(noise_dataset[i, j])
                    ).get()
                    noise_dataset[i + n_samples, j] = self.whiten_data(
                        xp.asarray(noise_dataset[i + n_samples, j])
                    ).get()

        self.waveform_dataset["train"]["signal"] = sig_dataset[
            :n_train, :, cut_win:-cut_win
        ]
        self.waveform_dataset["test"]["signal"] = sig_dataset[
            n_train:, :, cut_win:-cut_win
        ]
        self.waveform_dataset["train"]["noise"] = noise_dataset[
            : 2 * n_train, :, cut_win:-cut_win
        ]
        self.waveform_dataset["test"]["noise"] = noise_dataset[
            2 * n_train :, :, cut_win:-cut_win
        ]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    @Timer()
    def generate_asd_clsf_dataset(
        self,
        n_train,
        n_test,
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        # snr_factor = xp.sqrt(self.T / 2)
        snr_factor = 1.0

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_w = self.whiten_data(data)
                    log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                    log_asd = self.standardize_data(log_asd)
                    sig_dataset[i, j] = log_asd
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise_w = self.whiten_data(noise)
                log_asd = self.gen_log_asd(data=noise_w, ovlp=0.2)
                log_asd = self.standardize_data(log_asd)
                noise_dataset[i, j] = log_asd

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    @Timer()
    def generate_asd_clsf_z_dataset(
        self,
        n_train,
        n_test,
        z,
        fix_param={0: 1e5},
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        # snr_factor = xp.sqrt(self.T / 2)
        snr_factor = 1.0
        
        DL = Cosmology.DL(z, w=0)[0] / 1000 # Gpc
        print(f"DL = {DL} Gpc for z = {z}")
        fix_param[self.param_idx["snr"]] = DL
        # print fix_param
        print("=*=" * 10)
        print("Fix parameters:")
        for k, v in fix_param.items():
            if k == self.param_idx["snr"]:
                print(f"DL = {v} Gpc")
            else:
                print(f"{self.idx_param[k]} = {v}")
        print("=*=" * 10)

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            # set DL
            # params[self.param_idx["snr"]] = DL
            # fix parameter
            for k, v in fix_param.items():
                params[k] = v

            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    noise = self.gen_noise()
                    data = sig + noise
                    data_w = self.whiten_data(data)
                    log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                    log_asd = self.standardize_data(log_asd)
                    sig_dataset[i, j] = log_asd
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise_w = self.whiten_data(noise)
                log_asd = self.gen_log_asd(data=noise_w, ovlp=0.2)
                log_asd = self.standardize_data(log_asd)
                noise_dataset[i, j] = log_asd

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    @Timer()
    def generate_asd_clsf_snr_dataset(
        self,
        n_train,
        n_test,
        fix_param={0: 1e5},
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        # snr_factor = xp.sqrt(self.T / 2)
        snr_factor = 1.0
        
        # DL = Cosmology.DL(z, w=0)[0] / 1000 # Gpc
        # print(f"DL = {DL} Gpc for z = {z}")
        # fix_param[self.param_idx["snr"]] = DL
        # print fix_param
        print("=*=" * 10)
        print("Fix parameters:")
        for k, v in fix_param.items():
            print(f"{self.idx_param[k]} = {v}")
        print("=*=" * 10)

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            # set DL
            # params[self.param_idx["snr"]] = DL
            # fix parameter
            for k, v in fix_param.items():
                params[k] = v

            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_w = self.whiten_data(data)
                    log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                    log_asd = self.standardize_data(log_asd)
                    sig_dataset[i, j] = log_asd
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise_w = self.whiten_data(noise)
                log_asd = self.gen_log_asd(data=noise_w, ovlp=0.2)
                log_asd = self.standardize_data(log_asd)
                noise_dataset[i, j] = log_asd

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    @Timer()
    def generate_grid_M_snr_asd_clsf_dataset(
        self,
        dims=[0, 6],
        n_grid=[30, 30],
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        # n_samples = n_train + n_test

        M_snr_param = self.grid_M_snr(dims=dims, n_grid=n_grid)
        n_samples = np.prod(n_grid)

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        # snr_factor = xp.sqrt(self.T / 2)
        snr_factor = 1.0

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.float32
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            for dim in dims:
                params[dim] = M_snr_param[i, dim]
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_w = self.whiten_data(data)
                    log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                    log_asd = self.standardize_data(log_asd)
                    sig_dataset[i, j] = log_asd
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise_w = self.whiten_data(noise)
                log_asd = self.gen_log_asd(data=noise_w, ovlp=0.2)
                log_asd = self.standardize_data(log_asd)
                noise_dataset[i, j] = log_asd

        # self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset
        # self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset

        # self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array

    @Timer()
    def generate_fd_log_clsf_dataset(
        self,
        n_train,
        n_test,
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        snr_factor = xp.sqrt(self.T / 2)
        # snr_factor = 1.0

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.complex64
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nlogf), dtype=np.complex64
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_f = xp.fft.rfft(data)
                    # bspl = xpx_interpolate.make_interp_spline(self.sample_frequencies, data_f)
                    # data_log_f = bspl(self.sample_log_frequencies)
                    data_log_f = xp.interp(
                        self.sample_log_frequencies,
                        self.sample_frequencies,
                        data_f,
                    )
                    # data_w = self.whiten_data(data)
                    # log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                    # log_asd = self.standardize_data(log_asd)
                    sig_dataset[i, j] = data_log_f.get()
                params_array[i] = params

                # sig = xp.stack([emri_aet["A"], emri_aet["E"]], axis=0)
                # sig = sig * params[self.param_idx["snr"]] / rho * snr_factor
                # # for i in range(self.n_channels):
                # sig_w = xp.stack([self.whiten_data(sig[i]) for i in range(self.n_channels)], axis=0)
                # noise_w = xp.stack([self.whiten_data(self.gen_noise()) for i in range(self.n_channels)], axis=0)
                # data_w = sig_w + noise_w
                # log_asd = np.stack([self.gen_log_asd(data=data_w[i], ovlp=0.2) for i in range(self.n_channels)], axis=0)
                # sig_dataset[i] = log_asd
                # params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                # noise_w = self.whiten_data(noise)
                noise_f = xp.fft.rfft(noise)
                # bspl = xpx_interpolate.make_interp_spline(self.sample_frequencies, noise_f)
                # noise_log_f = bspl(self.sample_log_frequencies)
                noise_log_f = xp.interp(
                    self.sample_log_frequencies,
                    self.sample_frequencies,
                    noise_f,
                )
                # data_w = self.whiten_data(data)
                # log_asd = self.gen_log_asd(data=data_w, ovlp=0.2)
                # log_asd = self.standardize_data(log_asd)
                noise_dataset[i, j] = noise_log_f.get()

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    def fft_mf(self, data, h_idx):
        """Matched filter using FFT
        Args:
            data (np.ndarray): data with shape (Nt)
            w (np.ndarray): waveform template with shape (n_template, 1, Nf)
        return:
            np.ndarray: matched filter output
        """
        data = torch.from_numpy(data.get())
        torch.abs(
            torch.fft.ifft(
                torch.fft.fft(data, n=2 * data.shape[-1] - 1).unsqueeze(1)
                * self.fft_template[:, :, h_idx].unsqueeze(-2)
            )
        )

    @Timer()
    def generate_fftmf_clsf_dataset(self, n_train, n_test, trg_len=1024):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        snr_factor = xp.sqrt(self.T / 2)
        # snr_factor = 1.0
        # [64, 2, Nt]
        # whitened template
        template = np.load("../datasets/emri_64_5_template.npy")
        # [64, 16]
        template_params = np.load("../datasets/emri_64_5_params.npy")
        num_template = template.shape[0]
        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels * num_template, trg_len),
            dtype=np.float32,
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels * num_template, trg_len),
            dtype=np.float32,
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        idx_step = (
            0.3 / self.f_isco(template_params[:, 0]) / self.delta_t + 0.5
        )
        idx_mask = (
            np.outer(
                idx_step.astype(int), np.arange(-trg_len // 2, trg_len // 2)
            ).astype(int)
            + self.Nf // 2
        )
        row_indices, _ = np.meshgrid(
            np.arange(num_template), np.arange(trg_len), indexing="ij"
        )
        # # [64, 1024]
        # time_shift = np.expand_dims(
        #     np.arange(-trg_len // 2, trg_len // 2), axis=0
        # ) * np.expand_dims((1.0 / f_ref), axis=1)
        # ts_idx = time_shift + 2 * template.shape[-1] - 1

        template_torch = torch.from_numpy(template)
        # [64, 2, Nf]
        self.fft_template = torch.fft.rfft(
            template_torch.flip(-1),
        )

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                dataAE = xp.zeros(
                    (self.n_channels, self.Nf), dtype=xp.complex64
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_f = xp.fft.rfft(self.win * data)
                    dataAE[j] = self.whiten_data(data_f, flag="fd")
                    # sub_mfsnr = mfsnr[]
                # [1, 2, Nf]
                dataAE_torch = torch.from_numpy(dataAE.get()).unsqueeze(0)
                # [64, 2, Nf]
                mfsnr = torch.abs(
                    torch.fft.ifft(dataAE_torch * self.fft_template)
                )
                # [128, 1024]
                selected_snr = (
                    mfsnr[row_indices, :, idx_mask]
                    .permute(0, 2, 1)
                    .contiguous()
                    .view(-1, trg_len)
                )

                sig_dataset[i] = selected_snr
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            dataAE = xp.zeros((self.n_channels, self.Nf), dtype=xp.complex64)
            for j in range(self.n_channels):
                noise = self.gen_noise()
                # noise_f = xp.fft.rfft(noise)
                noise_f = xp.fft.rfft(self.win * noise)
                dataAE[j] = self.whiten_data(noise_f, flag="fd")

            # [1, 2, Nf]
            dataAE_torch = torch.from_numpy(dataAE.get()).unsqueeze(0)
            # [64, 2, Nf]
            mfsnr = torch.abs(torch.fft.ifft(dataAE_torch * self.fft_template))
            # [128, 1024]
            selected_snr = (
                mfsnr[row_indices, :, idx_mask]
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, trg_len)
            )

            noise_dataset[i] = selected_snr

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    def generate_fd_clsf_dataset(
        self,
        n_train,
        n_test,
    ):
        r"""rescaling by network SNR
        rho = \sqrt{\rho_A^2 + \rho_E^2}

        Generate whitened FD log-uniformly distributed ASD data
        """
        n_samples = n_train + n_test

        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        snr_factor = xp.sqrt(self.T / 2)
        # snr_factor = 1.0

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nf), dtype=np.complex64
        )
        noise_dataset = np.zeros(
            (n_samples, self.n_channels, self.Nf), dtype=np.complex64
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.float32)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if (
                    xp.isnan(emri_aet["A"]).any()
                    or xp.isnan(emri_aet["E"]).any()
                ):
                    raise ValueError("sig contains NaN values")
                if xp.allclose(emri_aet["A"], 0, atol=1e-50) or xp.allclose(
                    emri_aet["E"], 0, atol=1e-50
                ):
                    raise ValueError("sig generate fail, all zero values")

                # network snr
                rho = xp.sqrt(
                    self.get_snr(emri_aet["A"]) ** 2
                    + self.get_snr(emri_aet["E"]) ** 2
                )
                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    noise = self.gen_noise()
                    data = sig + noise
                    data_f = xp.fft.rfft(data)
                    sig_dataset[i, j] = data_f.get()
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}]"
                )
        pbar.close()
        assert np.isnan(sig_dataset).any() == False

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise_f = xp.fft.rfft(noise)
                noise_dataset[i, j] = noise_f.get()

        self.waveform_dataset["train"]["signal"] = sig_dataset[:n_train]
        self.waveform_dataset["test"]["signal"] = sig_dataset[n_train:]
        self.waveform_dataset["train"]["noise"] = noise_dataset[:n_train]
        self.waveform_dataset["test"]["noise"] = noise_dataset[n_train:]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]

    def fmfwraper(self, data):
        if self.temp64 is None:
            raise ValueError("temp64 is None")
        n_templates, n_channel, temp_len = self.temp64.shape
        n_station = 1
        step = 2

        templates = np.expand_dims(self.temp64, axis=1)
        data = np.expand_dims(np.stack(data, axis=0), axis=0)
        moveouts = np.zeros(
            [n_templates, n_station, n_channel], dtype=np.int32
        )
        weights = np.ones(
            [n_templates, n_station, n_channel], dtype=np.float32
        )
        weights = weights / np.sum(weights)
        return fmf.matched_filter(
            templates,
            moveouts,
            weights,
            data,
            step,
            n_samples_template=None,
            arch="cpu",
            check_zeros="first",
            normalize="short",
            network_sum=True,
        )

    @Timer()
    def generate_fmf_clsf_dataset(self, trg_len=1024, whitten=False):
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.mfcc[i][j] = np.zeros(
                    [self.waveform_dataset[i][j].shape[0], 64, trg_len]
                )

        self.temp64 = np.load("../datasets/emri_64_2_template.npy")
        for i in self.waveform_dataset.keys():
            # for j in self.waveform_dataset[i].keys():
            j = "signal"
            for k in tqdm(
                range(self.waveform_dataset[i][j].shape[0]), desc=f"{i} {j}"
            ):
                data = np.pad(
                    self.waveform_dataset[i][j][k]
                    + self.waveform_dataset[i]["noise"][k],
                    # ((0, 0), (trg_len - 1, trg_len - 1)),
                    ((0, 0), (trg_len, trg_len - 1)),
                    "constant",
                    constant_values=(0, 0),
                )
                cc = self.fmfwraper(data)
                # cc = np.pad(
                #     cc,
                #     ((0, 0), (0, 1)),
                #     'constant',
                #     constant_values=(0, 0)
                # )
                # cc = np.reshape(cc, [64, -1, trg_len]).sum(axis=1)
                # print(cc.shape)
                # exit()
                self.mfcc[i][j][k] = cc
                # data = self.waveform_dataset[i][j][k] + self.waveform_dataset[i]["noise"][k]

            # ==================================================================
            j = "noise"
            for k in tqdm(
                range(
                    int(self.waveform_dataset[i][j].shape[0] * 0.5 + 0.5),
                    self.waveform_dataset[i][j].shape[0],
                ),
                desc=f"{i} {j}",
            ):
                data = np.pad(
                    self.waveform_dataset[i][j][k],
                    # ((0, 0), (trg_len - 1, trg_len - 1)),
                    ((0, 0), (trg_len, trg_len - 1)),
                    "constant",
                    constant_values=(0, 0),
                )
                cc = self.fmfwraper(data)
                # cc = np.pad(
                #     cc,
                #     ((0, 0), (0, 1)),
                #     'constant',
                #     constant_values=(0, 0)
                # )
                # cc = np.reshape(cc, [64, -1, trg_len]).sum(axis=1)
                self.mfcc[i][j][k] = cc

    def f_isco(self, M):
        return 1 / (2 * Constant.PI * M * Constant.MTSUN_SI * 6**1.5)

    def generate_fd_inner_product_clsf_dataset(self, trg_len=1024):
        template = xp.load("../datasets/emri_64_3_template.npy")
        template_params = xp.load("../datasets/emri_64_3_params.npy")
        # fd_template = xp.fft.rfft(template, axis=2)
        fd_template = xp.zeros([64, 2, self.Nf], dtype=xp.complex64)
        for i in range(64):
            fd_template[i] = xp.fft.rfft(template[i], axis=-1)
        f_ref = 10 * self.f_isco(template_params[:, 0])
        time_shift = xp.expand_dims(
            xp.arange(-trg_len // 2, trg_len // 2), axis=0
        ) * xp.expand_dims((1.0 / f_ref), axis=1)
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.innerproduct[i][j] = np.zeros(
                    [self.waveform_dataset[i][j].shape[0], 64, trg_len]
                )
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                for k in tqdm(range(self.waveform_dataset[i][j].shape[0])):
                    for l in range(trg_len // 4):
                        dataA, dataE = (
                            self.waveform_dataset[i][j][k][0],
                            self.waveform_dataset[i][j][k][1],
                        )
                        ipA = self.get_inner_product_fd_w(
                            dataA,
                            fd_template[:, 0],
                            time_shift[:, l * 4 : l * 4 + 4],
                        )
                        ipE = self.get_inner_product_fd_w(
                            dataE,
                            fd_template[:, 1],
                            time_shift[:, l * 4 : l * 4 + 4],
                        )
                        self.innerproduct[i][j][k][:, l * 4 : l * 4 + 4] = (
                            ipA + ipE
                        )

    @Timer()
    def generate_mfcc_clsf_dataset(self, trg_len=1024, whitten=False):
        for i in self.waveform_dataset.keys():
            for j in self.waveform_dataset[i].keys():
                self.mfcc[i][j] = np.zeros(
                    [self.waveform_dataset[i][j].shape[0], 64, trg_len]
                )

        self.temp64 = np.load("../datasets/emri_64_2_template.npy")
        self.hh_sqrt = np.sqrt(np.load("../datasets/emri_64_2_hh_sqrt.npy"))

        for i in self.waveform_dataset.keys():
            # for j in self.waveform_dataset[i].keys():
            j = "signal"
            for k in tqdm(
                range(self.waveform_dataset[i][j].shape[0]), desc=f"{i} {j}"
            ):
                data = np.pad(
                    self.waveform_dataset[i][j][k]
                    + self.waveform_dataset[i]["noise"][k],
                    # ((0, 0), (trg_len - 1, trg_len - 1)),
                    ((0, 0), (trg_len, trg_len - 1)),
                    "constant",
                    constant_values=(0, 0),
                )
                cc = self.fmfwraper(data)
                # cc = np.pad(
                #     cc,
                #     ((0, 0), (0, 1)),
                #     'constant',
                #     constant_values=(0, 0)
                # )
                # cc = np.reshape(cc, [64, -1, trg_len]).sum(axis=1)
                # print(cc.shape)
                # exit()
                self.mfcc[i][j][k] = cc
                # data = self.waveform_dataset[i][j][k] + self.waveform_dataset[i]["noise"][k]

            # ==================================================================
            j = "noise"
            for k in tqdm(
                range(
                    int(self.waveform_dataset[i][j].shape[0] * 0.5 + 0.5),
                    self.waveform_dataset[i][j].shape[0],
                ),
                desc=f"{i} {j}",
            ):
                data = np.pad(
                    self.waveform_dataset[i][j][k],
                    # ((0, 0), (trg_len - 1, trg_len - 1)),
                    ((0, 0), (trg_len, trg_len - 1)),
                    "constant",
                    constant_values=(0, 0),
                )
                cc = self.fmfwraper(data)
                # cc = np.pad(
                #     cc,
                #     ((0, 0), (0, 1)),
                #     'constant',
                #     constant_values=(0, 0)
                # )
                # cc = np.reshape(cc, [64, -1, trg_len]).sum(axis=1)
                self.mfcc[i][j][k] = cc

    @Timer()
    def generate_fd_clsf_dataset_v2(
        self, n_train, n_test, trg_len=131072, whitten=True
    ):
        assert trg_len < self.Nt
        n_samples = n_train + n_test
        cut_win = 2 * int((self.Nt - trg_len) / 2 + 0.5)
        # assert self.Nt - 2 * cut_win == trg_len
        assert self.Nt - cut_win == trg_len
        # scale snr effective to 2 year
        # LDC 2yr snr range 50 - 70
        # snr_factor = xp.sqrt(self.T / 2)
        snr_factor = 1.0

        log.info("Generating FD dataset...")

        # Initialize the dataset with zeros, you can also use np.empty if you want.
        sig_dataset = np.zeros(
            (n_samples, self.n_channels, 2 * self.Nt - 1), dtype=np.complex64
        )
        noise_dataset = np.zeros(
            (n_samples * 2, self.n_channels, 2 * self.Nt - 1),
            dtype=np.complex64,
        )
        params_array = np.zeros((n_samples, self.nparams), dtype=np.complex64)

        log.info("Generating signal dataset")
        pbar = tqdm(total=n_samples)
        i = 0
        while i < n_samples:
            params = self.sampling_one_parameter()
            try:
                emri_aet = self.aak_tdi(*list(params))
                if np.isnan(emri_aet["A"]).any():
                    raise ValueError("sig contains NaN values")

                for j in range(self.n_channels):
                    sig = emri_aet[self.tdi_chan[j]]
                    rho = self.get_snr(sig)
                    sig = (
                        sig * params[self.param_idx["snr"]] / rho * snr_factor
                    )
                    sig = self.whiten_data(sig)
                    sig_f = xp.fft.fft(sig, 2 * self.Nt - 1)
                    sig_dataset[i, j] = sig_f.get()
                params_array[i] = params

                i += 1  # Only increment the counter if the sample was generated successfully.
                pbar.update(1)
            except Exception as e:
                print(
                    f"An error occurred while generating sample {i}: {e}. Skipping this sample."
                )
        pbar.close()

        log.info("Generating noise dataset")
        for i in tqdm(range(n_samples * 2)):
            for j in range(self.n_channels):
                noise = self.gen_noise()
                noise = self.whiten_data(noise)
                noise_f = xp.fft.fft(noise, 2 * self.Nt - 1)
                noise_dataset[i, j] = noise_f.get()

        self.waveform_dataset["train"]["signal"] = sig_dataset[
            :n_train, :, cut_win:-cut_win
        ]
        self.waveform_dataset["test"]["signal"] = sig_dataset[
            n_train:, :, cut_win:-cut_win
        ]
        self.waveform_dataset["train"]["noise"] = noise_dataset[
            : 2 * n_train, :, cut_win:-cut_win
        ]
        self.waveform_dataset["test"]["noise"] = noise_dataset[
            2 * n_train :, :, cut_win:-cut_win
        ]

        self.waveform_params["train"] = params_array[:n_train]
        self.waveform_params["test"] = params_array[n_train:]


def main():
    return 0


if __name__ == "__main__":
    main()
