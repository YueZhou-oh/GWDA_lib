import os
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

# import numpy as np
from omegaconf import DictConfig, OmegaConf
from rich import print

# from sklearn.metrics import classification_report, confusion_matrix
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset

# from torchsummary import summary
from tqdm import tqdm

from emridetection.data.dataloader import EMRIDatasetTorch, TinyEMRIDataset

# from emridetection.data.emridataset import EMRIDataset
# from emridetection.models.mfcnn.mfcnn import MFDCNNFFT, MFCNNFFT
from emridetection.models.mfcnn.tcn import TCN
from emridetection.train.trainer import Trainer


# @hydra.main(version_base="1.2", config_path="../configs", config_name="tcn")
def evaluate_once(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.training.gpu)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Create datasets
    wfd = TinyEMRIDataset(DIR=config.dataset.save_path, fn=config.dataset.fn)

    # Data loader
    wfdt_train = EMRIDatasetTorch(wfd, train=False)
    wfdt_test = EMRIDatasetTorch(wfd, train=False)

    train_loader = DataLoader(
        wfdt_train,
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.dataloader.num_workers,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1)),
    )
    test_loader = DataLoader(
        wfdt_test,
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.dataloader.num_workers,
        worker_init_fn=lambda _: np.random.seed(int(torch.initial_seed()) % (2**32 - 1)),
    )

    # Create model
    channel_sizes = [config.net.n_hidden] * config.net.n_levels
    model = TCN(
        input_size=config.net.input_channels,
        output_size=config.net.n_classes,
        num_channels=channel_sizes,
        kernel_size=config.net.kernel_size,
        dropout=config.net.dropout,
    )
    model = model.to(device)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_type=config.training.optimizer_type,
        optimizer_kwargs=config.training.optimizer_kwargs,
        scheduler_type=config.training.scheduler_type,
        scheduler_kwargs=config.training.scheduler_kwargs,
        loss_fn=config.training.loss_fn,
        device=device,
        result_dir=config.training.result_dir,
        result_fn=config.training.result_fn,
        checkpoint_dir=config.training.checkpoint_dir,
        use_wandb=config.training.use_wandb,
    )

    trainer.load_checkpoint()
    trainer.evaluate(trainer.test_loader, save=True)


def evaluate_a_z(result_dir="."):
    dataset_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets")
    config = OmegaConf.load("../configs/tcn.yaml")
    config.training.result_dir = result_dir
    for i in dataset_dir.glob("emri_asd_a*"):
        parts = i.stem.split("_", 2)
        fn = "inf_result_" + parts[2] + ".npy"
        config.training.result_fn = fn
        config.dataset.fn = str(i)
        print(f"dataset: {i}")
        print(f"result: {fn}")

        # print(config)
        evaluate_once(config)


def evaluate_x_snr(result_dir="."):
    dataset_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/dataset2")
    config = OmegaConf.load("../configs/tcn.yaml")
    config.training.result_dir = result_dir
    for i in dataset_dir.glob("emri_asd_a*"):
        parts = i.stem.split("_", 2)
        fn = "inf_result_" + parts[2] + ".npy"
        config.training.result_fn = fn
        config.dataset.fn = str(i)
        print(f"dataset: {i}")
        print(f"result: {fn}")

        # print(config)
        evaluate_once(config)


def evaluate_x_dir(result_dir=".", data_folder="dataset2"):
    # amp ratio and amp snr
    dataset_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets") / data_folder
    # print(dataset_dir)
    # print([i for i in dataset_dir.glob("emri_asd")])
    # dataset_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets/amp_snr")
    config = OmegaConf.load("../configs/tcn.yaml")
    config.training.result_dir = result_dir
    for i in dataset_dir.glob("emri_asd*"):
        parts = i.stem.split("_", 6)
        fn = "inf_result_" + parts[-1] + ".npy"
        config.training.result_fn = fn
        config.dataset.fn = str(i)
        print(f"dataset: {i}")
        print(f"result: {fn}")

        # print(config)
        evaluate_once(config)


def evaluate_M_z():
    dataset_dir = Path("/workspace/zhty/EMRI_Detection/emridetection/datasets")
    config = OmegaConf.load("../configs/tcn.yaml")
    for i in dataset_dir.glob("emri_asd_M*"):
        parts = i.stem.split("_", 2)
        fn = "inf_result_" + parts[2]
        config.training.result_fn = fn
        config.dataset.fn = i
        print(config)
        evaluate_once(config)


def main():
    # evaluate_M_z()
    result_dir = "/workspace/zhty/EMRI_Detection/emridetection/results/"
    # evaluate_a_z(result_dir)
    # evaluate_x_snr(result_dir)
    folder = "amp_snr"
    evaluate_x_dir(result_dir + folder + "_4/", folder)


if __name__ == "__main__":
    main()
