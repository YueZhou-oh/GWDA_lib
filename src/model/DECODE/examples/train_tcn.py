import os
from collections import Counter

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

from emridetection.data.dataloader import EMRIDatasetTorch, TinyEMRIDataset, NpzDatasetTorch

# from emridetection.data.emridataset import EMRIDataset
# from emridetection.models.mfcnn.mfcnn import MFDCNNFFT, MFCNNFFT
from emridetection.models.mfcnn.tcn import TCN
from emridetection.train.trainer import Trainer


@hydra.main(version_base="1.2", config_path="../configs", config_name="tcn")
def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.training.gpu)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # Create datasets
    wfd = TinyEMRIDataset(DIR=config.dataset.save_path, fn=config.dataset.fn)

    # Data loader
    wfdt_train = EMRIDatasetTorch(wfd, train=False)
    wfdt_test = EMRIDatasetTorch(wfd, train=False)
    # Npz dataloader
    # AK
    npz_data_fn = "/workspace/zhty/EMRI_Detection/emridetection/notebooks/AK_data.npy"
    # XSPEG
    # npz_data_fn = "/workspace/zhty/EMRI_Detection/emridetection/notebooks/XSPEG_data.npy"

    # wfdt_train = NpzDatasetTorch(npz_data_fn, train=False)
    # wfdt_test = NpzDatasetTorch(npz_data_fn, train=False)

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

    if not config.training.test_only:
        # Train the model
        trainer.train(n_epochs=config.training.n_epoch)
    else:
        trainer.load_checkpoint()
        trainer.evaluate(trainer.test_loader, save=False)


if __name__ == "__main__":
    main()
