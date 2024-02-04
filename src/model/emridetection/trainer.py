# mypy: ignore-errors
# type: ignore
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import SGD, Adadelta, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, ExponentialLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer_type,
        optimizer_kwargs,
        scheduler_type,
        scheduler_kwargs,
        loss_fn,
        device,
        result_dir,
        result_fn="inf_result.npy",
        checkpoint_dir=None,
        use_wandb=False,
    ):
        """Trainer class for training and evaluating the model."""
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = self.get_optimizer(optimizer_type, self.model, **optimizer_kwargs)
        self.scheduler = self.get_lr_scheduler(scheduler_type, self.optimizer, **scheduler_kwargs)
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.device = device

        # self.output_folder = result_dir
        self.output_folder = Path(result_dir)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.result_fn = result_fn
        self.checkpoint_dir = Path(checkpoint_dir)

        self.step = 0
        self.avg_train_loss = 0.0
        self.avg_test_loss = 0.0

        self.best_loss = float("inf")
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project="my_project", entity="my_entity")
            wandb.watch(model)
        self.count_parameters(self.model)

    def update_average(self, loss, avg_loss):
        """Update running average of the loss.

        Arguments
        ---------
        loss : torch.tensor
            detached loss, a single float value.
        avg_loss : float
            current running average.

        Returns
        -------
        avg_loss : float
            The average loss.
        """
        if torch.isfinite(loss):
            avg_loss -= avg_loss / self.step
            avg_loss += float(loss) / self.step
        return avg_loss

    def accuracy(self, output, target):
        with torch.no_grad():
            pred = torch.argmax(output, dim=1)
            assert pred.shape[0] == len(target)
            correct = 0
            correct += torch.sum(pred == target).item()
        return correct / len(target)

    def calculate_accuracy(self, logits, true_labels, threshold=0.5):
        """
        Calculate the classification accuracy for binary classification.

        :param logits: The raw output scores from the model (before sigmoid).
                    Expected shape is (batch_size).
        :param true_labels: The ground truth labels.
                            Expected shape is (batch_size).
        :param threshold: The threshold to classify samples as positive or negative.

        :return: The accuracy as a floating point number between 0 and 1.
        """

        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Convert probabilities to binary predictions
        predictions = (probabilities >= threshold).float()

        # Check how many of the predictions are equal to the true labels
        correct_predictions = (predictions == true_labels).float()

        # Calculate the accuracy
        accuracy = correct_predictions.sum() / len(true_labels)

        return accuracy.item()

    def load_checkpoint(
        self,
    ):
        print(f"load model from checkpoint {self.checkpoint_dir}")
        self.checkpoint = torch.load(self.checkpoint_dir)
        # print(self.checkpoint)
        # exit()
        self.model.load_state_dict(self.checkpoint)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        self.avg_train_loss = 0
        avg_acc = 0
        self.step = 0
        pbar = tqdm(total=len(dataloader.dataset), desc=f"Epoch {epoch+1}")
        for idx, inputs, targets in dataloader:
            self.step += 1
            idx, inputs, targets = (
                idx.to(self.device),
                inputs.to(self.device),
                targets.to(self.device),
            )
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            # loss = self.loss_fn(outputs.squeeze(-1), targets)
            loss = self.loss_fn(outputs, targets)
            loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)

            self.optimizer.step()
            acc = self.accuracy(outputs, targets)
            # acc = self.calculate_accuracy(outputs, targets)
            self.avg_train_loss = self.update_average(loss.detach().cpu(), self.avg_train_loss)
            avg_acc = self.update_average(torch.tensor(acc), avg_acc)

            pbar.update(inputs.shape[0])
            pbar.set_postfix({"loss": f"{self.avg_train_loss:.2e}", "acc": f"{avg_acc:.2f}"})
        pbar.close()

        # if self.use_wandb:
        #     wandb.log(
        #         {
        #             "train_loss": avg_loss,
        #             "train_acc": avg_acc,
        #             "epoch": epoch + 1,
        #         }
        #     )
        return self.avg_train_loss, avg_acc

    def evaluate(self, dataloader, save=False):
        self.model.eval()
        self.avg_test_loss = 0
        avg_acc = 0
        self.step = 0
        if save:
            inf_result = np.zeros([len(dataloader.dataset), 130])
        with torch.no_grad():
            pbar = tqdm(total=len(dataloader.dataset))
            for idx, inputs, targets in dataloader:
                idx, inputs, targets = (
                    idx.to(self.device),
                    inputs.to(self.device),
                    targets.to(self.device),
                )
                self.step += 1
                # add timing
                start = time.time()
                outputs, latent_vec = self.model(inputs)
                end = time.time()
                print(f"Time: {end-start}")
                # print(latent_vec.shape)
                # exit()
                if save:
                    inf_map = self.model.get_activation_maps()  # just one sample
                    print("Saving inf maps...")
                    np.save(f"inf_map_AK.npy", [inf_map[i].cpu().numpy() for i in range(len(inf_map))])
                    # self.model.clear_activation_maps()
                    for i, ind in enumerate(idx):
                        line = torch.cat((latent_vec, outputs), dim=1)
                        inf_result[ind] = line[i].cpu().numpy()
                        # print(line.shape)
                        # exit()
                # loss = self.loss_fn(outputs.squeeze(-1), targets)
                self.model.clear_activation_maps()
                loss = self.loss_fn(outputs, targets)
                loss = loss.mean()

                acc = self.accuracy(outputs, targets)
                # acc = self.calculate_accuracy(outputs, targets)
                self.avg_test_loss = self.update_average(loss.detach(), self.avg_test_loss)
                avg_acc = self.update_average(torch.tensor(acc), avg_acc)
                pbar.update(inputs.shape[0])
                pbar.set_postfix(
                    {
                        "loss": f"{self.avg_test_loss:.2e}",
                        "acc": f"{avg_acc:.4f}",
                    }
                )
            pbar.close()
        # if self.use_wandb:
        # wandb.log({"val_loss": avg_loss, "val_acc": avg_acc})
        # if save:
        # log.info(
        #     f"save inf results to: {self.output_folder}/{self.result_fn}"
        # )
        # np.save(self.output_folder / self.result_fn, inf_result)
        return self.avg_test_loss, avg_acc

    def train(self, n_epochs):
        avg_loss = []
        for epoch in range(n_epochs):
            train_loss, train_acc = self.train_epoch(self.train_loader, epoch)
            val_loss, valid_acc = self.evaluate(self.test_loader)
            log.info(f"EPOCH {epoch+1}\t: lr={self.optimizer.param_groups[0]['lr']:.2e},\t train_loss={train_loss:.2e}, \t train_acc={train_acc:.4f}, \t val_loss={val_loss:.2e} \t valid_acc={valid_acc:.4f}")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    self.output_folder / "best_model.pth",
                )
            avg_loss.append([train_loss, val_loss])
            # print(
            #     f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            # )

    def get_lr_scheduler(self, scheduler_choice, optimizer, **kwargs):
        if scheduler_choice == "step":
            return StepLR(
                optimizer,
                step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1),
            )
        elif scheduler_choice == "exponential":
            return ExponentialLR(optimizer, gamma=kwargs.get("gamma", 0.1))
        elif scheduler_choice == "plateau":
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=kwargs.get("factor", 0.1),
                patience=kwargs.get("patience", 10),
                verbose=True,
            )
        elif scheduler_choice == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get("T_max", 50),
                eta_min=kwargs.get("eta_min", 0),
            )
        elif scheduler_choice == "cyclic":
            return CyclicLR(
                optimizer,
                base_lr=kwargs.get("base_lr", 0.001),
                max_lr=kwargs.get("max_lr", 0.1),
                step_size_up=kwargs.get("step_size_up", 2000),
                mode=kwargs.get("mode", "triangular"),
            )
        else:
            raise ValueError(f"Invalid scheduler choice: {scheduler_choice}")

    def get_optimizer(self, optimizer_choice, model, **kwargs):
        if optimizer_choice == "adam":
            print("Using Adam optimizer, lr={}, weight_decay={}".format(kwargs.get("lr"), kwargs.get("weight_decay")))
            return Adam(
                model.parameters(),
                lr=kwargs.get("lr", 1e-3),
                weight_decay=kwargs.get("weight_decay", 0),
            )
        elif optimizer_choice == "sgd":
            return SGD(
                model.parameters(),
                lr=kwargs.get("lr", 1e-2),
                momentum=kwargs.get("momentum", 0.98),
                weight_decay=kwargs.get("weight_decay", 1e-3),
            )
        elif optimizer_choice == "rmsprop":
            return RMSprop(
                model.parameters(),
                lr=kwargs.get("lr", 1e-2),
                weight_decay=kwargs.get("weight_decay", 0),
            )
        elif optimizer_choice == "adagrad":
            return Adagrad(
                model.parameters(),
                lr=kwargs.get("lr", 1e-2),
                weight_decay=kwargs.get("weight_decay", 0),
            )
        elif optimizer_choice == "adadelta":
            return Adadelta(
                model.parameters(),
                lr=kwargs.get("lr", 1.0),
                weight_decay=kwargs.get("weight_decay", 0),
            )
        elif optimizer_choice == "adamw":
            return AdamW(
                model.parameters(),
                lr=kwargs.get("lr", 1e-3),
                weight_decay=kwargs.get("weight_decay", 0),
            )
        else:
            raise ValueError(f"Invalid optimizer choice: {optimizer_choice}")

    def get_loss_fn(self, loss_choice, **kwargs):
        if loss_choice == "cross_entropy":
            return nn.CrossEntropyLoss(**kwargs)
        elif loss_choice == "mse":
            return nn.MSELoss(**kwargs)
        elif loss_choice == "l1":
            return nn.L1Loss(**kwargs)
        elif loss_choice == "nll":
            return nn.NLLLoss(**kwargs)
        elif loss_choice == "bce":
            return nn.BCELoss(**kwargs)
        elif loss_choice == "bce_with_logits":
            return nn.BCEWithLogitsLoss(**kwargs)
        else:
            raise ValueError(f"Invalid loss choice: {loss_choice}")

    def get_gpu_tensors_sorted_by_size(self):
        gpu_tensors = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj.is_cuda:
                gpu_tensors.append((obj, obj.element_size() * obj.nelement()))

        gpu_tensors.sort(key=lambda x: x[1], reverse=True)
        return gpu_tensors

    def print_gpu_tensors(self, top_n=None):
        torch.cuda.empty_cache()
        gpu_tensors = self.get_gpu_tensors_sorted_by_size()

        if top_n is not None:
            gpu_tensors = gpu_tensors[:top_n]

        print("\nGPU tensors sorted by size:")
        for tensor, size in gpu_tensors:
            print(f"Size (MB): {size/1e6}, Tensor Shape: {tensor.shape}, Tensor dtype: {tensor.dtype}")

    def format_number_with_unit(self, num):
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.2f}B"
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.2f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.2f}K"
        else:
            return str(num)

    def count_parameters(self, model):
        total_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

        formatted_total_params = self.format_number_with_unit(total_params)
        formatted_trainable_params = self.format_number_with_unit(trainable_params)
        formatted_non_trainable_params = self.format_number_with_unit(total_params - trainable_params)

        print(f"Total parameters: {formatted_total_params}")
        print(f"Trainable parameters: {formatted_trainable_params}")
        print(f"Non-trainable parameters: {formatted_non_trainable_params}")


# def main():
#     # Import necessary libraries
#     import torch.optim as optim
#     import torch.nn as nn
#     from model import MyModel
#     from dataset import MyTrainDataset, MyValDataset

#     # Set device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Create datasets
#     train_dataset = MyTrainDataset()
#     val_dataset = MyValDataset()

#     # Create model
#     model = MyModel()
#     model.to(device)

#     # Create optimizer and loss function
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.CrossEntropyLoss()

#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         device=device,
#         use_wandb=False
#         )

#     # Train the model
#     trainer.train(n_epochs=10)
