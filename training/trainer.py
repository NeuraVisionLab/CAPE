"""
Trainer and run loop for CREMI 2D with MSE + CAPE loss.
Validation uses MSE only; optional TensorBoard logging.
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import from_torch, mkdir, process_in_chunks

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mse_loss: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        crop_size: List[int],
        margin_size: List[int],
        num_epochs: int,
        fold: int,
        use_amp: bool,
        checkpoint_dir: str,
        print_every: int,
        save_every: int,
        valid_every: int,
        output_path: str,
        config: Any,
        compute_train_loss: Callable[..., tuple],
        writer: SummaryWriter | None = None,
        resume_checkpoint_dir: str | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.mse_loss = mse_loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.num_epochs = num_epochs
        self.fold = fold
        self.use_amp = use_amp
        self.checkpoint_dir = checkpoint_dir
        self.print_every = print_every
        self.save_every = save_every
        self.valid_every = valid_every
        self.output_path = output_path
        self.config = config
        self.compute_train_loss = compute_train_loss
        self.writer = writer
        self.resume_checkpoint_dir = resume_checkpoint_dir or checkpoint_dir
        self.best_loss = float("inf")
        self.best_loss_epoch = 0
        self.scaler = torch.amp.GradScaler(enabled=use_amp)
        mkdir(checkpoint_dir)

    def save_checkpoint(self, epoch: int, name: str = "last") -> None:
        path = os.path.join(self.checkpoint_dir, f"{name}.pth")
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "best_loss_epoch": self.best_loss_epoch,
            "scaler_state": self.scaler.state_dict() if self.use_amp else None,
        }
        tmp = path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, path)
        logger.info("Saved %s", path)

    def load_checkpoint(self, option: str) -> int:
        if option == "no":
            return 0
        path = os.path.join(self.resume_checkpoint_dir, f"{option}.pth")
        if not os.path.isfile(path):
            logger.info("Checkpoint not found: %s. Starting from scratch.", path)
            return 0
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state"])
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_loss = state.get("best_loss", float("inf"))
        self.best_loss_epoch = state.get("best_loss_epoch", 0)
        if self.use_amp and state.get("scaler_state"):
            self.scaler.load_state_dict(state["scaler_state"])
        start_epoch = state.get("epoch", 0) + 1
        logger.info("Resumed from epoch %s (best_loss %.4f)", start_epoch, self.best_loss)
        return start_epoch

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        losses, l1_list, l2_list = [], [], []
        label_key = getattr(self.config, "label", "distance")
        start = time.time()
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images = batch["image"].to(self.device)
            labels = batch[label_key].to(self.device)
            # compute_train_loss receives full batch to support graph vs mask mode

            with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp):
                pred = self.model(images)
                if labels.dim() == 3:
                    labels = labels.unsqueeze(1)
                loss, l1, l2 = self.compute_train_loss(epoch, pred, labels, batch, self.device)
            losses.append(loss.item())
            l1_list.append(l1)
            l2_list.append(l2)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if (batch_idx + 1) % self.print_every == 0:
                logger.info(
                    "Fold %s Epoch [%s/%s] Batch [%s/%s] Loss: %.4f (MSE: %.4f, CAPE: %.4f)",
                    self.fold, epoch + 1, self.num_epochs, batch_idx + 1, len(self.train_loader),
                    loss.item(), l1, l2,
                )
        avg_loss = float(np.mean(losses))
        elapsed = time.time() - start
        logger.info(
            "Fold %s Epoch %s train loss: %.4f (MSE: %.4f, CAPE: %.4f) time: %.2fs",
            self.fold, epoch + 1, avg_loss, float(np.mean(l1_list)), float(np.mean(l2_list)), elapsed,
        )
        if self.writer:
            self.writer.add_scalar("train/loss", avg_loss, epoch)
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], epoch)
        return avg_loss

    def validate(self, epoch: int) -> float:
        self.model.eval()
        label_key = getattr(self.config, "label", "distance")
        val_losses = []
        out_valid = os.path.join(self.output_path, "output_valid")
        mkdir(out_valid)

        def process(chunk: torch.Tensor) -> torch.Tensor:
            return self.model(chunk)

        with torch.no_grad():
            with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp):
                for batch in self.val_loader:
                    images = batch["image"]
                    labels = batch[label_key]
                    for i in range(images.shape[0]):
                        img = images[i : i + 1].to(self.device)
                        if img.dim() == 3:
                            img = img.unsqueeze(0)
                        out_shape = (1, 1) + tuple(img.shape[2:])
                        pred = torch.empty(out_shape, dtype=torch.float32, device=self.device)
                        process_in_chunks(
                            img, pred, process,
                            self.crop_size, self.margin_size,
                        )
                        lab = labels[i : i + 1].to(self.device)
                        if lab.dim() == 3:
                            lab = lab.unsqueeze(1)
                        loss = self.mse_loss(pred, lab)
                        val_losses.append(loss.item())
        mean_loss = float(np.mean(val_losses))
        logger.info("Fold %s Epoch %s val MSE: %.4f", self.fold, epoch + 1, mean_loss)
        if self.writer:
            self.writer.add_scalar("val/loss", mean_loss, epoch)
        if mean_loss < self.best_loss:
            self.best_loss = mean_loss
            self.best_loss_epoch = epoch
            self.save_checkpoint(epoch, "best_loss")
        return mean_loss

    def run(self, start_epoch: int) -> None:
        for epoch in range(start_epoch, self.num_epochs):
            self.train_epoch(epoch)
            if (epoch + 1) % self.valid_every == 0:
                self.validate(epoch)
            self.save_checkpoint(epoch, "last")
            if self.save_every and (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, f"epoch_{epoch + 1}")


def run_training(
    args: Any,
    get_trainer_kwargs: Callable[[int], Dict[str, Any]],
    script_path: str,
) -> None:
    """Run training for each fold; optionally copy config and script to output."""
    mkdir(args.output_path)
    if os.path.isfile(args.config_file):
        shutil.copy(args.config_file, os.path.join(args.output_path, "config.yaml"))
    if os.path.isfile(script_path):
        shutil.copy(script_path, os.path.join(args.output_path, "train_script.py"))
    from .utils import config_logger
    config_logger(os.path.join(args.output_path, "train.log"))
    num_folds = getattr(args, "num_folds", 1)
    fold_list = list(range(num_folds)) if getattr(args, "fold", None) is None or args.fold is None else list(args.fold)
    trainers_info: Dict[int, Dict[str, Any]] = {}
    for fold_idx in fold_list:
        fold_dir = os.path.join(args.output_path, f"fold_{fold_idx}")
        mkdir(fold_dir)
        writer = SummaryWriter(log_dir=os.path.join(fold_dir, "tensorboard")) if getattr(args, "tensorboard", True) else None
        kwargs = get_trainer_kwargs(fold_idx)
        kwargs["writer"] = writer
        trainer = Trainer(**kwargs)
        start = trainer.load_checkpoint(args.resume)
        trainers_info[fold_idx] = {"trainer": trainer, "writer": writer}
        trainer.run(start)
    for info in trainers_info.values():
        if info.get("writer"):
            info["writer"].close()
