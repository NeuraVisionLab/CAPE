#!/usr/bin/env python
"""
Train on CREMI 2D with MSE + CAPE loss.
Uses CAPE from this repository (loss.py) with three_dimensional=False.
Run from CAPE-main directory:
  python train_cremi.py --config_file config/cremi_2d.yaml
"""
from __future__ import annotations

import logging
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is on path when running as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from loss import CAPE
from data.cremi_dataset import CremiDataset2D, collate_cremi
from model import UNet
from training.config import parse_config
from training.trainer import Trainer, run_training

logger = logging.getLogger(__name__)


def _compute_train_loss(mse_fn, cape_fn, aux_weight: float, initial_epoch_cape: int, use_graphs: bool):
    import networkx as nx

    def compute(epoch, pred, labels, batch, device):
        loss1 = mse_fn(pred, labels)
        loss2 = torch.tensor(0.0, device=device)
        if epoch >= initial_epoch_cape and cape_fn is not None:
            pred_flat = pred.squeeze(1)  # (B, H, W)
            if use_graphs:
                # batch["graph"] is list of nx.Graph or None (from .gpickle)
                graphs = [
                    g if (g is not None and isinstance(g, nx.Graph)) else nx.Graph()
                    for g in batch["graph"]
                ]
                loss2 = aux_weight * cape_fn(pred_flat, graphs)
            else:
                # CAPE skeletonizes from distance map at training time (0 = foreground)
                dist_np = [
                    batch["distance"][i].detach().cpu().numpy().squeeze()
                    for i in range(batch["distance"].shape[0])
                ]
                loss2 = aux_weight * cape_fn(pred_flat, dist_np)
        loss = loss1 + loss2
        return loss, loss1.item(), loss2.item()
    return compute


def main():
    args = parse_config(
        config_file=os.path.join(_SCRIPT_DIR, "config", "cremi_2d.yaml"),
        resume="last",
        training_order="fold_by_fold",
        tensorboard=True,
    )
    if args.resume not in ("best_loss", "best_quality", "last", "no"):
        args.resume = "last"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    mse_loss = torch.nn.MSELoss().to(device)
    cape_loss = CAPE(
        window_size=getattr(args, "window_size", 128),
        three_dimensional=getattr(args, "three_dimensional", False),
        dilation_radius=getattr(args, "dilation_radius", 10),
        shifting_radius=getattr(args, "shifting_radius", 5),
        is_binary=getattr(args, "is_binary", False),
        distance_threshold=getattr(args, "distance_threshold", 20),
        single_edge=getattr(args, "single_edge", False),
    ).to(device)
    aux_weight = getattr(args, "aux_loss_weight", 0.0001)
    initial_epoch_cape = getattr(args, "training_initial_epoch_custom_loss", 0)
    use_graphs = getattr(args, "use_graphs", True)

    def get_trainer_kwargs(fold_idx: int):
        args.fold = fold_idx
        train_ds = CremiDataset2D(
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            fold=args.fold,
            num_folds=args.num_folds,
            split="train",
            crop_size=args.crop_size,
            margin_size=args.margin_size,
            dist_thresh=args.dist_thresh,
            prob_augment=getattr(args, "prob_augment", 1.0),
            multiplier_size=getattr(args, "multiplier_size", 8),
            fixed_test_size=getattr(args, "fixed_test_size", 0),
            fold_random_state=getattr(args, "fold_random_state", 42),
            save_inds_path=getattr(args, "save_inds_path", "logs/cremi_splits"),
            simple_augment=getattr(args, "simple_augment", True),
            use_graphs=use_graphs,
        )
        val_ds = CremiDataset2D(
            root_dir=args.root_dir,
            dataset_name=args.dataset_name,
            fold=args.fold,
            num_folds=args.num_folds,
            split="val",
            crop_size=args.crop_size,
            margin_size=args.margin_size,
            dist_thresh=args.dist_thresh,
            multiplier_size=getattr(args, "multiplier_size", 8),
            fixed_test_size=getattr(args, "fixed_test_size", 0),
            fold_random_state=getattr(args, "fold_random_state", 42),
            save_inds_path=getattr(args, "save_inds_path", "logs/cremi_splits"),
            simple_augment=True,
            use_graphs=use_graphs,
        )
        num_workers = getattr(args, "num_workers", 4)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_cremi,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_cremi
        )

        model = UNet(
            in_channels=args.in_channels,
            m_channels=args.m_channels,
            out_channels=getattr(args, "num_classes", 1),
            n_convs=args.n_convs,
            n_levels=args.n_levels,
            dropout=args.dropout,
            batch_norm=args.batch_norm,
            upsampling=args.upsampling,
            pooling=args.pooling,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        compute_train_loss = _compute_train_loss(
            mse_loss, cape_loss, aux_weight, initial_epoch_cape, use_graphs
        )
        checkpoint_dir = os.path.join(args.output_path, f"fold_{fold_idx}", "checkpoints")
        return dict(
            model=model,
            optimizer=optimizer,
            mse_loss=mse_loss,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            crop_size=args.crop_size,
            margin_size=args.margin_size,
            num_epochs=args.num_epochs,
            fold=fold_idx,
            use_amp=getattr(args, "use_amp", False),
            checkpoint_dir=checkpoint_dir,
            print_every=getattr(args, "print_every", 10),
            save_every=getattr(args, "save_every", 500),
            valid_every=getattr(args, "valid_every", 500),
            output_path=os.path.join(args.output_path, f"fold_{fold_idx}"),
            config=args,
            compute_train_loss=compute_train_loss,
            resume_checkpoint_dir=(
                os.path.join(args.resume_from_experiment, f"fold_{fold_idx}", "checkpoints")
                if getattr(args, "resume_from_experiment", None) else None
            ),
        )

    run_training(args, get_trainer_kwargs, script_path=os.path.abspath(__file__))


if __name__ == "__main__":
    main()
