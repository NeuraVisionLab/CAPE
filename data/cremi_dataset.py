"""
CREMI 2D dataset for CAPE training.

Required: images/, distances/
  - <id>_image.npy, <id>_distance.npy

Optional (for graph mode, no skeletonization at training time):
  - graphs/ with <id>.gpickle (same format as extract_graph.py output)

No labels/ or skeletons/ needed. Set use_graphs=True to load pre-extracted graphs;
otherwise CAPE uses the distance map as mask (skeletonizes at training time).
"""
from __future__ import annotations

import math
import os
import random
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from .graph_utils import (
    crop_graph,
    update_graph_positions_for_padding,
    rotate_image,
    rotate_graph,
    flip_image,
    flip_skeleton,
    flip_graph,
)


SIMPLE_ANGLES = [0, 90, 180, 270]


def _rotate_distance(distance: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate distance map (1, H, W) by angle."""
    if distance.dim() == 2:
        distance = distance.unsqueeze(0)
    return rotate_image(distance, angle)


class CremiDataset2D(Dataset):
    def __init__(
        self,
        root_dir: str,
        dataset_name: str = "CREMI",
        fold: int = 0,
        num_folds: int = 3,
        split: str = "train",
        crop_size: List[int] = (512, 512),
        margin_size: List[int] = (100, 100),
        dist_thresh: float = 20.0,
        prob_augment: float = 1.0,
        angle_range: tuple = (-180, 180),
        multiplier_size: int = 8,
        fixed_test_size: int = 0,
        fold_random_state: int = 42,
        save_inds_path: Optional[str] = None,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
        test_dir: Optional[str] = None,
        simple_augment: bool = True,
        use_graphs: bool = True,
    ):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.fold = fold
        self.num_folds = num_folds
        self.split = split
        self.crop_size = list(crop_size)
        self.margin_size = list(margin_size)
        self.dist_thresh = dist_thresh
        self.prob_augment = prob_augment
        self.angle_range = angle_range
        self.multiplier_size = multiplier_size
        self.simple_augment = simple_augment
        self.use_graphs = use_graphs

        use_explicit = train_dir is not None and val_dir is not None

        if use_explicit:
            base = train_dir if split == "train" else (val_dir if split == "val" else (test_dir or ""))
            if base and os.path.isdir(base):
                self.image_dir = os.path.join(base, "images")
                self.distance_dir = os.path.join(base, "distances")
                self.graph_dir = os.path.join(base, "graphs") if self.use_graphs else ""
                if not os.path.isdir(self.image_dir):
                    raise FileNotFoundError(f"Image dir not found: {os.path.abspath(self.image_dir)}")
                if not os.path.isdir(self.distance_dir):
                    raise FileNotFoundError(f"Distance dir not found: {os.path.abspath(self.distance_dir)}")
                self.indices = sorted(
                    [f.replace("_image.npy", "") for f in os.listdir(self.image_dir) if f.endswith("_image.npy")]
                )
            else:
                self.image_dir = self.distance_dir = self.graph_dir = ""
                self.indices = []
            self._path = self._make_path(use_explicit, base if base else "")
            return

        dataset_dir = os.path.join(self.root_dir, self.dataset_name)
        self.image_dir = os.path.join(dataset_dir, "images")
        self.distance_dir = os.path.join(dataset_dir, "distances")
        self.graph_dir = os.path.join(dataset_dir, "graphs") if self.use_graphs else ""
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Dataset not found: {os.path.abspath(self.image_dir)}")
        if not os.path.isdir(self.distance_dir):
            raise FileNotFoundError(f"Distance dir not found: {os.path.abspath(self.distance_dir)}")
        all_files = sorted(
            [f.replace("_image.npy", "") for f in os.listdir(self.image_dir) if f.endswith("_image.npy")]
        )
        self._path = self._make_path(False, dataset_dir)

        if save_inds_path is None:
            save_inds_path = "logs/cremi_splits"
        os.makedirs(save_inds_path, exist_ok=True)
        train_f = os.path.join(save_inds_path, f"split_{fold}_train.txt")
        val_f = os.path.join(save_inds_path, f"split_{fold}_val.txt")
        test_f = os.path.join(save_inds_path, f"split_{fold}_test.txt")
        if os.path.isfile(train_f) and os.path.isfile(val_f) and os.path.isfile(test_f):
            with open(train_f) as f:
                train_ids = [line.strip() for line in f if line.strip()]
            with open(val_f) as f:
                val_ids = [line.strip() for line in f if line.strip()]
            with open(test_f) as f:
                test_ids = [line.strip() for line in f if line.strip()]
        else:
            random.seed(fold_random_state)
            shuffled = list(all_files)
            random.shuffle(shuffled)
            random.seed(None)
            test_ids = shuffled[:fixed_test_size]
            remaining = shuffled[fixed_test_size:]
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=fold_random_state)
            for fi in range(num_folds):
                tr_idx, va_idx = list(kf.split(remaining))[fi]
                for name, ids in [
                    ("train", [remaining[i] for i in tr_idx]),
                    ("val", [remaining[i] for i in va_idx]),
                    ("test", test_ids),
                ]:
                    path = os.path.join(save_inds_path, f"split_{fi}_{name}.txt")
                    if not os.path.isfile(path):
                        with open(path, "w") as f:
                            for fid in ids:
                                f.write(fid + "\n")
            with open(train_f) as f:
                train_ids = [line.strip() for line in f if line.strip()]
            with open(val_f) as f:
                val_ids = [line.strip() for line in f if line.strip()]
            with open(test_f) as f:
                test_ids = [line.strip() for line in f if line.strip()]

        if split == "train":
            self.indices = train_ids
        elif split == "val":
            self.indices = val_ids
        elif split == "test":
            self.indices = test_ids
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def _make_path(self, explicit: bool, base: str):
        def path(fid: str, name: str, ext: str) -> str:
            if name == "images":
                return os.path.join(self.image_dir, f"{fid}_{ext}")
            if name == "distances":
                return os.path.join(self.distance_dir, f"{fid}_{ext}")
            if name == "graphs":
                # Same naming as extract_graph.py: <id>.gpickle
                return os.path.join(self.graph_dir, f"{fid}.gpickle") if self.graph_dir else ""
            raise KeyError(name)
        return path

    def __len__(self) -> int:
        return len(self.indices)

    def _augment_simple(
        self,
        image: torch.Tensor,
        distance: torch.Tensor,
        graph: Optional[nx.Graph],
        h: int,
        w: int,
    ):
        if h < self.crop_size[0] or w < self.crop_size[1]:
            new_h = max(self.crop_size[0], math.ceil(h / self.multiplier_size) * self.multiplier_size)
            new_w = max(self.crop_size[1], math.ceil(w / self.multiplier_size) * self.multiplier_size)
            pad_h, pad_w = new_h - h, new_w - w
            pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
            pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
            image = torch.nn.functional.pad(image, [pad_left, pad_right, pad_top, pad_bottom], value=0)
            distance = torch.nn.functional.pad(distance, [pad_left, pad_right, pad_top, pad_bottom], value=self.dist_thresh)
            if graph is not None:
                graph = update_graph_positions_for_padding(graph.copy(), pad_left, pad_top)
            h, w = new_h, new_w
        x_min = random.randint(0, max(0, w - self.crop_size[1]))
        y_min = random.randint(0, max(0, h - self.crop_size[0]))
        x_max, y_max = x_min + self.crop_size[1], y_min + self.crop_size[0]
        image = image[:, y_min:y_max, x_min:x_max]
        distance = distance[:, y_min:y_max, x_min:x_max]
        if graph is not None:
            graph = crop_graph(graph, x_min, y_min, x_max, y_max)
        angle = random.choice(SIMPLE_ANGLES)
        center = (image.shape[2] / 2, image.shape[1] / 2)
        image = rotate_image(image, angle)
        distance = _rotate_distance(distance, angle)
        if distance.dim() == 2:
            distance = distance.unsqueeze(0)
        if graph is not None:
            graph = rotate_graph(graph, angle, center)
        direction = random.choice(["horizontal", "vertical"])
        image = flip_image(image, direction)
        distance = flip_skeleton(distance.squeeze(0), direction)
        if distance.dim() == 2:
            distance = distance.unsqueeze(0)
        if graph is not None:
            graph = flip_graph(graph, direction, (self.crop_size[0], self.crop_size[1]))
        return image, distance, graph

    def _augment_center_crop(
        self,
        image: torch.Tensor,
        distance: torch.Tensor,
        graph: Optional[nx.Graph],
        h: int,
        w: int,
    ):
        cx, cy = w // 2, h // 2
        x_min = cx - self.crop_size[1] // 2
        y_min = cy - self.crop_size[0] // 2
        x_max = x_min + self.crop_size[1]
        y_max = y_min + self.crop_size[0]
        image = image[:, y_min:y_max, x_min:x_max]
        distance = distance[:, y_min:y_max, x_min:x_max]
        if graph is not None:
            graph = crop_graph(graph, x_min, y_min, x_max, y_max)
        return image, distance, graph

    def __getitem__(self, idx: int) -> dict:
        fid = self.indices[idx]
        p = self._path
        image = np.load(p(fid, "images", "image.npy")).astype(np.float32) / 255.0
        if image.ndim == 2:
            image = torch.tensor(image).unsqueeze(0)
        else:
            image = torch.tensor(image)
        distance = torch.tensor(np.load(p(fid, "distances", "distance.npy")).astype(np.float32)).unsqueeze(0)
        distance = torch.clamp(distance, max=self.dist_thresh)
        graph = None
        if self.use_graphs and self.graph_dir:
            graph_path = p(fid, "graphs", "")
            if os.path.isfile(graph_path):
                from extract_graph import read_gpickle
                graph = read_gpickle(graph_path)
        h, w = image.shape[1], image.shape[2]

        if self.split == "train":
            if self.simple_augment:
                image, distance, graph = self._augment_simple(image, distance, graph, h, w)
            else:
                image, distance, graph = self._augment_center_crop(image, distance, graph, h, w)
        else:
            image, distance, graph = self._augment_center_crop(image, distance, graph, h, w)

        return {
            "image": image,
            "distance": distance,
            "graph": graph,
            "file_id": fid,
        }


def collate_cremi(batch: list) -> dict:
    """Stack image/distance, keep graph and file_id as lists."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "distance": torch.stack([b["distance"] for b in batch]),
        "graph": [b["graph"] for b in batch],
        "file_id": [b["file_id"] for b in batch],
    }
