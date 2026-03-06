"""
Minimal training utilities: config, chunked inference, logging.
Used only for CREMI 2D training with CAPE.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml


def yaml_read(filename: str) -> Dict[str, Any]:
    with open(filename, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def mkdir(directory: str) -> None:
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


def from_torch(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


# --- Chunked inference (from networktraining/cropping + process_in_chuncks) ---

def _no_crops(in_size: Tuple[int, ...], crop_size: List[int], margin_size: List[int], start_dim: int = 0) -> int:
    n_crops = 1
    for dim in range(start_dim, len(in_size)):
        rel_dim = dim - start_dim
        n_per_dim = (in_size[dim] - 2 * margin_size[rel_dim]) / (crop_size[rel_dim] - 2 * margin_size[rel_dim])
        if n_per_dim <= 0:
            n_per_dim = 1
        n_crops *= math.ceil(n_per_dim)
    return int(n_crops)


def _no_crops_per_dim(in_size: Tuple[int, ...], crop_size: List[int], margin_size: List[int], start_dim: int
                      ) -> Tuple[List[int], List[int]]:
    n_crops_per_dim = []
    cum_n_crops_per_dim = [1]
    for dim in reversed(range(start_dim, len(in_size))):
        rel_dim = dim - start_dim
        n_crops = (in_size[dim] - 2 * margin_size[rel_dim]) / (crop_size[rel_dim] - 2 * margin_size[rel_dim])
        if n_crops <= 0:
            n_crops = 1
        n_crops = math.ceil(n_crops)
        n_crops_per_dim.append(int(n_crops))
        cum_n_crops_per_dim.append(n_crops * cum_n_crops_per_dim[len(in_size) - dim - 1])
    n_crops_per_dim.reverse()
    cum_n_crops_per_dim.reverse()
    return n_crops_per_dim, cum_n_crops_per_dim


def _crop_inds(crop_ind: int, cum_n_crops_per_dim: List[int]) -> List[int]:
    rem = crop_ind
    result = []
    for dim in range(1, len(cum_n_crops_per_dim)):
        result.append(rem // cum_n_crops_per_dim[dim])
        rem = rem % cum_n_crops_per_dim[dim]
    return result


def _coord(crop_ind: int, crop_size: int, marg: int, in_size: int) -> Tuple[slice, slice]:
    start_ind = crop_ind * (crop_size - 2 * marg)
    start_valid = marg
    end_valid = crop_size - marg
    if start_ind >= in_size - crop_size:
        start_valid = crop_size + start_ind - in_size + marg
        start_ind = in_size - crop_size
        end_valid = crop_size
    if crop_ind == 0:
        start_valid = 0
    return slice(int(start_ind), int(start_ind + crop_size)), slice(int(start_valid), int(end_valid))


def _coords(crop_inds: List[int], crop_sizes: List[int], margs: List[int], in_sizes: List[int], start_dim: int
            ) -> Tuple[List[slice], List[slice]]:
    crop_coords = []
    valid_coords = []
    for i in range(start_dim):
        crop_coords.append(slice(0, in_sizes[i]))
        valid_coords.append(slice(0, in_sizes[i]))
    for i in range(start_dim, len(in_sizes)):
        reli = i - start_dim
        c, d = _coord(crop_inds[reli], crop_sizes[reli], margs[reli], in_sizes[i])
        crop_coords.append(c)
        valid_coords.append(d)
    return crop_coords, valid_coords


def split_with_margin(size: Tuple[int, ...], crop_size: List[int], margin: List[int]
                      ) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    assert len(crop_size) == len(margin)
    for c, m in zip(crop_size, margin):
        assert c > (m * 2), "margin must be smaller than half of crop_size"
    n_crops = _no_crops(size, crop_size, margin, 0)
    source_coords, valid_coords, destin_coords = [], [], []
    for i in range(n_crops):
        n_per_dim, cum = _no_crops_per_dim(size, crop_size, margin, 0)
        crop_idx = _crop_inds(i, cum)
        source_slices, valid_slices = _coords(crop_idx, crop_size, margin, list(size), 0)
        destin_slices = [slice(s.start + v.start, s.start + v.stop) for s, v in zip(source_slices, valid_slices)]
        source_coords.append(tuple(source_slices))
        valid_coords.append(tuple(valid_slices))
        destin_coords.append(tuple(destin_slices))
    return source_coords, valid_coords, destin_coords


def process_in_chunks(
    image: torch.Tensor,
    output: torch.Tensor,
    process: Any,
    patch_size: List[int],
    patch_margin: List[int],
) -> torch.Tensor:
    """Run model on overlapping patches and write valid regions into output."""
    assert len(image.shape) == len(output.shape)
    assert (len(image.shape) - 2) == len(patch_size)
    assert len(patch_margin) == len(patch_size)
    spatial_shape = image.shape[2:]
    chunk_coords = split_with_margin(spatial_shape, patch_size, patch_margin)
    semicol = (slice(None, None),)
    for source_c, valid_c, destin_c in zip(*chunk_coords):
        crop = image[semicol + semicol + source_c]
        proc_crop = process(crop)
        if len(proc_crop.shape) == 3:
            proc_crop = proc_crop.unsqueeze(1)
        output[semicol + semicol + destin_c] = proc_crop[semicol + semicol + valid_c]
    return output


def config_logger(log_file: str | None = None) -> None:
    class Formatter(logging.Formatter):
        info_fmt = "\x1b[32;1m%(asctime)s [%(name)s]\x1b[0m %(message)s"
        err_fmt = "\x1b[31;1m%(asctime)s [%(name)s] [%(levelname)s]\x1b[0m %(message)s"

        def format(self, record: logging.LogRecord) -> str:
            self._style._fmt = self.err_fmt if record.levelno > logging.INFO else self.info_fmt
            return super().format(record)

    root = logging.getLogger()
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] [%(levelname)s]> %(message)s"))
        root.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(Formatter())
    root.addHandler(ch)
    root.setLevel(logging.INFO)
