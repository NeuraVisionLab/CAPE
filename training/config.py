"""
Config parsing for CREMI 2D training. Loads YAML and exposes an args-like object.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from .utils import yaml_read


def merge_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one level of nested dict."""
    merged = {}
    for key, value in config.items():
        if isinstance(value, dict):
            merged.update(value)
        else:
            merged[key] = value
    return merged


def parse_config(
    config_file: str = "config.yaml",
    resume: str = "last",
    training_order: str = "fold_by_fold",
    tensorboard: bool = True,
    argv: Optional[List[str]] = None,
) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default=config_file)
    parser.add_argument("--resume", type=str, default=resume, choices=("best_loss", "best_quality", "last", "no"))
    parser.add_argument(
        "--training_order",
        type=str,
        choices=("epoch_by_epoch", "fold_by_fold"),
        default=training_order,
    )
    parser.add_argument("--tensorboard", action="store_true", default=tensorboard)
    parser.add_argument("--fold", type=int, nargs="*", default=None, metavar="N", help="Fold index (or indices) to train")
    args, _ = parser.parse_known_args(argv)

    config_full = yaml_read(args.config_file)
    config_params = merge_config(config_full)
    for key, value in config_params.items():
        setattr(args, key, value)
    return args
