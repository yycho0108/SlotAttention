#!/usr/bin/env python3

from typing import Dict, Any, Optional, Union
from pathlib import Path
from os import PathLike

import torch as th
nn = th.nn


def get_device(device: str = 'auto') -> th.device:
    if device == 'auto':
        if th.cuda.is_available():
            device = th.device('cuda')
        else:
            device = th.device('cpu')
    return th.device(device)


def ensure_dir(path: Union[str, PathLike]) -> Path:
    """ensure that directory exists."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_new_dir(root: Union[str, PathLike]) -> Path:
    """Get new runtime directory."""
    root = Path(root).expanduser()
    index = len([d for d in root.glob('run-*') if d.is_dir()])
    path = ensure_dir(root / F'run-{index:03d}')
    return path


def save_ckpt(ckpt_file: str, model: nn.Module,
              optimizer: Optional[th.optim.Optimizer] = None):
    ckpt_file = Path(ckpt_file)
    ensure_dir(ckpt_file.parent)
    save_dict: Dict[str, Any] = {}
    save_dict['model'] = model.state_dict()
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    th.save(save_dict, str(ckpt_file))


def load_ckpt(ckpt_file: str, model: nn.Module,
              optimizer: Optional[th.optim.Optimizer] = None):
    ckpt_file = Path(ckpt_file)
    save_dict = th.load(str(ckpt_file))
    model.load_state_dict(save_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(save_dict['optimizer'])
