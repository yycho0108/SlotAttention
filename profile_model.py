#!/usr/bin/env python3

from typing import Callable, Tuple, Iterable
from functools import partial
import torch as th
nn = th.nn
F = nn.functional
import einops
from model import SlotAttentionAE

import time
from contextlib import contextmanager
from torch.profiler import profile, record_function, ProfilerActivity


def main():
    device = th.device('cuda')
    resolution: Tuple[int, int] = (128, 128)
    model = SlotAttentionAE(resolution).to(device).train()
    img = th.zeros((16, 3,) + resolution, device=device)

    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function('model_inference'):
            for _ in range(16):
                out = model(img)
    print(
        prof.key_averages().table(
            sort_by='cuda_time_total',
            row_limit=16))


if __name__ == '__main__':
    main()
