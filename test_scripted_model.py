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


@contextmanager
def timer(msg: str = ''):
    try:
        t0 = time.time()
        yield
    finally:
        t1 = time.time()
        dt = (t1 - t0)
        print(msg.format(dt))


def main():
    device = th.device('cuda')
    resolution: Tuple[int, int] = (128, 128)
    model = SlotAttentionAE(resolution).to(device).eval()
    # NOTE(ycho): makes no difference
    # with th.jit.optimized_execution(True):
    smodel = th.jit.script(model).to(device).eval()

    img = th.zeros((16, 3,) + resolution, device=device)
    with th.no_grad():
        with timer('Took {} seconds'):
            out = model(img)
        with timer('Took {} seconds'):
            # NOTE(ycho): makes no difference
            # with th.jit.optimized_execution(True):
            out = smodel(img)

        with timer('Took {} seconds'):
            for _ in range(128):
                out = model(img)
        with timer('Took {} seconds'):
            # NOTE(ycho): makes no difference
            # with th.jit.optimized_execution(True):
            for _ in range(128):
                out = smodel(img)


if __name__ == '__main__':
    main()
