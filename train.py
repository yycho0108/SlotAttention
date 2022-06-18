#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass
import numpy as np
import torch as th
nn = th.nn
F = nn.functional
from functools import partial
from tqdm.auto import tqdm
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize)

from model import SlotAttentionAE
from dataset import CLEVRDataset
from util import (get_new_dir, ensure_dir, load_ckpt, save_ckpt)

from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    batch_size: int = 22
    num_workers: int = 4
    shuffle: bool = True
    num_epochs: int = 4
    img_size: Tuple[int, int] = (128, 128)
    num_slots: int = 8

    learning_rate: float = 3e-4
    decay_steps: int = 10000
    decay_rate: float = 0.5
    warmup_steps: int = 1000
    load_ckpt: str = '/tmp/slot-attention/run-009/last.pt'


def google_learning_rate(
        global_step: int,
        warmup_steps: int,
        base_learning_rate: float,
        decay_steps: int,
        decay_rate: float):
    # Learning rate warm-up.
    if global_step < warmup_steps:
        learning_rate = base_learning_rate * float(
            global_step) / float(warmup_steps)
    else:
        learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate **
                                     (float(global_step) / float(decay_steps)))
    return learning_rate


def main():
    cfg = Config()
    #print(google_learning_rate(10000,
    #    cfg.warmup_steps,
    #    cfg.learning_rate,
    #    cfg.decay_steps,
    #    cfg.decay_rate))
    path = get_new_dir('/tmp/slot-attention')
    device: th.device = th.device('cuda')
    model = SlotAttentionAE(cfg.img_size, cfg.num_slots).to(device)
    # model = th.jit.script(model)
    transform = Compose([
        ToTensor(),
        Resize(cfg.img_size),
        Normalize(0.5, 0.5, True),
    ])
    dataset = CLEVRDataset('/media/ssd/datasets/CLEVR/CLEVR_v1.0/', 'train',
                           transform=transform)
    val_dataset = CLEVRDataset(
        '/media/ssd/datasets/CLEVR/CLEVR_v1.0/',
        'val',
        transform=transform)
    criterion = nn.MSELoss()
    data_loader = th.utils.data.DataLoader(dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.num_workers,
                                           shuffle=cfg.shuffle,
                                           pin_memory=True)
    val_data_loader = th.utils.data.DataLoader(val_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=False)
    optimizer = th.optim.Adam(model.parameters(),
                              lr=cfg.learning_rate)
    load_ckpt(cfg.load_ckpt, model, optimizer)
    # NOTE(ycho): LambdaLR should return a
    # _multiplicative factor_!!!
    scheduler = th.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            google_learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_rate=cfg.decay_rate,
            decay_steps=cfg.decay_steps,
            # base_learning_rate=cfg.learning_rate
            base_learning_rate=1.0
        ))
    log_period: int = 64
    losses = []

    # FIXME(ycho): avoid hard-coding `step`...
    # somehow load from previous ckpt if possible
    # this matter because of learning rate scheduling :P
    # step: int = 0
    step: int = int(17.47e3)
    writer = SummaryWriter(path)
    th.backends.cudnn.benchmark=True
    try:
        for epoch in range(cfg.num_epochs):
            model.train()

            for images in tqdm(data_loader):
                images = images.to(device)

                pred_images, _, _, _ = model(images)
                loss = criterion(images, pred_images)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().numpy())

                if step % log_period == 0:
                    if len(losses) > 0:
                        writer.add_scalar('loss', np.mean(losses), step)
                        writer.add_scalar(
                            'lr', scheduler.get_last_lr()[0], step)

                    ## +++ VALIDATION +++
                    model.eval()
                    for images in val_data_loader:
                        images = images.to(device)
                        pred_images, _, _, _ = model(images)
                        val_loss = criterion(
                            images, pred_images).detach().cpu().numpy()
                        writer.add_scalar('val_loss', np.mean(val_loss),
                                          step)

                        # NOTE(ycho): for now, we're not going to bother
                        # going through the whole dataset.
                        break
                    model.train()
                    ## ++++++++++++++++++
                    # FIXME(ycho): duplicated image mean/std values
                    # for normalization/unnormalization
                    writer.add_images('input_images', images * 0.5 + 0.5, step)
                    writer.add_images(
                        'out_images', pred_images * 0.5 + 0.5, step)
                    losses = []
                step += 1
                scheduler.step()

            # Save after every epoch.
            save_ckpt(path / F'epoch-{epoch:02d}.pt', model, optimizer)
    finally:
        save_ckpt(path / 'last.pt', model, optimizer)


if __name__ == '__main__':
    main()
