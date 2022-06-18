#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
import torch as th
nn = th.nn
F = nn.functional
from model import SlotAttentionAE
from dataset import CLEVRDataset
from tqdm.auto import tqdm
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize)


@dataclass
class Config:
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    learning_rate: float = 3e-4
    num_epochs: int = 4


def main():
    cfg = Config()

    device: th.device = th.device('cuda')
    model = SlotAttentionAE((128, 128), 8).to(device)
    dataset = CLEVRDataset('/media/ssd/datasets/CLEVR/CLEVR_v1.0/', 'train',
                           transform=Compose([
                               ToTensor(),
                               Resize((128, 128)),
                               Normalize(0.5, 0.5, True),
                           ]))
    criterion = nn.MSELoss()
    data_loader = th.utils.data.DataLoader(dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.num_workers,
                                           # collate_fn=collate_fn,
                                           shuffle=cfg.shuffle)
    optimizer = th.optim.Adam(model.parameters(),
                              lr=cfg.learning_rate)
    log_period: int = 128
    losses = []

    step: int = 0
    for epoch in range(cfg.num_epochs):
        model.train()
        for images in tqdm(data_loader):
            # with th.autograd.set_detect_anomaly(True):
            images = images.to(device)

            pred_images, _, _, _ = model(images)
            loss = criterion(images, pred_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().cpu().numpy())

            if step % log_period == 0:
                if len(losses) > 0:
                    print('loss', np.mean(losses))
                losses = []
            step += 1


if __name__ == '__main__':
    main()
