#!/usr/bin/env python3

from dataclasses import dataclass
import torch as th
nn = th.nn
F = nn.functional
from model import SlotAttention
from dataset import CLEVRDataset
from tqdm.auto import tqdm


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
    model = SlotAttention(4, 128).to(device)
    dataset = CLEVRDataset('/media/ssd/datasets/CLEVR/CLEVR_v1.0/')
    criterion = nn.MSELoss()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=cfg.batch_size,
                                              num_workers=cfg.num_workers,
                                              collate_fn=collate_fn,
                                              shuffle=cfg.shuffle)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.learning_rate)
    log_period: int = 128
    losses = []

    step: int = 0
    for epoch in range(args.num_epochs):
        model.train()
        for images in tqdm(data_loader):
            pred_images = model(images)
            loss = criterion(images, pred_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_period == 0:
                if len(losses) > 0:
                    print('loss', th.mean(losses).item())
                losses = []
            step += 1


if __name__ == '__main__':
    main()
