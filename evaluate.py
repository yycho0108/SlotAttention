#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass
import numpy as np
import torch as th
import cv2
nn = th.nn
F = nn.functional
from functools import partial
from tqdm.auto import tqdm
from torchvision.transforms import (
    Compose, ToTensor, Normalize, Resize)

from model import SlotAttentionAE
from dataset import CLEVRDataset
from util import (get_new_dir, ensure_dir, load_ckpt, save_ckpt)
import einops


@dataclass
class Config:
    batch_size: int = 16
    img_size: Tuple[int, int] = (128, 128)
    num_slots: int = 8
    ckpt_path: str = '/tmp/slot-attention/run-009/last.pt'
    num_workers:int = 4


def main():
    cfg = Config()
    device: th.device = th.device('cuda')
    model = SlotAttentionAE(cfg.img_size, cfg.num_slots).to(device)
    load_ckpt(cfg.ckpt_path, model)

    transform = Compose([
        ToTensor(),
        Resize(cfg.img_size),
        Normalize(0.5, 0.5, True),
    ])
    val_dataset = CLEVRDataset(
        '/media/ssd/datasets/CLEVR/CLEVR_v1.0/', 'val',
        transform=transform)
    val_data_loader = th.utils.data.DataLoader(val_dataset,
                                               batch_size=cfg.batch_size,
                                               num_workers=cfg.num_workers,
                                               shuffle=False)
    model.eval()
    for images in tqdm(val_data_loader):
        images = images.to(device)
        with th.no_grad():
            pred_imgs, recs, msks, _ = model(images)
            msks = msks[...,None,:,:]
            slot_imgs = recs * msks + (1 - msks)  # BxSxCXHXW
            slot_vis = slot_imgs * 0.5 + 0.5
            vis = einops.rearrange(slot_vis,
                    'b s c h w -> (b h) (s w) c')
        vis_np = vis.detach().cpu().numpy()
        cv2.imwrite('/tmp/vis.png', (vis_np * 255).astype(np.uint8) )
        break



if __name__ == '__main__':
    main()
