#!/usr/bin/env python3

"""from https://github.com/vadimkantorov/yet_another_pytorch_slot_attention/blo
b/master/clevr.py."""

import os
import json
import torch as th
import torchvision as tv
import numpy as np
from pathlib import Path
from typing import Callable, List, Dict, Any, Union


class CLEVRDataset(tv.datasets.VisionDataset):
    def __init__(self, root: str, split_name: str,
                 transform=tv.transforms.ToTensor(),
                 image_loader=tv.datasets.folder.default_loader,
                 filter_fn: Callable[[str], bool] = None):
        super().__init__(root, transform=transform)

        self.image_loader = image_loader
        root: Path = Path(root)
        scenes_json_path = root / 'scenes' / F'CLEVR_{split_name}_scenes.json'
        images_split_dir = root / 'images' / split_name
        self.image_paths: List[Path] = sorted(images_split_dir.iterdir())

        if scenes_json_path.exists():
            with open(str(scenes_json_path.resolve()), 'r') as fp:
                scenes = json.load(fp)['scenes']
            metadata = {s['image_filename']: s['objects']
                        for s in scenes}
        else:
            metadata = {
                image_path.name: {}
                for image_path in self.image_paths
            }
        self.metadata: Dict[str, Any] = metadata

        if filter_fn is not None:
            self.image_paths = filter(lambda image_path: filter_fn(
                self.metadata[image_path.name]), self.image_paths)
            self.metadata = {
                k.name: self.metadata[k.name] for k in self.image_paths}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, th.Tensor]:
        image_path: str = str(self.image_paths[index])
        mask_path: str = image_path.replace(
            'images', 'masks').replace(
            '.png', '.npy')

        image = self.image_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)

        if Path(mask_path).exists():
            mask = torch.as_tensor(np.load(mask_path))
        else:
            mask = None

        return dict(
            image=image,
            mask=mask,
            # image_name=str(Path(image_path).name)
        )


def main():
    dataset = CLEVRDataset('/media/ssd/datasets/CLEVR/CLEVR_v1.0/')


if __name__ == '__main__':
    main()
