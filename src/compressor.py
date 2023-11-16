import math
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision.io import read_image
from torchvision.transforms import Compose, Lambda


class ImageIterator(IterableDataset):
    def __init__(
        self, img: torch.Tensor, transforms: Compose | None = None, patch_size: int = 128, pad_value: float = 0.0
    ) -> None:
        self.img = img
        _, w, h = self.img.shape
        self.n_x = math.ceil(w / patch_size)
        self.n_y = math.ceil(h / patch_size)
        self.patch_size = patch_size
        self.pad_value = pad_value
        if transforms is None:
            transforms = Compose([Lambda(lambd=lambda i: i)])
        self.transforms = transforms

    def __iter__(self) -> Iterator[torch.Tensor]:
        for idx_y in range(self.n_y):
            for idx_x in range(self.n_x):
                x_start = idx_x * self.patch_size
                y_start = idx_y * self.patch_size
                patch = self.img[:, y_start : y_start + self.patch_size, x_start : x_start + self.patch_size]
                patch = self.maybe_pad(patch)
                yield self.transforms(patch)

    def maybe_pad(self, patch: torch.Tensor) -> torch.Tensor:
        if (patch.size(1), patch.size(2)) == (self.patch_size, self.patch_size):
            return patch
        pad = (0, self.patch_size - patch.size(2), 0, self.patch_size - patch.size(1), 0, 0)
        return F.pad(patch, pad, value=self.pad_value)


class Comporessor:
    def __init__(
        self,
        encoder: nn.Module,
        batch_size: int,
        transforms: Compose,
        patch_size: int = 128,
        pad_value: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.image_iterator = partial(ImageIterator, transforms=transforms, patch_size=patch_size, pad_value=pad_value)
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def compress(self, img_path: str | Path) -> torch.Tensor:
        img = read_image(path=str(img_path))
        it = self.image_iterator(img)
        loader = DataLoader(it, batch_size=self.batch_size)
        features = torch.concat([self.encoder(x.to(self.device)) for x in loader], dim=0)
        tiled = torch.concat(
            [torch.concat([features[it.n_x * iy + ix] for ix in range(it.n_x)], dim=2) for iy in range(it.n_y)], dim=1
        )
        return tiled
