import math
from functools import partial
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.io import read_image  # type: ignore[import-untyped]


class PatchedImage(Dataset):
    def __init__(
        self, img: Tensor, transforms: transforms.Compose, patch_size: int = 128, pad_value: float = 0.0
    ) -> None:
        self.img = img
        _, h, w = self.img.shape
        self.n_x = math.ceil(w / patch_size)
        self.n_y = math.ceil(h / patch_size)
        self.patch_size = patch_size
        self.pad_value = pad_value
        self.transforms = transforms

    def __len__(self) -> int:
        return self.n_x * self.n_y

    def __getitem__(self, index: int) -> Tensor:
        idx_y = index // self.n_x
        idx_x = index % self.n_x
        x_start = idx_x * self.patch_size
        y_start = idx_y * self.patch_size
        patch = self.img[:, y_start : y_start + self.patch_size, x_start : x_start + self.patch_size]
        patch = self.maybe_pad(patch)
        patch = self.transforms(patch)
        return patch

    def maybe_pad(self, patch: Tensor) -> Tensor:
        if (patch.size(1), patch.size(2)) == (self.patch_size, self.patch_size):
            return patch
        pad = (0, self.patch_size - patch.size(2), 0, self.patch_size - patch.size(1), 0, 0)
        return F.pad(patch, pad, value=self.pad_value)


class Comporessor:
    def __init__(
        self,
        encoder: nn.Module,
        batch_size: int,
        transforms: transforms.Compose,
        patch_size: int = 128,
        pad_value: float = 0.0,
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder
        self.Dataset = partial(PatchedImage, transforms=transforms, patch_size=patch_size, pad_value=pad_value)
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def compress(self, img_path: str | Path) -> Tensor:
        img = read_image(path=str(img_path))
        ds = self.Dataset(img)
        loader = DataLoader(ds, batch_size=self.batch_size)
        features = torch.concat([self.encoder(x.to(self.device)) for x in loader], dim=0)
        tiled = torch.concat(
            [torch.concat([features[ds.n_x * iy + ix] for ix in range(ds.n_x)], dim=2) for iy in range(ds.n_y)], dim=1
        )
        return tiled
