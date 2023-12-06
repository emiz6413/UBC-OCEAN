import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm.auto import tqdm


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
        device: torch.device | None = None,
    ) -> None:
        self.encoder = encoder.eval()
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    @torch.no_grad()
    def compress(self, img: PatchedImage) -> np.ndarray:
        loader = DataLoader(img, batch_size=self.batch_size, shuffle=False)
        features = np.concatenate([self.encoder(x.to(self.device)).cpu().numpy() for x in tqdm(loader)], axis=0)
        features = features.squeeze()
        features = features.reshape(img.n_y, img.n_x, features.shape[-1])  # (xy, ch) -> (y, x, ch)
        return features.transpose(2, 0, 1)  # (y, x, ch) -> (ch, y, x)
