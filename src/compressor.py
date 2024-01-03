import math

import torch
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm.auto import tqdm


class PatchedImageWuffs(Dataset):
    """Much faster but load the entire image in memory"""

    def __init__(
        self,
        img: Tensor,
        patch_size: int = 128,
        pad_value: int = 0,
        dtype: torch.dtype = torch.float,
    ) -> None:
        img = self.pad(img, patch_size, pad_value)
        self.img = self.tile(img, patch_size)
        self.n_y = self.img.size(1)
        self.n_x = self.img.size(2)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.n_x * self.n_y

    def __getitem__(self, index: int) -> Tensor:
        idx_y = index // self.n_x
        idx_x = index % self.n_x
        return self.img[:, idx_y, idx_x]

    @staticmethod
    def tile(img: Tensor, patch_size: int) -> Tensor:
        return img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)  # 3,squeezed_h,squeezed_w,p,p

    @staticmethod
    def pad(img: Tensor, patch_size: int, pad_value: int) -> Tensor:
        """make height and width multiple of patch size"""
        _, h, w = img.shape
        target_h = int(math.ceil(float(h) / patch_size) * patch_size)
        target_w = int(math.ceil(float(w) / patch_size) * patch_size)
        pad = (0, target_w - w, 0, target_h - h, 0, 0)
        return F.pad(img, pad, value=pad_value)


class Compressor:
    def __init__(
        self,
        encoder: nn.Module,
        transforms: transforms.Compose,
        batch_size: int,
        device: torch.device | None = None,
        num_workers: int = 0,
        amp: bool = True,
    ) -> None:
        self.encoder = encoder.eval()
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def compress(self, img: PatchedImageWuffs) -> Tensor:
        loader = DataLoader(img, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        with autocast(enabled=self.amp):
            features = torch.concat([self.encoder(self.transforms(x.to(self.device))) for x in tqdm(loader)], dim=0)
        features = features.squeeze()
        features = features.view(img.n_y, img.n_x, features.shape[-1])  # (xy, ch) -> (y, x, ch)
        return features.permute(2, 0, 1)  # (y, x, ch) -> (ch, y, x)
