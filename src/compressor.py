import bisect
import math

import pyvips
import pywuffs
import torch
from torch import Tensor, nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm.auto import tqdm


class PatchedImageFast(Dataset):
    """Faster version for smaller images"""

    config = pywuffs.aux.ImageDecoderConfig()
    config.enabled_decoders = [pywuffs.ImageDecoderType.PNG]
    config.pixel_format = pywuffs.PixelFormat.BGR
    image_decoder = pywuffs.aux.ImageDecoder(config)

    def __init__(
        self,
        img_path: str,
        patch_size: int = 128,
        pad_value: float = 0.0,
    ) -> None:
        img = self.read_image(img_path)
        img = self.pad(img, patch_size, pad_value)
        self.img = self.tile(img, patch_size)
        self.n_y = self.img.size(1)
        self.n_x = self.img.size(2)

    def __len__(self) -> int:
        return self.n_x * self.n_y

    def __getitem__(self, index: int) -> Tensor:
        idx_y = index // self.n_x
        idx_x = index % self.n_x
        return self.img[:, idx_y, idx_x]

    @classmethod
    def read_image(cls, path: str) -> Tensor:
        data = cls.image_decoder.decode(path).pixbuf
        data = torch.from_numpy(data)
        data = data.flip(2)  # BGR -> RGB
        return data.permute(2, 0, 1)

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


class PatchedImageChunk(Dataset):
    def __init__(
        self,
        img: pyvips.Image,
        patch_size: int = 128,
        pad_value: int = 0,
    ) -> None:
        self.img = img
        self._img_arr = None
        h = img.height
        w = img.width
        self.n_x = math.ceil(w / patch_size)
        self.n_y = math.ceil(h / patch_size)
        self.patch_size = patch_size
        self.pad_value = pad_value

    def __len__(self) -> int:
        return self.n_x * self.n_y

    def __getitem__(self, index: int) -> Tensor:
        idx_y = index // self.n_x
        idx_x = index % self.n_x
        return self.img_arr[:, idx_y, idx_x]

    @property
    def img_arr(self) -> Tensor:
        if self._img_arr is not None:
            return self._img_arr
        img = torch.from_numpy(self.img.numpy()[..., :3])  # some png are rgba
        img = img.permute(2, 0, 1)  # h,w,c -> c,h,w
        img = self.pad(img, self.patch_size, self.pad_value)
        self._img_arr = self.tile(img, self.patch_size)  # 3,n_y,n_x,p,p
        return self._img_arr

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


class PatchedImageEfficient(ConcatDataset):
    """Memory efficient version for large images"""

    def __init__(
        self,
        img_path: str,
        patch_size: int = 128,
        pad_value: int = 0,
        num_splits: int = 4,
    ) -> None:
        img = pyvips.Image.new_from_file(img_path, access="sequential")  # sequential access disables cache
        self.n_x = math.ceil(img.width / patch_size)  # for compatibility
        self.n_y = math.ceil(img.height / patch_size)  # for compatibility
        height = round(img.height / num_splits / patch_size) * patch_size
        imgs = []
        for i in range(num_splits):
            h = height if i < (num_splits - 1) else img.height - height * i
            imgs.append(img.crop(0, height * i, img.width, h))
        datasets = [PatchedImageChunk(im, patch_size=patch_size, pad_value=pad_value) for im in imgs]
        super().__init__(datasets=datasets)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            if sample_idx == 0:
                self.datasets[dataset_idx - 1]._img_arr = None  # deletes the previous image from memory
        return self.datasets[dataset_idx][sample_idx]


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
    def compress(self, img: PatchedImageFast) -> Tensor:
        loader = DataLoader(img, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        with autocast(enabled=self.amp):
            features = torch.concat([self.encoder(self.transforms(x.to(self.device))) for x in tqdm(loader)], dim=0)
        features = features.squeeze()
        features = features.view(img.n_y, img.n_x, features.shape[-1])  # (xy, ch) -> (y, x, ch)
        return features.permute(2, 0, 1)  # (y, x, ch) -> (ch, y, x)
