from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import io, transforms


class UBCODataset(Dataset):
    def __init__(self, image_paths: list[Path], transforms: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.read_image(index)
        return self.transforms(img)

    def read_image(self, idx: int) -> torch.FloatTensor:
        return io.read_image(str(self.image_paths[idx])).float()
