import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms.functional import InterpolationMode
from tqdm.auto import tqdm

from .utils import AverageMeter


def off_diagnoal(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        lambd: float = 0.0051,
        amp: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.backbone = resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()  # 2048
        self.lambd = lambd

        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=8192, bias=False),
            nn.BatchNorm2d(num_features=8192),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8192, out_features=8192, bias=False),
            nn.BatchNorm2d(num_features=8192),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=8192, out_features=latent_dim, bias=False),
        )
        self.bn = nn.BatchNorm1d(num_features=latent_dim, affine=False)
        self.optimizer = self.configure_optimizer()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)
        self.device = device

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """
        Note:
            The original implelemtation uses LARS optimizer
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1.5e-6)
        return self.optimizer

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x1.size(0)
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        c.div_(bs)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagnoal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss, on_diag, off_diag

    def train_step(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss, on_diag, off_diag = self.forward(x1, x2)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, on_diag, off_diag

    def train_single_epoch(self, data_loader: DataLoader) -> tuple[float, float, float]:
        loss = AverageMeter()
        on_diag = AverageMeter()
        off_diag = AverageMeter()
        pbar = tqdm(total=len(data_loader))
        for x1, x2 in data_loader:
            _loss, _on_diag, _off_diag = self.train_step(x1, x2)
            if not torch.isfinite(_loss):
                raise RuntimeError(f"loss not finite: {_loss}")
            loss.update(_loss.item())
            on_diag.update(_on_diag.item())
            off_diag.update(_off_diag.item())
            pbar.set_description(f"loss: {loss.average:.3f} on-diag: {on_diag.average} off-diag: {off_diag.average}")
            pbar.update()
        pbar.close()
        return loss.average, on_diag.average, off_diag.average

    @torch.no_grad()
    def eval_step(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.cuda.amp.autocast():
            loss, on_diag, off_diag = self.forward(x1, x2)
        return loss, on_diag, off_diag

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        loss = AverageMeter()
        on_diag = AverageMeter()
        off_diag = AverageMeter()
        pbar = tqdm(total=len(data_loader))
        for x1, x2 in data_loader:
            _loss, _on_diag, _off_diag = self.eval_step(x1, x2)
            loss.update(_loss.item())
            on_diag.update(_on_diag.item())
            off_diag.update(_off_diag.item())
            pbar.set_description(f"loss: {loss.average:.3f} on-diag: {on_diag.average} off-diag: {off_diag.average}")
            pbar.update()
        pbar.close()
        return loss.average, on_diag.average, off_diag.average


class Transform:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def __init__(self, image_size: int = 128):
        self.image_size = image_size
        self.transform = self.configure_transform()
        self.transform_prime = self.configure_transform_prime()

    def configure_transform(self) -> transforms.Compose:
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambd=lambda i: i / 255.0),
                transforms.RandomResizedCrop(
                    size=self.image_size, scale=(0.5, 1), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.GaussianBlur(kernel_size=3),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return self.transform

    def configure_transform_prime(self) -> transforms.Compose:
        self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=self.image_size, scale=(0.5, 1), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8
                ),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(kernel_size=3),
                    ],
                    p=0.1,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)],
                    p=0.2,
                ),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return self.transform_prime

    @classmethod
    def denormalize(cls, x: torch.Tensor) -> torch.Tensor:
        denorm = transforms.Compose(
            [
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / cls.std),
                transforms.Normalize(mean=-cls.mean, std=[1.0, 1.0, 1.0]),
            ]
        )
        return denorm(x)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
