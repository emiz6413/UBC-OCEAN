import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.models import resnet50  # type: ignore[import-untyped]
from torchvision.transforms.functional import (  # type: ignore[import-untyped]
    InterpolationMode,
    normalize,
)
from tqdm.auto import tqdm

from .utils import AverageMeter


class BarlowTwins(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        projector_dim: int = 8192,
        lambd: float = 0.0051,
        amp: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__()
        self.backbone = self.configure_backbone()
        self.lambd = lambd

        self.projector = nn.Sequential(
            nn.Linear(in_features=2048, out_features=projector_dim, bias=False),
            nn.BatchNorm1d(num_features=projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=projector_dim,
                out_features=projector_dim,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=projector_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=projector_dim, out_features=latent_dim, bias=False),
        )
        self.bn = nn.BatchNorm1d(num_features=latent_dim, affine=False)
        self.optimizer = self.configure_optimizer()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)
        self.device = device

    def configure_backbone(self) -> torch.nn.Module:
        self.backbone = resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()  # 2048
        return self.backbone

    def configure_optimizer(self) -> torch.optim.Optimizer:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-6)
        return self.optimizer

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)
        bs = x1.size(0)
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))

        # cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        c.div_(bs)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagnoal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss, on_diag, off_diag

    @staticmethod
    def off_diagnoal(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
            pbar.set_description(
                f"loss: {loss.average:.3f} on-diag: {on_diag.average:.3f} off-diag: {off_diag.average:.3f}"
            )
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
            pbar.set_description(
                f"loss: {loss.average:.3f} on-diag: {on_diag.average:.3f} off-diag: {off_diag.average:.3f}"
            )
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
                    size=self.image_size,
                    scale=(0.25, 1),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambd=lambda i: i.clamp(min=0, max=1)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.GaussianBlur(kernel_size=3),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return self.transform

    def configure_transform_prime(self) -> transforms.Compose:
        self.transform_prime = transforms.Compose(
            [
                transforms.Lambda(lambd=lambda i: i / 255.0),
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(0.25, 1),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambd=lambda i: i.clamp(min=0, max=1)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
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
    def normalize(cls, x: torch.Tensor) -> torch.Tensor:
        return normalize(x, mean=cls.mean, std=cls.std)

    @classmethod
    def denormalize(cls, x: torch.Tensor) -> torch.Tensor:
        x = normalize(x, mean=[0.0, 0.0, 0.0], std=1 / cls.std)
        x = normalize(x, mean=-cls.mean, std=[1.0, 1.0, 1.0])
        return x

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1 = self.transform(x)
        x2 = self.transform_prime(x)
        return x1, x2
