import math

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torchvision.models.convnext import CNBlock  # type: ignore[import-untyped]
from tqdm.auto import tqdm

from .utils import AverageMeter


class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling

    Apply multiple poolings to a feature of arbitrary size and generate a feature of fixed size

    .._He et al., (2014)
        https://arxiv.org/abs/1406.4729
    """

    def __init__(self, num_pools: tuple[int, ...] = (1, 4, 16), mode: str = "max") -> None:
        super().__init__()
        if mode == "max":
            pool_func = nn.AdaptiveMaxPool2d
        elif mode == "avg":
            pool_func = nn.AdaptiveAvgPool2d
        else:
            raise NotImplementedError
        self.poolers = nn.ModuleList()
        for p in num_pools:
            root = math.sqrt(p)
            assert root.is_integer(), root
            self.poolers.append(pool_func(output_size=int(root)))

    def forward(self, x: Tensor) -> Tensor:
        bs, c, _, _ = x.shape
        return torch.cat([pooler(x).view(bs, c, -1) for pooler in self.poolers], dim=2)


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_pools: tuple[int, ...] = (1, 4, 16),
        device: torch.device = torch.device("cuda"),
        amp: bool = True,
    ) -> None:
        super().__init__()
        self.stages = nn.Sequential(
            CNBlock(dim=in_channels, layer_scale=1e-6, stochastic_depth_prob=0.1),
            CNBlock(dim=in_channels, layer_scale=1e-6, stochastic_depth_prob=0.1),
            CNBlock(dim=in_channels, layer_scale=1e-6, stochastic_depth_prob=0.1),
        )
        self.spatial_pyramid_pool = SpatialPyramidPooling(num_pools)
        in_channels = sum(num_pools) * in_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6), nn.Flatten(start_dim=1), nn.Linear(in_channels, num_classes)
        )
        self.optimizer = self.configure_optimizer()
        self.criterion = self.configure_criterion()
        self.scaler = GradScaler(enabled=amp)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def configure_optimizer(self) -> optim.Optimizer:
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return self.optimizer

    def configure_criterion(self) -> nn.Module:
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        return self.criterion

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(device=self.device)
        x = self.stages(x)
        x = self.spatial_pyramid_pool(x)
        x = x.flatten(start_dim=1)  # (bs, ch, sum(pools)) -> (bs, ch * sum(pools))
        x = self.classifier(x)
        return x

    def train_step(self, x: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()
        with autocast(enabled=self.scaler.is_enabled()):
            pred = self.forward(x)
            loss = self.criterion(input=pred, target=label.to(pred.device))
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, pred

    @torch.no_grad()
    def eval_step(self, x: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
        pred = self.forward(x)
        loss = self.criterion(input=pred, target=label.to(pred.device))
        return loss, pred

    def train_single_epoch(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        preds = []
        labels = []
        loss = AverageMeter()
        pbar = tqdm(data_loader, leave=False)
        for x, label in pbar:
            _loss, pred = self.train_step(x, label)
            loss.update(_loss.item())
            preds.append(pred.detach().cpu().numpy())
            labels.append(label.numpy())
            pbar.set_description(f"train loss: {loss.average:.3f}")

        return loss.average, np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)

    def evaluate(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        preds = []
        labels = []
        loss = AverageMeter()
        pbar = tqdm(data_loader, leave=False)
        for x, label in pbar:
            _loss, pred = self.eval_step(x, label)
            loss.update(_loss.item())
            preds.append(pred.cpu().numpy())
            labels.append(label.numpy())
            pbar.set_description(f"eval loss: {loss.average:.3f}")

        return loss.average, np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)
