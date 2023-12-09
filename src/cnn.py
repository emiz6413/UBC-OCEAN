import math

import torch
from torch import Tensor, nn
from torchvision.models.convnext import CNBlock  # type: ignore[import-untyped]


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
            pool_func = nn.AdaptiveAvgPool2d  # type: ignore
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
    def __init__(self, in_channels: int, num_classes: int, num_pools: tuple[int, ...] = (1, 4, 16)) -> None:
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.stages(x)
        x = self.spatial_pyramid_pool(x)
        x = x.flatten(start_dim=1)  # (bs, ch, sum(pools)) -> (bs, ch * sum(pools))
        x = self.classifier(x)
        return x
