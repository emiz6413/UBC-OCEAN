from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.spectral_norm import spectral_norm


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """Self-attention layer

        Note:
            key is pooled by maxpooling with kernel & stride = 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.query_projection = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False))
        self.key_projection = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.value_projection = spectral_norm(nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False))
        self.out_conv = spectral_norm(nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self attention + skip connection"""
        bs, c, w, h = x.size()
        wh = w * h

        query = self.query_projection(x)
        query = query.view(bs, -1, wh)  # bs, C/8, wh
        query = query.permute(0, 2, 1)  # bs, wh, C/8

        key = self.key_projection(x)
        key = self.maxpool(key)
        key = key.view(bs, -1, wh // 4)  # bs, C/8, wh/4

        dot = self.softmax(torch.bmm(query, key))  # bs, wh, wh/4
        dot = dot.permute(0, 2, 1)  # bs, wh/4, wh

        value = self.value_projection(x)
        value = self.maxpool(value)
        value = value.view(bs, -1, wh // 4)  # bs, C/2, wh/4

        attn = torch.bmm(value, dot)  # bs, channel, wh
        attn = attn.view(bs, -1, w, h)
        out = self.out_conv(attn)

        return self.gamma * out + x


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool,
        downsample: bool,
        batch_norm: bool,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        self.skip_proj = False
        if in_channels != out_channels or upsample or downsample:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
            self.skip_proj = True

        self.maybe_upsample = partial(F.interpolate, scale_factor=2) if upsample else nn.Identity()
        self.maybe_downsample = partial(F.avg_pool2d, kernel_size=2) if downsample else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.maybe_bn1 = nn.BatchNorm2d(in_channels) if batch_norm else nn.Identity()
        self.maybe_bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        x = self.maybe_bn1(x)
        x = self.maybe_upsample(x)  # type: ignore
        x = self.conv1(x)
        x = self.maybe_bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.maybe_downsample(x)  # type: ignore

        if self.skip_proj:
            skip = self.maybe_upsample(skip)  # type: ignore
            skip = self.maybe_downsample(skip)  # type: ignore
            skip = self.conv_sc(skip)

        return x + skip


class GeneratorBlock(Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = True,
        downsample: bool = False,
        batch_norm: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            upsample=upsample,
            downsample=downsample,
            batch_norm=batch_norm,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )


class DiscriminatorBlock(Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
        downsample: bool = True,
        batch_norm: bool = False,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, upsample, downsample, batch_norm, kernel_size, padding, stride)


class Generator128(nn.Module):
    def __init__(self, latent_dim: int = 120, middle_dim: int = 96, sa_idx: int = 4) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.sa_idx = sa_idx
        self.pre_linear = spectral_norm(nn.Linear(self.latent_dim, 4 * 4 * 16 * middle_dim))
        self.first_channels = 16 * middle_dim
        self.convs = nn.Sequential(
            GeneratorBlock(16 * middle_dim, 16 * middle_dim),  # 4 -> 8
            GeneratorBlock(16 * middle_dim, 8 * middle_dim),  # 8 -> 16
            GeneratorBlock(8 * middle_dim, 4 * middle_dim),  # 16 -> 32
            GeneratorBlock(4 * middle_dim, 2 * middle_dim),  # 32 -> 64
            SelfAttention(2 * middle_dim),
            GeneratorBlock(2 * middle_dim, 1 * middle_dim),  # 64 -> 128
            nn.BatchNorm2d(middle_dim, eps=1e-4),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(middle_dim, 3, kernel_size=3, padding=1)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_linear(x)
        x = x.view(-1, 4, 4, self.first_channels).permute(0, 3, 1, 2)  # bs, channel, 4, 4
        x = self.convs(x)
        return x


class Encoder128(nn.Module):
    """
    Note:
        This is based on the implementation of Discriminator.
    """

    def __init__(self, latent_dim: int = 128, middle_dim: int = 96, in_channels: int = 3) -> None:
        super().__init__()
        self.pre_convs = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=3, padding=1)),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1)),
            nn.AvgPool2d(kernel_size=2),
        )
        self.pre_skip = spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=1))
        self.pool = nn.AvgPool2d(kernel_size=2)

        self.convs = nn.Sequential(
            DiscriminatorBlock(middle_dim, middle_dim),  # 64 -> 32
            DiscriminatorBlock(middle_dim, 2 * middle_dim),  # 32 -> 16
            SelfAttention(2 * middle_dim),
            DiscriminatorBlock(middle_dim * 2, middle_dim * 4),  # 16 -> 8
            DiscriminatorBlock(middle_dim * 4, middle_dim * 8),  # 8 -> 4
            DiscriminatorBlock(middle_dim * 8, middle_dim * 16),  # 4 -> 2
            DiscriminatorBlock(middle_dim * 16, middle_dim * 16, downsample=False),  # 2 -> 2
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        self.final_projection = spectral_norm(nn.Linear(middle_dim * 16, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.pre_skip(self.pool(x))
        x = self.pre_convs(x) + skip
        x = self.convs(x)  # bs, middlex16, 2, 2
        x = x.flatten(start_dim=2)  # bs, middlex16, 4
        x = x.sum(dim=2)  # bs, middlex16
        x = self.final_projection(x)
        return x


class Discriminator128(Encoder128):
    def __init__(self, middle_dim: int = 96, in_channels: int = 3) -> None:
        super().__init__(latent_dim=1, middle_dim=middle_dim, in_channels=in_channels)