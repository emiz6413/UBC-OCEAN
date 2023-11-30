from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from .utils import spectral_norm


class SelfAttention(nn.Module):
    def __init__(self, in_channels: int, sn_enabled: bool = True) -> None:
        """Self-attention layer

        Note:
            key is pooled by maxpooling with kernel & stride = 2
        """
        super().__init__()
        self.in_channels = in_channels
        self.query_projection = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False), enabled=sn_enabled
        )
        self.key_projection = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, bias=False), enabled=sn_enabled
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.value_projection = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False), enabled=sn_enabled
        )
        self.out_conv = spectral_norm(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, bias=False), enabled=sn_enabled
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Self attention + skip connection"""
        bs, _, w, h = x.size()
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
        upsample: bool = True,
        downsample: bool = False,
        batch_norm: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        sn_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            enabled=sn_enabled,
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            enabled=sn_enabled,
        )
        if in_channels != out_channels or upsample or downsample:
            self.conv_sc: nn.Module = spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0), enabled=sn_enabled
            )
        else:
            self.conv_sc = nn.Identity()

        self.maybe_upsample = partial(F.interpolate, scale_factor=2) if upsample else nn.Identity()
        self.maybe_downsample = partial(F.avg_pool2d, kernel_size=2) if downsample else nn.Identity()
        self.activation = nn.ReLU(inplace=True)
        self.maybe_bn1 = nn.BatchNorm2d(in_channels, eps=1e-4, momentum=0.1) if batch_norm else nn.Identity()
        self.maybe_bn2 = nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.1) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x

        x = self.maybe_bn1(x)
        x = self.activation(x)
        x = self.maybe_upsample(x)  # type: ignore
        x = self.conv1(x)
        x = self.maybe_bn2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.maybe_downsample(x)  # type: ignore

        skip = self.maybe_upsample(skip)  # type: ignore
        skip = self.conv_sc(skip)
        skip = self.maybe_downsample(skip)  # type: ignore

        return x + skip


class GeneratorBlock(Block):
    """
    Note:
        Using spectral norm in generator & encoder didn't work well in preliminary experiments
    """

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
        sn_enabled: bool = True,
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
            sn_enabled=sn_enabled,
        )


class EncoderBlock(Block):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
        downsample: bool = True,
        batch_norm: bool = True,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        sn_enabled: bool = True,
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
            sn_enabled=sn_enabled,
        )


DiscriminatorBlock = EncoderBlock  # alias


class Generator128(nn.Module):
    def __init__(self, latent_dim: int = 120, middle_dim: int = 96, sn_enabled: bool = False) -> None:
        super().__init__()
        self.latent_dim = latent_dim  # for compatibility
        self.pre_linear = spectral_norm(
            nn.Conv2d(latent_dim, 4 * 4 * 16 * middle_dim, kernel_size=1), enabled=sn_enabled
        )
        self.first_channels = 16 * middle_dim
        self.convs = nn.Sequential(
            GeneratorBlock(16 * middle_dim, 16 * middle_dim, sn_enabled=sn_enabled),  # 4 -> 8
            GeneratorBlock(16 * middle_dim, 8 * middle_dim, sn_enabled=sn_enabled),  # 8 -> 16
            GeneratorBlock(8 * middle_dim, 4 * middle_dim, sn_enabled=sn_enabled),  # 16 -> 32
            GeneratorBlock(4 * middle_dim, 2 * middle_dim, sn_enabled=sn_enabled),  # 32 -> 64
            SelfAttention(2 * middle_dim, sn_enabled=sn_enabled),
            GeneratorBlock(2 * middle_dim, 1 * middle_dim, sn_enabled=sn_enabled),  # 64 -> 128
            nn.BatchNorm2d(middle_dim),  # original code: eps=1e-4
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(middle_dim, 3, kernel_size=3, padding=1), enabled=sn_enabled),
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
        This is based on the original implementation of Discriminator.
    """

    def __init__(
        self, latent_dim: int = 120, middle_dim: int = 96, in_channels: int = 3, sn_enabled: bool = False
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim  # for compatibility
        self.pre_convs = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=3, padding=1), enabled=sn_enabled),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1), enabled=sn_enabled),
            nn.AvgPool2d(kernel_size=2),
        )
        self.pre_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=1), enabled=sn_enabled),
        )

        bn = not sn_enabled
        self.convs = nn.Sequential(
            EncoderBlock(middle_dim, middle_dim, batch_norm=bn, sn_enabled=sn_enabled),  # 64 -> 32
            EncoderBlock(middle_dim, 2 * middle_dim, batch_norm=bn, sn_enabled=sn_enabled),  # 32 -> 16
            SelfAttention(2 * middle_dim, sn_enabled=sn_enabled),
            EncoderBlock(middle_dim * 2, middle_dim * 4, batch_norm=bn, sn_enabled=sn_enabled),  # 16 -> 8
            EncoderBlock(middle_dim * 4, middle_dim * 8, batch_norm=bn, sn_enabled=sn_enabled),  # 8 -> 4
            EncoderBlock(middle_dim * 8, middle_dim * 16, batch_norm=bn, sn_enabled=sn_enabled),  # 4 -> 2
            EncoderBlock(
                middle_dim * 16, middle_dim * 16, batch_norm=bn, downsample=False, sn_enabled=sn_enabled
            ),  # 2 -> 2
            nn.ReLU(inplace=True),
        )

        self.final_projection = spectral_norm(nn.Conv2d(middle_dim * 16, latent_dim, kernel_size=1), enabled=sn_enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.pre_skip(x)
        x = self.pre_convs(x) + skip
        x = self.convs(x)  # bs, middlex16, 2, 2
        x = x.sum(dim=(2, 3), keepdim=True)  # bs, middlex16, 1, 1
        x = self.final_projection(x)
        return x


class Discriminator128(nn.Module):
    def __init__(
        self, latent_dim: int = 120, middle_dim: int = 96, in_channels: int = 3, sn_enabled: bool = True
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim  # for compatibility
        self.pre_convs = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=3, padding=1), enabled=sn_enabled),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(middle_dim, middle_dim, kernel_size=3, padding=1), enabled=sn_enabled),
            nn.AvgPool2d(kernel_size=2),
        )
        self.pre_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2),
            spectral_norm(nn.Conv2d(in_channels, middle_dim, kernel_size=1), enabled=sn_enabled),
        )

        bn = not sn_enabled
        self.x_mappings = nn.Sequential(
            DiscriminatorBlock(middle_dim, middle_dim, batch_norm=bn, sn_enabled=sn_enabled),  # 64 -> 32
            DiscriminatorBlock(middle_dim, middle_dim * 2, batch_norm=bn, sn_enabled=sn_enabled),  # 32 -> 16
            SelfAttention(2 * middle_dim, sn_enabled=sn_enabled),
            DiscriminatorBlock(middle_dim * 2, middle_dim * 4, batch_norm=bn, sn_enabled=sn_enabled),  # 16 -> 8
            DiscriminatorBlock(middle_dim * 4, middle_dim * 8, batch_norm=bn, sn_enabled=sn_enabled),  # 8 -> 4
            DiscriminatorBlock(middle_dim * 8, middle_dim * 16, batch_norm=bn, sn_enabled=sn_enabled),  # 4 -> 2
            DiscriminatorBlock(middle_dim * 16, middle_dim * 16, batch_norm=bn, sn_enabled=sn_enabled),  # 2 -> 1
        )

        self.z_mappings = nn.Sequential(
            DiscriminatorBlock(
                latent_dim * 1,
                middle_dim * 1,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
            DiscriminatorBlock(
                middle_dim * 1,
                middle_dim * 2,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
            DiscriminatorBlock(
                middle_dim * 2,
                middle_dim * 4,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
        )

        self.joint_mappings = nn.Sequential(
            DiscriminatorBlock(
                middle_dim * 20,
                middle_dim * 16,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
            DiscriminatorBlock(
                middle_dim * 16,
                middle_dim * 8,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
            DiscriminatorBlock(
                middle_dim * 8,
                middle_dim * 4,
                downsample=False,
                kernel_size=1,
                padding=0,
                batch_norm=bn,
                sn_enabled=sn_enabled,
            ),
            spectral_norm(nn.Conv2d(middle_dim * 4, 1, kernel_size=1), enabled=sn_enabled),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        skip = self.pre_skip(x)
        x = self.pre_convs(x) + skip
        x = self.x_mappings(x)  # bs, middlex16, 1, 1

        z = self.z_mappings(z)  # bs, middlex4, 1, 1

        joint = torch.cat((x, z), dim=1)  # bs, middlex20, 1, 1
        joint = self.joint_mappings(joint)  # bs, 1, 1, 1
        return joint.view(-1, 1)
