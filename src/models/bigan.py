from typing import ClassVar

import torch
from torch import nn


class EncoderBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GeneratorBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DiscriminatorBlock(nn.Module):
    leak: ClassVar[int] = 0.2

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True
        )
        self.activation = nn.LeakyReLU(negative_slope=self.leak, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class Encoder(nn.Module):
    n_blocks: ClassVar[int] = 3  # 32 = 4 * 2 ** n_blocks

    def __init__(self, input_channels: int, latent_dim: int = 128, dim: int = 128) -> None:
        super().__init__()
        self.downs = nn.Sequential(
            *[
                EncoderBlock(dim if i else input_channels, dim, kernel_size=4, stride=2, padding=1)
                for i in range(self.n_blocks)
            ],
            EncoderBlock(dim, dim, kernel_size=4, stride=1, padding=0),
            nn.Conv2d(dim, latent_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downs(x)


class Generator(nn.Module):
    n_blocks: ClassVar[int] = 3  # 32 = 4 * 2 ** n_blocks

    def __init__(self, output_channels: int, latent_dim: int = 128, dim: int = 128) -> None:
        super().__init__()
        self.ups = nn.Sequential(
            GeneratorBlock(latent_dim, dim, kernel_size=4, stride=1, padding=0),
            *[GeneratorBlock(dim, dim, kernel_size=4, stride=2, padding=1) for _ in range(self.n_blocks - 1)],
            nn.ConvTranspose2d(dim, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ups(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int = 128, dim: int = 128) -> None:
        self.x_mapping = nn.Sequential(
            *[
                DiscriminatorBlock(dim if i else input_channels, dim, kernel_size=4, stride=2, padding=1)
                for i in range(3)
            ],
            DiscriminatorBlock(dim, dim, kernel_size=4),
        )

        self.z_mapping = nn.Sequential(
            DiscriminatorBlock(latent_dim, dim, kernel_size=1), DiscriminatorBlock(dim, dim, kernel_size=1)
        )

        self.join_mapping = nn.Sequential(
            DiscriminatorBlock(dim * 2, dim * 2, kernel_size=1),
            DiscriminatorBlock(dim * 2, dim * 2, kernel_size=1),
            nn.Conv2d(dim * 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.x_mapping(x)
        z = self.z_mapping(z)
        joint = torch.cat((x, z), dim=1)
        joint = self.join_mapping(joint)
        return joint


class BiGAN(nn.Module):
    def __init__(self, encoder: Encoder, generator: Generator, discriminator: Discriminator) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.generate(self.encode(x))
