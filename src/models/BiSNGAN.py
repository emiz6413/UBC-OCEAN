from itertools import chain
from typing import ClassVar

import torch
from torch import nn, optim
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader


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


class SNDiscriminatorBlock(nn.Module):
    leak: ClassVar[int] = 0.2

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 4, stride: int = 1, padding: int = 0
    ) -> None:
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        )
        self.activation = nn.ReLU(inplace=True)

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
        self.latent_dim = latent_dim

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
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ups(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int = 128, dim: int = 128) -> None:
        self.latent_dim = latent_dim
        self.x_mapping = nn.Sequential(
            *[
                SNDiscriminatorBlock(dim if i else input_channels, dim, kernel_size=4, stride=2, padding=1)
                for i in range(3)
            ],
            SNDiscriminatorBlock(dim, dim, kernel_size=4),
        )

        self.z_mapping = nn.Sequential(
            SNDiscriminatorBlock(latent_dim, dim, kernel_size=1), SNDiscriminatorBlock(dim, dim, kernel_size=1)
        )

        self.join_mapping = nn.Sequential(
            SNDiscriminatorBlock(dim * 2, dim * 2, kernel_size=1),
            SNDiscriminatorBlock(dim * 2, dim * 2, kernel_size=1),
            nn.Conv2d(dim * 2, 1, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = self.x_mapping(x)
        z = self.z_mapping(z)
        joint = torch.cat((x, z), dim=1)
        joint = self.join_mapping(joint)
        return self.sigmoid(joint)


class BiSNGAN(nn.Module):
    def __init__(self, encoder: Encoder, generator: Generator, discriminator: Discriminator) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = self.encoder.latent_dim
        self.criterion = nn.BCELoss()
        self.eg_optimizer = self.create_eg_optimizer()
        self.d_optimizer = self.create_d_optimizer()

    def create_eg_optimizer(self, lr: float = 1e-4, betas: tuple[float, float] = (0.5, 0.999)) -> optim.Optimizer:
        self.eg_optimizer = optim.Adam(
            chain(self.encoder.parameters(), self.generator.parameters()), lr=lr, betas=betas
        )
        return self.eg_optimizer

    def create_d_optimizer(self, lr: float = 1e-4, betas: tuple[float, float] = (0.5, 0.999)) -> optim.Optimizer:
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return self.d_optimizer

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def generate(self, z: torch.Tensor) -> torch.Tensor:
        return self.generator(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.generate(self.encode(x))

    def discriminate(
        self, x: torch.Tensor, z_hat: torch.Tensor, x_tilde: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([x, x_tilde], dim=0)
        z = torch.cat([z_hat, z], dim=0)
        output = self.discriminator(x, z)
        data_preds, sample_preds = torch.split(output, 2, dim=0)
        return data_preds, sample_preds

    def forward(self, x: torch.Tensor, z: torch.Tensor, lamb: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
        z_hat = self.encode(x)
        x_tilde = self.generate(z)
        data_preds, sample_preds = self.discriminate(x, z_hat, x_tilde, z)
        eg_loss = torch.mean(data_preds - sample_preds)
        d_loss = -eg_loss + lamb * self.calc_grad_penalty(x.detach(), z_hat.detach(), x_tilde.detach(), z.detach())
        return d_loss, eg_loss

    def train(self, train_loader: DataLoader) -> "BiSNGAN":
        # ge_loss = 0
        # d_loss = 0
        for idx, (x, _) in enumerate(train_loader, 1):
            _ge_loss, d_loss = self.train_step(x)

        return self

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_true = torch.ones((x.size(0), 1), device=x.device)
        y_fake = torch.zeros_like(y_true)

        self.d_optimizer.zero_grad()
        self.eg_optimizer.zero_grad()

        # generator
        z_fake = torch.randn(x.size(0), self.latent_dim, 1, 1, device=x.device)
        x_fake = self.generate(z_fake)

        # encoder
        z_true = self.encode(x)

        # discriminator
        real_preds, fake_preds = self.discriminate(x, z_true, x_fake, z_fake)

        d_loss: torch.Tensor = self.criterion(real_preds, y_true) + self.criterion(fake_preds, y_fake)
        ge_loss: torch.Tensor = self.criterion(fake_preds, y_true) + self.criterion(real_preds, y_fake)

        # back-prop
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        ge_loss.backward(retain_graph=True)
        self.eg_optimizer.step()

        return d_loss.item(), ge_loss.item()
