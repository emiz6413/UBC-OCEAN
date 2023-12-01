from itertools import chain
from typing import Literal

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import AverageMeter, HingeLoss, spectral_norm


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Encoder64(nn.Module):
    def __init__(self, latent_dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim  # for compatibility
        self.layers = nn.Sequential(
            EncoderBlock(in_channels=in_channels, out_channels=64),  # 64 -> 32
            EncoderBlock(in_channels=64, out_channels=128),  # 32 -> 16
            EncoderBlock(in_channels=128, out_channels=256),  # 16 -> 8
            EncoderBlock(in_channels=256, out_channels=512),  # 8 -> 4
            EncoderBlock(in_channels=512, out_channels=1024, stride=1, padding=0),  # 4 -> 1
            nn.Conv2d(in_channels=1024, out_channels=latent_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Encoder128(nn.Module):
    def __init__(self, latent_dim: int = 128, in_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            EncoderBlock(in_channels=in_channels, out_channels=64),  # 128 -> 64
            EncoderBlock(in_channels=64, out_channels=128),  # 64 -> 32
            EncoderBlock(in_channels=128, out_channels=256),  # 32 -> 16
            EncoderBlock(in_channels=256, out_channels=512),  # 16 -> 8
            EncoderBlock(in_channels=512, out_channels=1024),  # 8 -> 4
            EncoderBlock(in_channels=1024, out_channels=2048, stride=1, padding=0),  # 4 -> 1
            nn.Conv2d(in_channels=2048, out_channels=latent_dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        in_channles: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(
            in_channels=in_channles,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_t(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Generator64(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            GeneratorBlock(in_channles=latent_dim, out_channels=1024, stride=1, padding=0),  # 1 -> 4
            GeneratorBlock(in_channles=1024, out_channels=512),  # 4 -> 8
            GeneratorBlock(in_channles=512, out_channels=256),  # 8 -> 16
            GeneratorBlock(in_channles=256, out_channels=128),  # 16 -> 32
            nn.ConvTranspose2d(
                in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 32 -> 64
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class Generator128(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            GeneratorBlock(in_channles=latent_dim, out_channels=1024, stride=1, padding=0),  # 1 -> 4
            GeneratorBlock(in_channles=1024, out_channels=512),  # 4 -> 8
            GeneratorBlock(in_channles=512, out_channels=256),  # 8 -> 16
            GeneratorBlock(in_channles=256, out_channels=128),  # 16 -> 32
            GeneratorBlock(in_channles=128, out_channels=64),  # 32 -> 64
            nn.ConvTranspose2d(
                in_channels=64, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 64 -> 128
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channles: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        bias: bool = True,
        sn_enabled: bool = False,
        bn_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels=in_channles,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            enabled=sn_enabled,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels) if bn_enabled else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Discriminator64(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        sn_enabled: bool = False,
        bn_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.x_mapping = nn.Sequential(
            DiscriminatorBlock(
                in_channles=in_channels, out_channels=64, sn_enabled=sn_enabled, bn_enabled=bn_enabled
            ),  # 64 -> 32
            DiscriminatorBlock(
                in_channles=64, out_channels=128, sn_enabled=sn_enabled, bn_enabled=bn_enabled
            ),  # 32 -> 16
            DiscriminatorBlock(
                in_channles=128, out_channels=256, sn_enabled=sn_enabled, bn_enabled=bn_enabled
            ),  # 16 -> 8
            DiscriminatorBlock(
                in_channles=256, out_channels=512, sn_enabled=sn_enabled, bn_enabled=bn_enabled
            ),  # 8 -> 4
            DiscriminatorBlock(
                in_channles=512, out_channels=1024, stride=1, padding=0, sn_enabled=sn_enabled, bn_enabled=bn_enabled
            ),  # 4 -> 1
        )

        self.z_mapping = nn.Sequential(
            DiscriminatorBlock(
                in_channles=latent_dim,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            DiscriminatorBlock(
                in_channles=512,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
        )

        self.joint_mapping = nn.Sequential(
            DiscriminatorBlock(
                in_channles=1024 + 1024,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            DiscriminatorBlock(
                in_channles=2048,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1),
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.x_mapping(x)
        z = self.z_mapping(z)
        joint = torch.concat((x, z), dim=1)
        joint = self.joint_mapping(joint)
        return joint


class Discriminator128(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 128,
        sn_enabled: bool = False,
        bn_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.x_mapping = nn.Sequential(
            DiscriminatorBlock(in_channels, 64, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 128 -> 64
            DiscriminatorBlock(64, 128, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 64 -> 32
            DiscriminatorBlock(128, 256, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 32 -> 16
            DiscriminatorBlock(256, 512, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 16 -> 8
            DiscriminatorBlock(512, 1024, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 8 -> 4
            DiscriminatorBlock(1024, 2048, stride=1, padding=0, sn_enabled=sn_enabled, bn_enabled=bn_enabled),  # 4 -> 1
        )

        self.z_mapping = nn.Sequential(
            DiscriminatorBlock(
                in_channles=latent_dim,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            DiscriminatorBlock(
                in_channles=256,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            DiscriminatorBlock(
                in_channles=512,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
        )

        self.joint_mapping = nn.Sequential(
            DiscriminatorBlock(
                in_channles=2048 + 1024,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            DiscriminatorBlock(
                in_channles=2048,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0,
                sn_enabled=sn_enabled,
                bn_enabled=bn_enabled,
            ),
            nn.Conv2d(in_channels=2048, out_channels=1, kernel_size=1),
        )

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.x_mapping(x)
        z = self.z_mapping(z)
        joint = torch.concat((x, z), dim=1)
        joint = self.joint_mapping(joint)
        return joint


class BiGAN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        generator: nn.Module,
        discriminator: nn.Module,
        device: torch.device = torch.device("cpu"),
        amp: bool = False,
        eval_amp: bool = False,
        loss_type: Literal["BCE", "Hinge"] = "Hinge",
        disc_iters: int = 2,
        ge_iters: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = self.encoder.latent_dim
        self.loss_type = loss_type
        self.ge_criterion = self.create_eg_criterion()
        self.d_criterion = self.create_d_criterion()
        self.criterion = nn.BCEWithLogitsLoss()
        self.ge_optimizer = self.create_eg_optimizer()
        self.d_optimizer = self.create_d_optimizer()
        self.device = device
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)
        self.eval_amp = eval_amp
        self.disc_iters = disc_iters
        self.ge_iters = ge_iters
        self.init_parameters()

    def init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def create_eg_optimizer(self, lr: float = 5e-5, betas: tuple[float, float] = (0.0, 0.999)) -> optim.Optimizer:
        self.ge_optimizer = optim.Adam(
            chain(self.encoder.parameters(), self.generator.parameters()), lr=lr, betas=betas
        )
        return self.ge_optimizer

    def create_d_optimizer(self, lr: float = 2e-4, betas: tuple[float, float] = (0.0, 0.999)) -> optim.Optimizer:
        """
        Note: Setting Discriminator's learning rate larger converges faster

        .. Heusel et al. (2017) https://arxiv.org/abs/1706.08500
        """
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        return self.d_optimizer

    def create_eg_criterion(self) -> nn.Module:
        if self.loss_type == "BCE":
            return nn.BCEWithLogitsLoss()
        if self.loss_type == "Hinge":
            return HingeLoss(for_discriminator=False)

    def create_d_criterion(self) -> nn.Module:
        if self.loss_type == "BCE":
            return nn.BCEWithLogitsLoss()
        if self.loss_type == "Hinge":
            return HingeLoss(for_discriminator=True)

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
        real_preds, tilde_preds = torch.tensor_split(output, 2, dim=0)
        return real_preds, tilde_preds

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        rec_loss_meter = AverageMeter()
        pbar = tqdm(total=len(eval_loader), leave=False)
        self.eval()
        for x in eval_loader:
            x = x.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.eval_amp):
                reconstructed = self.reconstruct(x)
            mse = nn.functional.mse_loss(input=reconstructed, target=x)
            rec_loss_meter.update(mse.item())
            pbar.set_description(f"reconstruction loss: {rec_loss_meter.average:.3f}")
            pbar.update()
        pbar.close()
        return rec_loss_meter.average

    def train_single_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        ge_loss_meter = AverageMeter()
        d_loss_meter = AverageMeter()
        pbar = tqdm(total=len(train_loader), leave=False)
        self.train()
        d_iter = 0
        ge_iter = 0
        for x in train_loader:
            x = x.to(self.device)
            if d_iter < self.disc_iters:
                d_loss = self.train_disc(x)
                d_loss_meter.update(d_loss.item())
                d_iter += 1
            else:
                ge_loss = self.train_ge(x)
                ge_loss_meter.update(ge_loss.item())
                ge_iter += 1

                if ge_iter == self.ge_iters:
                    d_iter = 0
                    ge_iter = 0

            pbar.set_description(f"GE loss: {ge_loss_meter.average:.3f}. D loss: {d_loss_meter.average:.3f}")
            pbar.update()
        pbar.close()

        return ge_loss_meter.average, d_loss_meter.average

    def train_disc(self, x_real: torch.Tensor) -> torch.Tensor:
        """Train the discriminator"""
        self.d_optimizer.zero_grad()

        y_real = torch.ones((x_real.size(0), 1), device=x_real.device)
        y_fake = torch.zeros_like(y_real)
        z_fake = torch.randn(x_real.size(0), self.latent_dim, 1, 1, device=x_real.device)
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            with torch.no_grad():
                z_real = self.encode(x_real)
                x_fake = self.generate(z_fake)

            real_preds, fake_preds = self.discriminate(x_real, z_real, x_fake, z_fake)
            d_loss: torch.Tensor = self.d_criterion(real_preds.view(-1, 1), y_real) + self.d_criterion(
                fake_preds.view(-1, 1), y_fake
            )

        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)
        self.scaler.update()
        return d_loss

    def train_ge(self, x_real: torch.Tensor) -> torch.Tensor:
        """Train the generator and encoder"""
        self.ge_optimizer.zero_grad()

        y_real = torch.ones((x_real.size(0), 1), device=x_real.device)
        y_fake = torch.zeros_like(y_real)
        z_fake = torch.randn(x_real.size(0), self.latent_dim, 1, 1, device=x_real.device)

        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            z_real = self.encode(x_real)
            x_fake = self.generate(z_fake)

            real_preds, fake_preds = self.discriminate(x_real, z_real, x_fake, z_fake)
            ge_loss: torch.Tensor = self.ge_criterion(fake_preds.view(-1, 1), y_real) + self.ge_criterion(
                real_preds.view(-1, 1), y_fake
            )

        self.scaler.scale(ge_loss).backward()
        self.scaler.step(self.ge_optimizer)
        self.scaler.update()
        return ge_loss
