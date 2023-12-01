from typing import ClassVar, NamedTuple, TypeVar

import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import AverageMeter, spectral_norm

T = TypeVar("T")


class GanLoss(NamedTuple):
    """Output container for GAN losses"""

    loss_real: Tensor
    loss_p: Tensor
    loss_tilde: Tensor


class VaeGanOutput(NamedTuple):
    """Output container for VAEGAN"""

    rec_loss: Tensor  # not for training but for record
    kl_loss: Tensor
    gan_loss: GanLoss
    log_like_loss: Tensor


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        sn_enabled: bool = False,
        kernel_size: int = 5,
        padding: int = 2,
        stride: int = 2,
        bias: bool = False,
        momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            ),
            enabled=sn_enabled,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DiscriminatorBlock(EncoderBlock):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        intermediate = self.conv(x)
        x = self.bn(intermediate)
        x = self.activation(x)
        return x, intermediate


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        stride: int = 2,
        output_padding: int = 1,
        bias: bool = False,
        momentum: float = 0.9,
        sn_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.conv = spectral_norm(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
                bias=bias,
            ),
            enabled=sn_enabled,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=momentum)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Encoder64(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        in_channels: int = 3,
        sn_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            EncoderBlock(in_channels=in_channels, out_channels=64, sn_enabled=sn_enabled),  # 64 -> 32
            EncoderBlock(in_channels=64, out_channels=128, sn_enabled=sn_enabled),  # 32 -> 16
            EncoderBlock(in_channels=128, out_channels=256, sn_enabled=sn_enabled),  # 16 -> 8
        )
        self.fc_bn_relu = nn.Sequential(
            spectral_norm(
                nn.Linear(in_features=256 * 8 * 8, out_features=1024, bias=False),
                enabled=sn_enabled,
            ),
            nn.BatchNorm1d(num_features=1024, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = spectral_norm(
            nn.Linear(in_features=1024, out_features=latent_dim),
            enabled=sn_enabled,
        )
        self.fc_var = spectral_norm(
            nn.Linear(in_features=1024, out_features=latent_dim),
            enabled=sn_enabled,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.blocks(x)  # bs, 256, 8, 8
        x = x.flatten(start_dim=1)  # bs, 256 x 8 x 8
        x = self.fc_bn_relu(x)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class Encoder32(Encoder64):
    def __init__(self, latent_dim: int = 128, in_channels: int = 3, sn_enabled: bool = False) -> None:
        super().__init__(latent_dim, in_channels, sn_enabled)
        self.blocks = nn.Sequential(
            EncoderBlock(in_channels=in_channels, out_channels=64, sn_enabled=sn_enabled),  # 32 -> 16
            EncoderBlock(in_channels=64, out_channels=256, sn_enabled=sn_enabled),  # 16 -> 8
        )


class Decoder64(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        sn_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.fc_bn_relu = nn.Sequential(
            spectral_norm(
                nn.Linear(in_features=latent_dim, out_features=256 * 8 * 8, bias=False),
                enabled=sn_enabled,
            ),
            nn.BatchNorm1d(num_features=256 * 8 * 8, momentum=0.9),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            DecoderBlock(in_channels=256, out_channels=256, sn_enabled=sn_enabled),  # 8 -> 16
            DecoderBlock(in_channels=256, out_channels=128, sn_enabled=sn_enabled),  # 16 -> 32
            DecoderBlock(in_channels=128, out_channels=32, sn_enabled=sn_enabled),  # 32 -> 16
        )
        self.final_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=32,
                    out_channels=out_channels,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
                enabled=sn_enabled,
            ),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc_bn_relu(x)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.blocks(x)
        x = self.final_conv(x)
        return x


class Decoder32(Decoder64):
    def __init__(self, latent_dim: int = 128, out_channels: int = 3, sn_enabled: bool = False) -> None:
        super().__init__(latent_dim, out_channels, sn_enabled)
        self.blocks = nn.Sequential(
            DecoderBlock(in_channels=256, out_channels=128, sn_enabled=sn_enabled),  # 8 -> 16
            DecoderBlock(in_channels=128, out_channels=32, sn_enabled=sn_enabled),  # 16 -> 32
        )


class Discriminator64(nn.Module):
    reconstruction_level: ClassVar[int] = 3

    def __init__(self, in_channels: int = 3, sn_enabled: bool = False) -> None:
        super().__init__()
        self.init_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ),
                enabled=sn_enabled,
            ),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            [
                DiscriminatorBlock(in_channels=32, out_channels=128, sn_enabled=sn_enabled),
                DiscriminatorBlock(in_channels=128, out_channels=256, sn_enabled=sn_enabled),
                DiscriminatorBlock(in_channels=256, out_channels=256, sn_enabled=sn_enabled),
            ]
        )
        self.final_fc = nn.Sequential(
            nn.Linear(in_features=256 * 8 * 8, out_features=512, bias=False),
            nn.BatchNorm1d(num_features=512, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=1),
            # nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.init_conv(x)
        for module in self.blocks:
            x, intermediate = module(x)
        x = x.flatten(start_dim=1)
        x = self.final_fc(x)
        return x, intermediate


class Discriminator32(Discriminator64):
    def __init__(self, in_channels: int = 3, sn_enabled: bool = False) -> None:
        super().__init__(in_channels, sn_enabled)
        self.blocks = nn.ModuleList(
            [
                DiscriminatorBlock(in_channels=32, out_channels=128, sn_enabled=sn_enabled),
                DiscriminatorBlock(in_channels=128, out_channels=256, sn_enabled=sn_enabled),
            ]
        )


class VaeGan(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        discriminator: nn.Module,
        device: torch.device = torch.device("cpu"),
        equilibrium: float = 0.68,
        margin: float = 0.4,
        rec_vs_gan_w: float = 1e-2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.device = device
        self.equilibrium = equilibrium
        self.margin = margin
        self.rec_vs_gan_w = rec_vs_gan_w
        self.init_parameters()
        self.encoder_optimizer = self.configure_encoder_optimizer()
        self.decoder_optimizer = self.configure_decoder_optimizer()
        self.discriminator_optimizer = self.configure_discriminator_optimizer()

    def init_parameters(self) -> None:
        for m in self.modules():
            if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                continue
            if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                scale = 1.0 / np.sqrt(np.prod(m.weight.shape[1:]))
                scale = scale / np.sqrt(3)
                nn.init.uniform_(m.weight, -scale, scale)
            if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                nn.init.constant_(m.bias, 0.0)

    def configure_encoder_optimizer(self) -> optim.Optimizer:
        """
        Note:
            the original paper used RMSProp
        """
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=3e-4)
        return self.encoder_optimizer

    def configure_decoder_optimizer(self) -> optim.Optimizer:
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=3e-4)
        return self.decoder_optimizer

    def configure_discriminator_optimizer(self) -> optim.Optimizer:
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=3e-4)
        return self.discriminator_optimizer

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x: Tensor) -> VaeGanOutput:
        x = x.to(self.device)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_tilde = self.decoder(z)
        rec = F.mse_loss(x_tilde, x, reduction="mean")

        z_p = torch.randn_like(z)
        x_p = self.decoder(z_p)

        x_concat = torch.concat([x, x_p, x_tilde], dim=0)
        d_prob, d_layer = self.discriminator(x_concat)
        d_real, d_p, d_tilde = torch.split(d_prob, [x.size(0)] * 3, dim=0)
        d_l_real, d_l_p, d_l_tilde = torch.split(d_layer, [x.size(0)] * 3, dim=0)

        kl_loss = self.compute_kl_loss(mu, log_var)
        gan_loss = self.compute_gan_loss(d_real, d_p, d_tilde)
        log_like_loss = self.compute_llike_loss(d_l_real, d_l_tilde, d_l_p)

        return VaeGanOutput(rec_loss=rec, kl_loss=kl_loss, gan_loss=gan_loss, log_like_loss=log_like_loss)

    @staticmethod
    def compute_kl_loss(mu: Tensor, log_var: Tensor) -> Tensor:
        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        return kl_loss

    @staticmethod
    def compute_gan_loss(d_real: Tensor, d_p: Tensor, d_tilde: Tensor) -> GanLoss:
        loss_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real), reduction="mean")
        loss_p = F.binary_cross_entropy_with_logits(d_p, torch.zeros_like(d_p), reduction="mean")
        loss_tilde = F.binary_cross_entropy_with_logits(d_tilde, torch.zeros_like(d_tilde), reduction="mean")
        return GanLoss(loss_real=loss_real, loss_p=loss_p, loss_tilde=loss_tilde)

    @staticmethod
    def compute_llike_loss(dis_l_x: Tensor, dis_l_x_tilde: Tensor, dis_l_x_p: Tensor) -> Tensor:
        # p(Dis_l(x) | z) = N(Dis_l(x) | Dis_l(x_tilde), I)
        # log p(Dis_l(x) | z) ∝ 0.5 * (Dis_l(x) - Dis_l(x_tilde))^2
        mse_x_tilde = 0.5 * F.mse_loss(dis_l_x, dis_l_x_tilde, reduction="mean")
        mse_x_p = 0.5 * F.mse_loss(dis_l_x, dis_l_x_p, reduction="mean")  # this is not mentioned in the paper
        mse_sum = mse_x_tilde + mse_x_p
        return mse_sum

    def train_step(self, x: Tensor) -> VaeGanOutput:
        losses = self.forward(x)
        enc_loss = losses.kl_loss + losses.log_like_loss

        train_dec = True
        train_disc = True

        if (losses.gan_loss.loss_real < self.equilibrium - self.margin) or (
            losses.gan_loss.loss_p < self.equilibrium - self.margin
        ):
            train_disc = False
        if (losses.gan_loss.loss_real > self.equilibrium + self.margin) or (
            losses.gan_loss.loss_p > self.equilibrium + self.margin
        ):
            train_dec = False
        if (train_dec or train_disc) is False:
            train_disc = True
            train_dec = True

        # 1. update encoder regardless of the losses
        self.encoder_optimizer.zero_grad()
        enc_loss.backward()
        self.encoder_optimizer.step()

        # 2. update decoder
        if train_dec:
            self.decoder_optimizer.zero_grad()
            losses = self.forward(x)
            dec_gan_loss = -(losses.gan_loss.loss_p + losses.gan_loss.loss_tilde)
            dec_loss = self.rec_vs_gan_w * losses.log_like_loss + (1 - self.rec_vs_gan_w) * dec_gan_loss
            dec_loss.backward()
            self.decoder_optimizer.step()

        # 3. update discriminator
        if train_disc:
            self.discriminator_optimizer.zero_grad()
            losses = self.forward(x)
            disc_loss = losses.gan_loss.loss_real + losses.gan_loss.loss_p + losses.gan_loss.loss_tilde
            disc_loss.backward()
            self.discriminator_optimizer.step()

        return losses

    @torch.no_grad()
    def eval_step(self, x: Tensor) -> VaeGanOutput:
        return self.forward(x)

    def train_single_epoch(self, data_loader: DataLoader) -> VaeGanOutput:
        self.train()
        return self._iterate_epoch(data_loader, "train")

    def evaluate(self, data_loader: DataLoader) -> VaeGanOutput:
        self.eval()
        return self._iterate_epoch(data_loader, "eval")

    def _iterate_epoch(self, data_loader: DataLoader, mode: str) -> VaeGanOutput:
        rec_loss = AverageMeter()
        kl_loss = AverageMeter()
        log_like_loss = AverageMeter()
        g_loss_real = AverageMeter()
        g_loss_p = AverageMeter()
        g_loss_tilde = AverageMeter()
        pbar = tqdm(total=len(data_loader))
        for x in data_loader:
            if mode == "train":
                output = self.train_step(x)
            else:
                output = self.eval_step(x)
            rec_loss.update(output.rec_loss.item())
            kl_loss.update(output.kl_loss.item())
            log_like_loss.update(output.log_like_loss.item())
            g_loss_real.update(output.gan_loss.loss_real.item())
            g_loss_p.update(output.gan_loss.loss_p.item())
            g_loss_tilde.update(output.gan_loss.loss_tilde.item())
            pbar.set_description(
                f"rec loss: {rec_loss.average:.3f}\n"
                f"kl loss: {kl_loss.average:.3f}\n"
                f"log like loss: {log_like_loss.average:.3f}\n"
                f"gan loss real: {g_loss_real.average:.3f}\n"
                f"gan loss p: {g_loss_p.average:.3f}\n"
                f"gan loss tilde: {g_loss_tilde.average:.3f}\n"
            )
            pbar.update()
        pbar.close()
        out = VaeGanOutput(
            rec_loss=torch.Tensor([rec_loss.average]),
            kl_loss=torch.Tensor([kl_loss.average]),
            log_like_loss=torch.Tensor([log_like_loss.average]),
            gan_loss=GanLoss(
                loss_real=torch.Tensor([g_loss_real.average]),
                loss_p=torch.Tensor([g_loss_p.average]),
                loss_tilde=torch.Tensor([g_loss_tilde.average]),
            ),
        )
        return out
