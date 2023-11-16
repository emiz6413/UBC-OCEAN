import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import AverageMeter


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Encoder128(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: tuple[int, int, int, int, int] = (32, 64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential()
        in_channels = 3
        for out_channel in out_channels:
            self.blocks.append(EncoderBlock(in_channels=in_channels, out_channels=out_channel))
            in_channels = out_channel

        self.linear1 = nn.Linear(in_features=out_channel * 4 * 4, out_features=512)
        self.bn = nn.BatchNorm1d(num_features=512)
        self.activation = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(in_features=512, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)  # bs, 512, 4, 4
        x = x.flatten(start_dim=1)
        x = self.linear1(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Decoder128(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: tuple[int, int, int, int, int] = (256, 128, 64, 32, 16),
    ) -> None:
        super().__init__()
        self.pre_linear = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=512 * 4 * 4),
            nn.BatchNorm1d(num_features=512 * 4 * 4),
            nn.LeakyReLU(inplace=True),
        )

        self.blocks = nn.Sequential()
        in_channels = 512
        for out_channel in out_channels:
            self.blocks.append(DecoderBlock(in_channels=in_channels, out_channels=out_channel))
            in_channels = out_channel

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channel, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre_linear(x)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.blocks(x)
        x = self.final_conv(x)
        return x


class VanillaVAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 128,
        kl_w: float = 5e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.kl_w = kl_w
        self.optimizer = self.configure_optimizer()
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

    def configure_optimizer(self) -> torch.optim.Optimizer:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return self.optimizer

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.rand_like(std)

        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def compute_losses(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(x_hat.device)
        rec_loss = nn.functional.mse_loss(input=x_hat, target=x, reduction="mean")

        kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)

        return rec_loss, kl_loss

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.optimizer.zero_grad()
        x_hat, mu, log_var = self.forward(x)
        rec_loss, kl_loss = self.compute_losses(x, x_hat, mu, log_var)
        loss = rec_loss + self.kl_w * kl_loss
        loss.backward()
        self.optimizer.step()
        return loss, rec_loss, kl_loss

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, mu, log_var = self.forward(x)
        rec_loss, kl_loss = self.compute_losses(x, x_hat, mu, log_var)
        loss = rec_loss + self.kl_w * kl_loss
        return loss

    def train_single_epoch(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.train()
        loss = AverageMeter()
        rec_loss = AverageMeter()
        kl_loss = AverageMeter()
        pbar = tqdm(total=len(data_loader))
        for x in data_loader:
            _loss, _rec_loss, _kl_loss = self.train_step(x)
            loss.update(_loss.item())
            rec_loss.update(_rec_loss.item())
            kl_loss.update(_kl_loss.item())
            pbar.set_description(
                f"loss: {loss.average:.3f} rec loss: {rec_loss.average:.3f} kl loss: {kl_loss.average:.3f}"
            )
            pbar.update()
        pbar.close()
        return loss.average, rec_loss.average, kl_loss.average

    def evaluate(self, data_loader: DataLoader) -> tuple[float, float, float]:
        self.eval()
        loss = AverageMeter()
        rec_loss = AverageMeter()
        kl_loss = AverageMeter()
        pbar = tqdm(total=len(data_loader))
        for x in data_loader:
            _loss, _rec_loss, _kl_loss = self.eval_step(x)
            loss.update(_loss.item())
            rec_loss.update(_rec_loss.item())
            kl_loss.update(_kl_loss.item())
            pbar.set_description(
                f"loss: {loss.average:.3f} rec loss: {rec_loss.average:.3f} kl loss: {kl_loss.average:.3f}"
            )
            pbar.update()
        pbar.close()
        return loss.average, rec_loss.average, kl_loss.average
