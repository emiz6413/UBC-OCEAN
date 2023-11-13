from itertools import chain
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import AverageMeter, HingeLoss

LOSS_TYPE = Literal["BCE", "Hinge"]


class BiGAN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        generator: nn.Module,
        discriminator: nn.Module,
        amp: bool = False,
        eval_amp: bool = False,
        loss_type: LOSS_TYPE = "Hinge",
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
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)
        self.eval_amp = eval_amp
        self.disc_iters = disc_iters
        self.ge_iters = ge_iters

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
        data_preds, sample_preds = torch.tensor_split(output, 2, dim=0)
        return data_preds, sample_preds

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        rec_loss_meter = AverageMeter()
        pbar = tqdm(total=len(eval_loader), leave=False)
        self.eval()
        for x in eval_loader:
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

            real_preds, fake_preds = self.discriminate(x_real, z_fake, x_fake, z_real)
            d_loss: torch.Tensor = self.d_criterion(real_preds, y_real) + self.d_criterion(fake_preds, y_fake)

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

            real_preds, fake_preds = self.discriminate(x_real, z_fake, x_fake, z_real)
            ge_loss: torch.Tensor = self.ge_criterion(fake_preds, y_real) + self.ge_criterion(real_preds, y_fake)

        self.scaler.scale(ge_loss).backward()
        self.scaler.step(self.ge_optimizer)
        self.scaler.update()
        return ge_loss
