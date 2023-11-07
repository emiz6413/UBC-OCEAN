from itertools import chain
from typing import Literal

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

LOSS_TYPE = Literal["BCE", "Hinge"]


class HingeLoss(nn.Module):
    def __init__(self, for_discriminator: bool) -> None:
        super().__init__()
        self.for_discriminator = for_discriminator

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.for_discriminator:
            return self.compute_loss_for_discriminator(input, target)
        else:
            return self.compute_loss_for_generator(input, target)

    @staticmethod
    def compute_loss_for_discriminator(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if all(target == 1):
            min_val = torch.min(input - 1, torch.zeros_like(input))
            return -torch.mean(min_val)
        elif all(target == 0):
            min_val = torch.min(-input - 1, torch.zeros_like(input))
            return -torch.mean(min_val)
        else:
            raise ValueError("invalid target value")

    @staticmethod
    def compute_loss_for_generator(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if all(target == 1):
            return -input.mean()
        elif all(target == 0):
            return input.mean()
        else:
            raise ValueError("invalid target value")


class BiGAN(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        generator: nn.Module,
        discriminator: nn.Module,
        amp: bool = False,
        loss_type: LOSS_TYPE = "Hinge",
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

    def create_eg_optimizer(self, lr: float = 1e-4, betas: tuple[float, float] = (0.5, 0.999)) -> optim.Optimizer:
        self.ge_optimizer = optim.Adam(
            chain(self.encoder.parameters(), self.generator.parameters()), lr=lr, betas=betas
        )
        return self.ge_optimizer

    def create_d_optimizer(self, lr: float = 3e-4, betas: tuple[float, float] = (0.5, 0.999)) -> optim.Optimizer:
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
    def evaluate(self, eval_loader: DataLoader) -> torch.Tensor:
        rec_loss = []
        pbar = tqdm(total=len(eval_loader), leave=False)
        self.eval()
        for x, _ in eval_loader:
            reconstructed = self.reconstruct(x)
            mse = nn.functional.mse_loss(input=reconstructed, target=x)
            pbar.set_description(f"reconstruction loss: {mse.item():.3f}")
            rec_loss.append(mse)
            pbar.update()
        pbar.close()
        return torch.mean(torch.tensor(rec_loss))

    def train_single_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        ge_loss = 0.0
        d_loss = 0.0
        pbar = tqdm(total=len(train_loader), leave=False)
        self.train()
        for x, _ in train_loader:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None, dtype=torch.float16):
                _ge_loss, _d_loss = self.train_step(x)
            pbar.set_description(
                f"Generator/Encoder loss: {_ge_loss.item():.3f}. Discriminator loss: {_d_loss.item():.3f}"
            )
            ge_loss += _ge_loss.item()
            d_loss += _d_loss.item()
            pbar.update()
        pbar.close()

        return ge_loss / len(train_loader), d_loss / len(train_loader)

    def train_step(self, x_real: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_real = torch.ones((x_real.size(0), 1), device=x_real.device)
        y_fake = torch.zeros_like(y_real)

        z_fake = torch.randn(x_real.size(0), self.latent_dim, 1, 1, device=x_real.device)
        z_real = self.encode(x_real)  # keep the graph untouched while training discriminator

        x_fake = self.generate(z_fake)  # keep the graph untouched while training discriminator

        # 1. train the discriminator
        self.d_optimizer.zero_grad()

        real_preds, fake_preds = self.discriminate(x_real, z_fake, x_fake.detach(), z_real.detach())
        d_loss: torch.Tensor = self.d_criterion(real_preds, y_real) + self.d_criterion(fake_preds, y_fake)

        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optimizer)

        # 2. train the encoder and generator
        self.ge_optimizer.zero_grad()

        real_preds, fake_preds = self.discriminate(x_real, z_fake, x_fake, z_real)
        ge_loss: torch.Tensor = self.ge_criterion(fake_preds, y_real) + self.ge_criterion(real_preds, y_fake)

        self.scaler.scale(ge_loss).backward()
        self.scaler.step(self.ge_optimizer)

        self.scaler.update()

        return d_loss, ge_loss
