import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import AverageMeter


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, amp: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = self.configure_optimizer()
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=amp)

    def to(self, device: torch.device) -> "AutoEncoder":  # type: ignore[override]
        self.device = device
        return super().to(device=device)

    def configure_optimizer(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=1e-4)

    def criterion(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x_hat, x.to(x_hat.device), reduction="mean")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x.to(self.device))
        x_hat = self.decode(z)
        return x_hat

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            x_hat = self(x)
            loss = self.criterion(x, x_hat)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss

    @torch.no_grad()
    def eval_step(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self(x)
        loss = self.criterion(x, x_hat)
        return loss

    def train_single_epoch(self, data_loader: DataLoader) -> float:
        self.train()
        rec_loss = AverageMeter()
        pbar = tqdm(total=len(data_loader), leave=False)
        for x in data_loader:
            loss = self.train_step(x)
            rec_loss.update(loss.item())
            pbar.set_description(f"Rec loss: {rec_loss.average:.3f}")
            pbar.update()
        pbar.close()

        return rec_loss.average

    def evaluate(self, data_loader: DataLoader) -> float:
        self.eval()
        rec_loss = AverageMeter()
        pbar = tqdm(total=len(data_loader), leave=False)
        for x in data_loader:
            loss = self.eval_step(x)
            rec_loss.update(loss.item())
            pbar.set_description(f"Rec loss: {rec_loss.average:.3f}")
            pbar.update()
        pbar.close()

        return rec_loss.average
