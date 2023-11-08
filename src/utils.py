import numpy as np
import torch
from scipy.stats import truncnorm  # type: ignore
from torch import nn


def truncated_z_sample(batch_size: int, z_dim: int, truncation: float = 0.5, seed: int | None = None):
    state = np.random.RandomState(seed) if seed is not None else None
    values = truncnorm.rvs(-2, 2, size=(batch_size, z_dim), random_state=state)
    return truncation * values


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


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
