import torch
from torch import nn


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
