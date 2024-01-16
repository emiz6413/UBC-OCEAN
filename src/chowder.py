import torch
from torch import Tensor, nn


class ChowderNetwork(nn.Module):
    def __init__(self, in_channels: int = 128, r: int = 5, num_classes: int = 5) -> None:
        super().__init__()
        self.conv1d = nn.Linear(in_features=in_channels, out_features=1)
        self.fc = nn.Linear(in_features=2 * r, out_features=num_classes)
        self.drop = nn.Dropout(p=0.5)
        self.r = r

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1d(x)  # *, N, in_channels -> *, N, 1
        x = x.squeeze(-1)  # *, N, 1 -> *, N
        x, _ = torch.sort(x, dim=-1, descending=True)  # sort values along dim=1
        x = torch.concat([x[..., : self.r], x[..., -self.r :]], dim=-1)  # *, 2r
        x = self.drop(self.fc(x))  # *, num_class
        return x
