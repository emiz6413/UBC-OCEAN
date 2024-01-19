from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .utils import AverageMeter


class ChowderNetwork(nn.Module):
    def __init__(self, in_channels: int = 128, kernel_size: int = 1, r: int = 5, num_classes: int = 5) -> None:
        super().__init__()
        self.conv1d = nn.Linear(in_features=in_channels, out_features=1)
        self.fc = nn.Linear(in_features=2 * r, out_features=num_classes)
        self.drop = nn.Dropout(p=0.5)
        self.r = r
        self.optimizer = self.configure_optimizer()
        self.criterion = self.configure_criterion()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1d(x)  # *, N, in_channels -> *, N, 1
        x = x.squeeze(-1)  # *, N, 1 -> *, N
        x, _ = torch.sort(x, dim=-1, descending=True)  # sort values along dim=1
        x = torch.concat([x[..., : self.r], x[..., -self.r :]], dim=-1)  # *, 2r
        x = self.drop(self.fc(x))  # *, num_class
        return x

    def configure_optimizer(self) -> optim.Optimizer:
        self.optimizer = optim.Adam(
            [{"params": self.conv1d.parameters(), "weight_decay": 0.5}, {"params": self.fc.parameters()}],
            lr=1e-3,
        )
        return self.optimizer

    def configure_criterion(self) -> nn.Module:
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion

    def train_step(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()
        pred = self.forward(x)
        loss = self.criterion(input=pred, target=target.to(pred.device))
        loss.backward()
        self.optimizer.step()
        return loss, pred

    def train_single_epoch(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        preds = []
        labels = []
        loss = AverageMeter()
        pbar = tqdm(data_loader, leave=False)
        for x, label in pbar:
            _loss, pred = self.train_step(x, label)
            loss.update(_loss.item())
            preds.append(pred.detach().cpu().numpy())
            labels.append(label.cpu().numpy())
            pbar.set_description(f"train loss: {loss.average:.3f}")

        return loss.average, np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)

    @torch.no_grad()
    def eval_step(self, x: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        pred = self.forward(x)
        loss = self.criterion(input=pred, target=target.to(pred.device))
        return loss, pred

    def evaluate(self, data_loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
        preds = []
        labels = []
        loss = AverageMeter()
        pbar = tqdm(data_loader, leave=False)
        for x, label in pbar:
            _loss, pred = self.eval_step(x, label)
            loss.update(_loss.item())
            preds.append(pred.cpu().numpy())
            labels.append(label.numpy())
            pbar.set_description(f"eval loss: {loss.average:.3f}")

        return loss.average, np.concatenate(preds, axis=0), np.concatenate(labels, axis=0)


class RandomSamplingDataset(Dataset):
    label_to_id = {"HGSC": 0, "LGSC": 1, "EC": 2, "CC": 3, "MC": 4}

    def __init__(self, data_dir: Path, df: pd.DataFrame, n: int = 1000) -> None:
        self.data_dir = data_dir
        self.df = df
        self.n = n

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]
        target = row["image_id"]
        label = self.label_to_id[row["label"]]
        with open(str(self.data_dir.joinpath(f"{target}.npy")), "rb") as f:
            arr = torch.from_numpy(np.load(f))
        arr = arr.flatten(start_dim=1, end_dim=2).transpose(1, 0)
        replace = arr.size(0) < self.n
        indices = np.random.choice(np.arange(arr.size(0)), size=self.n, replace=replace)
        return arr[indices], torch.tensor(label)
