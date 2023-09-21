from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader, random_split


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        nclass: int
    ):
        super().__init__()
        self._num_samples = num_samples
        self._seq_length = seq_length
        self._nclass = nclass
        self._data = []
        self._labels = []

        for _ in range(num_samples):
            class_label = np.random.randint(nclass)
            freq = class_label + 1
            time_series = np.sin(np.linspace(0, freq * np.pi, seq_length))
            self._data.append(time_series)
            self._labels.append(class_label)

        self._data = torch.tensor(self._data, dtype=torch.float32)
        self._labels = torch.tensor(self._labels, dtype=torch.long)

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._data[index], self._labels[index]


class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_samples: int = 10000,
        seq_length: int = 10,
        nclass: int = 2
    ):
        super().__init__()
        self._batch_size = batch_size
        self._num_samples = num_samples
        self._seq_length = seq_length
        self._dataset = TimeSeriesDataset(num_samples, seq_length, nclass)

    def setup(self, stage=None):
        train_length = int(len(self._dataset) * 0.8)
        val_length = int(len(self._dataset) * 0.1)
        test_length = len(self._dataset) - train_length - val_length

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            self._dataset, [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size)

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = x[..., None]
        batch = x, y
        return batch
