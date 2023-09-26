from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class NonUniformVCODataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        seq_length: int,
        sensitivity: float,
        dt: float
    ):
        super().__init__()
        self._num_samples = num_samples
        self._seq_length = seq_length
        self._sensitivity = sensitivity
        self._dt = dt

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(index)

        phase_shift = torch.rand(1, generator=generator) * 2 * np.pi
        amplitude = torch.rand(1, generator=generator)
        f = torch.rand(1, generator=generator) * 2 * np.pi
        offset = 1.5 + torch.rand(1, generator=generator)

        time = torch.cumsum(torch.rand(self._seq_length, generator=generator) * self._dt, dim=0)
        voltage = offset + amplitude * torch.cos(f * time + phase_shift)

        frequency = self._sensitivity * voltage
        phase = 2 * np.pi * torch.cumsum(frequency, dim=0) * self._dt
        output = torch.sin(phase)

        voltage = voltage[:, None]
        output = output[:, None]

        return voltage, output, time


class NonUniformVCODataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_samples: int = 10000,
        seq_length: int = 1000,
        sensitivity: float = 1.0,
        dt: float = 0.01
    ):
        super().__init__()
        self._dataset = NonUniformVCODataset(
            num_samples=num_samples,
            seq_length=seq_length,
            sensitivity=sensitivity,
            dt=dt
        )
        self._batch_size = batch_size

    def setup(self, stage=None):
        train_length = int(len(self._dataset) * 0.8)
        val_length = int(len(self._dataset) * 0.1)
        test_length = len(self._dataset) - train_length - val_length

        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            self._dataset, [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self._val_dataset, batch_size=self._batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        return tuple(pad_sequence([s.type(torch.float32) for s in sequences], batch_first=True) for sequences in batch)
