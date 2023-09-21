"""
Code adapted from
https://github.com/RuslanKhalitov/ChordMixer/tree/main/dataloaders
"""

import torch
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import os
from pathlib import Path


def LoadGrayscale(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("L")


class PathFinderXDataset(Dataset):
    """Path Finder dataset."""

    def __init__(self, data_dir: str, transform=None):
        # https://github.com/HazyResearch/state-spaces/blob/a246043077dbff8563a6b172426443dced9a9d96/src/dataloaders/lra.py#L372C5-L372C69
        self._blacklist = {
            "/home/yhl48/seq2seq/lra_release/lra_release/pathfinder32/curv_baseline/imgs/0/sample_172.png"}

        self._data_dir = Path(data_dir)
        print('DATA DIR', self._data_dir)
        assert self._data_dir.is_dir(
        ), f"data_dir {str(self._data_dir)} does not exist"
        self._transform = transform
        samples = []

        metadata_list = [
            os.path.join(self._data_dir, "metadata", file)
            for file in os.listdir(os.path.join(self._data_dir, "metadata"))
            if file.endswith(".npy")
        ]
        for metadata_file in metadata_list:
            with open(metadata_file, "r") as f:
                for metadata in f.read().splitlines():
                    metadata = metadata.split()
                    image_path = Path(self._data_dir) / \
                        metadata[0] / metadata[1]
                    if str(
                            Path(
                                self._data_dir.stem) /
                            image_path) not in self._blacklist:
                        label = int(metadata[3])
                        samples.append((image_path, label))
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path, target = self._samples[idx]
        with open(self._data_dir / path, "rb") as f:
            sample = Image.open(f).convert("L")
            sample = self._transform(sample)
        return sample, target


class PathfinderXDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 1,
        batch_size: int = 32
    ):
        super().__init__()

        self._data_dir = data_dir
        self._batch_size = batch_size
        self._num_workers = num_workers

        transform = [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ]
        self._transform = transforms.Compose(transform)

    def setup(self, stage=None):
        dataset = PathFinderXDataset(self._data_dir, transform=self._transform)
        len_dataset = len(dataset)
        print('LEN DATASET', len_dataset)
        # LRA Setup
        val_len = int(0.1 * len_dataset)
        test_len = int(0.1 * len_dataset)
        # val_len = 50
        # test_len = 199000
        # test_len = 199800
        train_len = len_dataset - val_len - test_len
        self._train_dataset, self._val_dataset, self._test_dataset = random_split(
            dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1], -1)
        x = torch.permute(x, (0, 2, 1))  # (nbatch, nseq, ndim)
        # x += torch.randn_like(x) * 0.05
        batch = x, y
        return batch


if __name__ == "__main__":
    import pdb
    dm = PathfinderXDataModule(
        data_dir="/home/yhl48/seq2seq/lra_release/lra_release/pathfinder32/curv_baseline")
    dm.setup()
    for i, batch in enumerate(dm.train_dataloader()):
        x, y = dm.on_before_batch_transfer(batch, i)
        print(x.shape, y.shape)
        pdb.set_trace()
