"""
Code adapted from
https://github.com/RuslanKhalitov/ChordMixer/tree/main/dataloaders
"""

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import pytorch_lightning as pl


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 1,
        batch_size: int = 32,
        grayscale: bool = True
    ):
        super().__init__()

        # Save parameters to self
        # self.data_dir = utils.get_original_cwd() + data_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_aug = False
        self.grayscale = grayscale

        self.output_channels = 10

        if self.grayscale:
            train_transform = [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0)
            ]
        else:
            train_transform = [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                )
            ]

        val_test_transform = train_transform

        if self.data_aug:
            train_transform = [
                transforms.RandomCrop(32, padding=4, padding_mode="symmetric"),
                transforms.RandomHorizontalFlip(),
            ] + train_transform

        self.train_transform = transforms.Compose(train_transform)
        self.val_test_transform = transforms.Compose(val_test_transform)

    def prepare_data(self):
        # download data, train then test
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar10 = datasets.CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transform,
            )
            self.train_dataset, self.val_dataset = random_split(
                cifar10, [
                    45000, 5000], generator=torch.Generator().manual_seed(
                    getattr(
                        self, "seed", 42)), )
            # self.train_dataset, self.val_dataset, _ = random_split(
            #     cifar10, [
            #         10000, 5000, 35000], generator=torch.Generator().manual_seed(
            #         getattr(
            #             self, "seed", 42)), )
            print('LEN TRAIN DATASET', len(self.train_dataset))
            # self.val_test_dataset = datasets.CIFAR10(
            #     self.data_dir,
            #     train=False,
            #     transform=self.val_test_transform,
            # )
        else:
            self.test_dataset = datasets.CIFAR10(
                self.data_dir,
                train=False,
                transform=self.val_test_transform,
            )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            # self.val_test_dataset,
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x_shape = x.shape
        x = x.view(x_shape[0], x_shape[1], -1)
        x = torch.permute(x, (0, 2, 1))
        batch = x, y
        return batch


if __name__ == "__main__":
    import pdb
    dm = CIFAR10DataModule(
        data_dir="/home/yhl48/seq2seq/data"
    )
    dm.setup()
    print(len(dm.train_dataloader()))
    for i, batch in enumerate(dm.train_dataloader()):
        x, y = dm.on_before_batch_transfer(batch, i)
        print(x.shape, y.shape)
        pdb.set_trace()
