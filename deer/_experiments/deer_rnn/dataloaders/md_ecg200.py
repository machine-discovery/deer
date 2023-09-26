import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class ECG200DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_file = "/home/yhl48/seq2seq/ECG200/ECG200_TRAIN.txt"
        self.test_file = "/home/yhl48/seq2seq/ECG200/ECG200_TEST.txt"

    def prepare_data(self):
        pass

    def load_data(self, file_path: str):
        data = np.loadtxt(file_path)
        x, y = data[:, 1:], data[:, 0]
        y[y == -1] = 0
        y = y.astype(np.int64)
        return x, y

    def setup(self, stage=None):
        self.train_x, self.train_y = self.load_data(self.train_file)
        self.test_x, self.test_y = self.load_data(self.test_file)

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(self.train_x), torch.tensor(self.train_y))
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataset = TensorDataset(torch.tensor(self.test_x), torch.tensor(self.test_y))
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataset = TensorDataset(torch.tensor(self.test_x), torch.tensor(self.test_y))
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        x = x[..., None]
        batch = x, y
        return batch


if __name__ == "__main__":
    import pdb
    dm = ECG200DataModule()
    dm.setup()
    for i, batch in enumerate(dm.train_dataloader()):
        x, y = dm.on_before_batch_transfer(batch, i)
        print(x.shape, y.shape)
        pdb.set_trace()
