import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import arff


class RightWhaleCallsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_file = "/home/yhl48/seq2seq/rightwhalecalls/RightWhaleCalls_TRAIN.arff"
        self.test_file = "/home/yhl48/seq2seq/rightwhalecalls/RightWhaleCalls_TEST.arff"

    def prepare_data(self):
        pass

    def load_arff_data(self, file_path: str):
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df["class"] = df["class"].map({b'RightWhale': 0, b'NoWhale': 1})

        x = df.drop("class", axis=1).values[..., None]  # (ndata, nseq, ndim)
        y = df["class"].values  # [0, nclass)
        return x, y

    def setup(self, stage=None):
        self.train_x, self.train_y = self.load_arff_data(self.train_file)
        self.test_x, self.test_y = self.load_arff_data(self.test_file)

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(self.train_x), torch.tensor(self.train_y))
        print('LEN TRAIN DATASET', len(train_dataset))
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
        return batch


if __name__ == "__main__":
    import pdb
    dm = RightWhaleCallsDataModule()
    dm.setup()
    for i, batch in enumerate(dm.train_dataloader()):
        x, y = dm.on_before_batch_transfer(batch, i)
        print(x.shape, y.shape)
        pdb.set_trace()
