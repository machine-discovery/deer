import sys
import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import arff

sys.path.append("../")


class EigenWormsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.train_file = "neuralrde_split/eigenworms_train.pkl"
        self.val_file = "neuralrde_split/eigenworms_val.pkl"
        self.test_file = "neuralrde_split/eigenworms_test.pkl"

    def prepare_data(self):
        pass

    def load_arff_data(self, file_path: str):
        # not in use
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df["eigenWormMultivariate_attribute"] = df["eigenWormMultivariate_attribute"].apply(lambda cell: np.array(cell.tolist()))
        df["target"] = df["target"].str.decode("utf-8").astype(int)

        x = np.array(df["eigenWormMultivariate_attribute"].tolist())
        x = np.transpose(x, (0, 2, 1))  # (ndata, nseq, ndim)
        y = np.array(df["target"].tolist()) - 1  # [0, nclass)
        return x, y

    def setup(self, stage=None):
        with open(self.train_file, "rb") as f:
            x, y = pickle.load(f)
            self._train_dataset = TensorDataset(x, y)
        with open(self.val_file, "rb") as f:
            x, y = pickle.load(f)
            self._val_dataset = TensorDataset(x, y)
        with open(self.test_file, "rb") as f:
            x, y = pickle.load(f)
            self._test_dataset = TensorDataset(x, y)
        print('LEN TRAIN DATASET', len(self._train_dataset))
        print('LEN VAL DATASET', len(self._val_dataset))
        print('LEN TEST DATASET', len(self._test_dataset))

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )
        return test_dataloader

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch


if __name__ == "__main__":
    import pdb
    dm = EigenWormsDataModule()
    dm.setup()
    for i, batch in enumerate(dm.train_dataloader()):
        x, y = dm.on_before_batch_transfer(batch, i)
        print(x.shape, y.shape)
        pdb.set_trace()
