import pickle
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.io import arff


class EigenWormsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        super().__init__()

        self.batch_size = batch_size
        # self.train_file = "/home/yhl48/seq2seq/eigenworm/EigenWorms_TRAIN.arff"
        # self.test_file = "/home/yhl48/seq2seq/eigenworm/EigenWorms_TEST.arff"
        self.train_file = "/home/yhl48/seq2seq/eigenworm/neuralrde_split/eigenworms_train.pkl"
        self.val_file = "/home/yhl48/seq2seq/eigenworm/neuralrde_split/eigenworms_val.pkl"
        self.test_file = "/home/yhl48/seq2seq/eigenworm/neuralrde_split/eigenworms_test.pkl"

    def prepare_data(self):
        pass

    def load_arff_data(self, file_path: str):
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        df["eigenWormMultivariate_attribute"] = df["eigenWormMultivariate_attribute"].apply(lambda cell: np.array(cell.tolist()))
        df["target"] = df["target"].str.decode("utf-8").astype(int)

        x = np.array(df["eigenWormMultivariate_attribute"].tolist())
        x = np.transpose(x, (0, 2, 1))  # (ndata, nseq, ndim)
        y = np.array(df["target"].tolist()) - 1  # [0, nclass)
        return x, y

    def setup(self, stage=None):
        # train_x, train_y = self.load_arff_data(self.train_file)
        # test_x, test_y = self.load_arff_data(self.test_file)

        # x = np.concatenate([train_x, test_x], axis=0)
        # y = np.concatenate([train_y, test_y], axis=0)

        # self._dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))

        # train_length = int(len(self._dataset) * 0.7)
        # val_length = int(len(self._dataset) * 0.15)
        # test_length = len(self._dataset) - train_length - val_length

        # self._train_dataset, self._val_dataset, self._test_dataset = random_split(
        #     self._dataset, [train_length, val_length, test_length],
        #     generator=torch.Generator().manual_seed(42)
        # )
        with open(self.train_file, "rb") as f:
            x, y = pickle.load(f)
            # mean, std = x.mean(dim=(0, 1)), x.std(dim=(0, 1))
            # print("NORMALISE INPUT")
            # x = (x - mean) / std
            # print(f"After normalising, training input mean={x.mean()}, std={x.std()}")
            self._train_dataset = TensorDataset(x, y)
        with open(self.val_file, "rb") as f:
            x, y = pickle.load(f)
            # x = (x - mean) / std
            # print(f"After normalising, validation input mean={x.mean()}, std={x.std()}")
            self._val_dataset = TensorDataset(x, y)
        with open(self.test_file, "rb") as f:
            x, y = pickle.load(f)
            # x = (x - mean) / std
            # print(f"After normalising, test input mean={x.mean()}, std={x.std()}")
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
