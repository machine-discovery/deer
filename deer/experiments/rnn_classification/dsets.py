from typing import Tuple, Optional
import os
import pickle
import torch
import numpy as np


FDIR = os.path.dirname(os.path.realpath(__file__))

class UEADataset(torch.utils.data.Dataset):
    prefix: str

    def __init__(self):
        fdir = os.path.join(FDIR, "data", self.prefix)
        with open(os.path.join(fdir, f"{self.prefix}_TRAIN.pkl"), "rb") as f:
            train_data, train_target, target_map = pickle.load(f)
        with open(os.path.join(fdir, f"{self.prefix}_TEST.pkl"), "rb") as f:
            test_data, test_target, _ = pickle.load(f)

        # normalize the data
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        # shuffle the train and test data
        np.random.seed(0)
        train_perm = np.random.permutation(len(train_data))
        train_data = train_data[train_perm]
        train_target = train_target[train_perm]
        test_perm = np.random.permutation(len(test_data))
        test_data = test_data[test_perm]
        test_target = test_target[test_perm]

        # concatenate the train data and test data
        data = np.concatenate([train_data, test_data], axis=0)[..., None]  # (ndata, nsamples, 1)
        # data = data[:, ::2, :]  # (ndata, nsamples, 1)
        target = np.concatenate([train_target, test_target], axis=0)  # (ndata,) int
        self.target_map = target_map

        # convert the data to torch tensors
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.preprocess(self.data[index]), self.target[index]

    @property
    def nclasses(self) -> int:
        return len(self.target_map)

    def nquantization(self) -> Optional[int]:
        return None

    def splits(self) -> Tuple[int, int, int]:
        pass

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x

class AudioDataset(UEADataset):
    @property
    def ninputs(self) -> int:
        return 1

    @property
    def nquantization(self) -> Optional[int]:
        return None
        # return 256

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x
        # return torch.round((x + 1) / 2 * (self.nquantization - 1)).to(torch.long)

class UrbanSound(AudioDataset):
    prefix = "UrbanSound"

    def splits(self) -> Tuple[int, int, int]:
        # train, val, test
        return 2173, 2717 - 2173, 2718

class AbnormalHeartbeat(AudioDataset):
    prefix = "AbnormalHeartbeat"

    def splits(self) -> Tuple[int, int, int]:
        # train, val, test
        return 204 - 40, 40, 205

class RightWhaleCalls(AudioDataset):
    prefix = "RightWhaleCalls"

    def splits(self) -> Tuple[int, int, int]:
        # train, val, test
        return 10934 - 2000, 2000, 1962

def get_dataset(dataset: str) -> torch.utils.data.Dataset:
    dataset = dataset.lower()
    if dataset == "urbansound":
        return UrbanSound()
    elif dataset == "abnormalheartbeat":
        return AbnormalHeartbeat()
    elif dataset == "rightwhalecalls":
        return RightWhaleCalls()
    else:
        raise ValueError(f"Unknown dataset {dataset}!")
