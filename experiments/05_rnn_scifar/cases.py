from typing import Tuple, List, Dict, Any, Union
import os
from abc import ABC, abstractmethod, abstractproperty
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torchvision
from torchvision.transforms import v2


FDIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

class Case(ABC, torch.utils.data.Dataset):
    @abstractproperty
    def with_embedding(self) -> bool:
        pass

    @abstractproperty
    def reduce_length(self) -> int:
        pass

    @abstractproperty
    def num_inps(self) -> int:
        pass

    @abstractproperty
    def num_outs(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def train_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        # output: (length, num_outs)
        # target: (length, num_outs) or (length,) int or just an int
        # returns: ()
        pass

    @abstractmethod
    def val_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Any]]:
        # output: (length, num_outs)
        # target: (length, num_outs) or (length,) int or just an int
        # returns: ()
        pass

    def test_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        val = self.val_loss_fn(output, target)
        assert isinstance(val, jnp.ndarray)
        return val

    @abstractproperty
    def train_idxs(self) -> jnp.ndarray:
        pass  # (num_train,) int

    @abstractproperty
    def val_idxs(self) -> jnp.ndarray:
        pass  # (num_val,) int

    @abstractproperty
    def test_idxs(self) -> jnp.ndarray:
        pass  # (num_test,) int

class ToChannelLast(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.movedim(x, 0, -1)

class ToSequential(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (height, width, channels)
        return x.reshape(-1, x.shape[-1])

class SeqCIFAR10(Case):
    def __init__(self, rootdir: str = os.path.join(FDIR, "data", "cifar10"), val_pct: float = 0.2, normtype: int = 1):
        np.random.seed(123)
        if normtype == 1:
            std = (0.2470, 0.2435, 0.2616)
        elif normtype == 2:
            std = (0.2023, 0.1994, 0.2010)
        else:
            raise ValueError(f"Invalid normtype: {normtype}")
        tfms = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), std),
            ToChannelLast(),
            ToSequential(),
        ])
        target_tfm = v2.ToDtype(torch.long)
        self._train = torchvision.datasets.CIFAR10(root=rootdir, transform=tfms, target_transform=target_tfm,
                                                   train=True, download=True)
        self._test = torchvision.datasets.CIFAR10(root=rootdir, transform=tfms, target_transform=target_tfm,
                                                  train=False, download=True)
        self._ntrain = int((1 - val_pct) * len(self._train))

        # get the train and validation indices
        all_train_idxs = np.arange(len(self._train))
        np.random.shuffle(all_train_idxs)
        self._train_idxs = all_train_idxs[:self._ntrain]
        self._val_idxs = all_train_idxs[self._ntrain:]

    @property
    def with_embedding(self) -> bool:
        return False

    @property
    def reduce_length(self) -> int:
        return True

    @property
    def num_inps(self) -> int:
        return 3

    @property
    def num_outs(self) -> int:
        return 10

    def __len__(self) -> int:
        return len(self._train) + len(self._test)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < len(self._train):
            img, target = self._train[i]
        else:
            img, target = self._test[i - len(self._train)]
        return img, target

    def train_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        # output: (num_outs,)
        # target: just an int
        # returns: ()
        return optax.softmax_cross_entropy_with_integer_labels(output, target)

    def val_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # output: (num_outs,)
        # target: just an int
        # returns: ()
        acc = (jnp.argmax(output) == target) * (-1.0)  # negative to make it a loss, not a reward
        bce = optax.softmax_cross_entropy_with_integer_labels(output, target)
        return acc, {"acc": -acc, "bce": bce}

    def test_loss_fn(self, output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        return (jnp.argmax(output) == target)

    @property
    def train_idxs(self) -> jnp.ndarray:
        return jnp.array(self._train_idxs)

    @property
    def val_idxs(self) -> jnp.ndarray:
        return jnp.array(self._val_idxs)

    @property
    def test_idxs(self) -> jnp.ndarray:
        return jnp.array(list(range(len(self._train), len(self))))
