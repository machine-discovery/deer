from typing import Tuple, Any

import jax
import jax.numpy as jnp
import torch
from torch.utils.data import Dataset

from dataloaders.lra_image import CIFAR10DataModule
from dataloaders.lra_pathfinderx import PathfinderXDataModule
from dataloaders.lra_listops_var import ListOpsVarDataModule
from dataloaders.lra_text import IMDBDataModule
from dataloaders.lra_retrieval import AANDataModule


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(batch) == 2
    x, y = batch
    x = jnp.asarray(x.numpy(), dtype=dtype)
    y = jnp.asarray(y.numpy())
    return x, y


def count_params(params):
    return sum(jnp.prod(jnp.asarray(p.shape)) for p in jax.tree_util.tree_leaves(params))


def get_datamodule(
    dset: str,
    batch_size: int
) -> Dataset:
    dset = dset.lower()
    if dset == "pathfinder128":
        return PathfinderXDataModule(
            data_dir="/home/yhl48/seq2seq/lra_release/lra_release/pathfinder128/curv_baseline",
            batch_size=batch_size
        )
    elif dset == "pathfinder64":
        return PathfinderXDataModule(
            data_dir="/home/yhl48/seq2seq/lra_release/lra_release/pathfinder64/curv_baseline",
            batch_size=batch_size
        )
    elif dset == "pathfinder32":
        return PathfinderXDataModule(
            data_dir="/home/yhl48/seq2seq/lra_release/lra_release/pathfinder32/curv_baseline",
            batch_size=batch_size
        )
    elif dset == "cifar10":
        return CIFAR10DataModule(
            data_dir="/home/yhl48/seq2seq/data",
            batch_size=batch_size,
            grayscale=False
        )
    elif dset == "cifar10grayscale":
        return CIFAR10DataModule(
            data_dir="/home/yhl48/seq2seq/data",
            batch_size=batch_size,
            grayscale=True
        )
    elif dset == "listops":
        return ListOpsVarDataModule(
            data_dir="/home/yhl48/seq2seq/lra_release/listops-1000",
            batch_size=batch_size
        )
    elif dset == "imdb":
        return IMDBDataModule(
            data_dir="/home/yhl48/seq2seq/imdb/IMDB",
            batch_size=batch_size
        )
    elif dset == "aan":
        return AANDataModule(
            data_dir="/home/yhl48/seq2seq/lra_release/lra_release/tsv_data",
            batch_size=batch_size
        )