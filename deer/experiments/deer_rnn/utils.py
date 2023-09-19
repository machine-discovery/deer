from typing import Tuple, Any, Dict

import jax
import jax.numpy as jnp
import optax
import torch
from torch.utils.data import Dataset

from dataloaders.lra_image import CIFAR10DataModule
from dataloaders.lra_pathfinderx import PathfinderXDataModule
from dataloaders.lra_listops_var import ListOpsVarDataModule
from dataloaders.lra_text import IMDBDataModule
from dataloaders.lra_retrieval import AANDataModule
from dataloaders.md_sine import TimeSeriesDataModule
from dataloaders.md_eigenworms import EigenWormsDataModule
from dataloaders.md_ecg200 import ECG200DataModule


import pdb


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(batch) == 2
    x, y = batch
    # x = jnp.asarray(x.numpy()[:, ::2, :], dtype=dtype)
    x = jnp.asarray(x.numpy(), dtype=dtype)
    y = jnp.asarray(y.numpy())
    return x, y


def count_params(params):
    return sum(jnp.prod(jnp.asarray(p.shape)) for p in jax.tree_util.tree_leaves(params))


def grad_norm(grads):
    flat_grads = jnp.concatenate([jnp.reshape(g, (-1,)) for g in jax.tree_util.tree_leaves(grads)])
    return jnp.linalg.norm(flat_grads)


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    # print(logits)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    # jax.debug.print("{loss}", loss=loss)
    # jax.debug.print("{logits}", logits=logits)

    # print_argmax = jax.tree_map(print, jnp.argmax(logits, -1))
    # jax.debug.print("{classes} {labels}", classes=jnp.argmax(logits, -1), labels=labels)
    # print(loss)
    # pdb.set_trace()
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


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
    elif dset == "sanity_check":
        return TimeSeriesDataModule(
            batch_size=batch_size,
            nclass=2,
            seq_length=1000
        )
    elif dset == "eigenworms":
        return EigenWormsDataModule(
            batch_size=batch_size,  # nseq = 17984, nclass = 5
        )
    elif dset == "ecg200":
        return ECG200DataModule(
            batch_size=batch_size,  # nseq = 96, nclass = 2
        )
