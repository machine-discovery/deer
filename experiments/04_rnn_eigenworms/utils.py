from typing import Tuple, Any, Dict

import jax
import jax.numpy as jnp
import optax
import torch
import pytorch_lightning as pl

from dataloaders.md_eigenworms import EigenWormsDataModule


def prep_batch(
    batch: Tuple[torch.Tensor, torch.Tensor],
    dtype: Any
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert len(batch) == 2
    x, y = batch
    x = jnp.asarray(x.numpy(), dtype=dtype)
    y = jnp.asarray(y.numpy())
    return x, y


def count_params(params) -> jnp.ndarray:
    return sum(jnp.prod(jnp.asarray(p.shape)) for p in jax.tree_util.tree_leaves(params))


def grad_norm(grads) -> jnp.ndarray:
    flat_grads = jnp.concatenate([jnp.reshape(g, (-1,)) for g in jax.tree_util.tree_leaves(grads)])
    return jnp.linalg.norm(flat_grads)


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy
    }
    return metrics


def get_datamodule(
    dset: str,
    batch_size: int,
    datafile: str = "neuralrde"
) -> pl.LightningDataModule:
    dset = dset.lower()
    datafile = datafile.lower()
    if datafile not in ["neuralrde", "lem"]:
        raise NotImplementedError()
    if dset == "eigenworms":
        return EigenWormsDataModule(
            batch_size=batch_size,  # nseq = 17984, nclass = 5
            datafile=datafile
        )
    else:
        return NotImplementedError("only eigenworms dataset is available")
