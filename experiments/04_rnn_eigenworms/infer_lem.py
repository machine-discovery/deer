import argparse
import os
import sys
from functools import partial
from typing import Tuple, Any, List
from glob import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm

from utils import prep_batch, count_params, get_datamodule, compute_metrics
from models import ScaledLEM
from sklearn.metrics import confusion_matrix

# # run on cpu
# jax.config.update('jax_platform_name', 'cpu')
# enable float 64
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)


@partial(jax.jit, static_argnames=("model"))
def rollout(
    model: eqx.Module,
    y0: jnp.ndarray,
    inputs: jnp.ndarray,
    yinit_guess: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    y0 (nstate,)
    inputs (nsequence, ninp)
    yinit_guess (nsequence, nstate)

    return: (nclass,)
    """
    out = model(inputs, y0, yinit_guess)
    # return out[-1]
    return out.mean(axis=0)


@partial(jax.jit, static_argnames=("static"))
def loss_fn(
    params: Any,
    static: Any,
    y0: jnp.ndarray,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    yinit_guess: List[jnp.ndarray],
) -> jnp.ndarray:
    """
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    batch (nbatch, nsequence, ninp) (nbatch,)
    """
    model = eqx.combine(params, static)
    x, y = batch

    # ypred: (nbatch, nclass)
    ypred = jax.vmap(
        rollout, in_axes=(None, 0, 0, 0), out_axes=(0)
    )(model, y0, x, yinit_guess)

    metrics = compute_metrics(ypred, y)
    loss, accuracy = metrics["loss"], metrics["accuracy"]
    return loss, accuracy, ypred


def main():
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nepochs", type=int, default=999999999)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--ninps", type=int, default=6)
    parser.add_argument("--nstates", type=int, default=32)
    parser.add_argument("--nsequence", type=int, default=17984)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.0016)
    parser.add_argument("--use_scan", action="store_true", help="Doing --use_scan sets it to True")
    parser.add_argument(
        "--dset", type=str, default="eigenworms",
        choices=[
            "eigenworms",
        ],
    )
    args = parser.parse_args()

    # set seed for pytorch
    torch.manual_seed(42)

    ninp = args.ninps
    nstate = args.nstates
    nsequence = args.nsequence
    nclass = args.nclass
    batch_size = args.batch_size
    use_scan = args.use_scan
    dt = args.dt

    if args.precision == 32:
        dtype = jnp.float32
    elif args.precision == 64:
        dtype = jnp.float64
    else:
        raise ValueError("Only 32 or 64 accepted")

    # check the path
    logpath = "logs_instance_3"
    path = os.path.join(logpath, f"version_{args.version}")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    model = ScaledLEM(
        ninp=ninp,
        nstate=nstate,
        nclass=nclass,
        key=key,
        use_scan=use_scan,
        dt=dt
    )
    model = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model)
    y0 = jnp.zeros(
        (batch_size, int(nstate * 2)),
        dtype=dtype
    )  # (nbatch, nstate)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate * 2)),
        dtype=dtype
    )  # (nbatch, nsequence, nstate)

    # checkpoint_path = glob(f"{path}/best_model_epoch_*")[0]
    # checkpoint_path = glob(f"{path}/last_epoch_model*")[0]
    checkpoint_path = glob(f"{path}/step_840*")[0]
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    params, static = eqx.partition(model, eqx.is_array)
    print(f"Total parameter count: {count_params(params)}")

    dm = get_datamodule(dset=args.dset, batch_size=args.batch_size, datafile="lem")
    dm.setup()
    inference_model = eqx.combine(params, static)
    inference_model = eqx.tree_inference(inference_model, value=True)
    inference_params, inference_static = eqx.partition(inference_model, eqx.is_array)

    true_labels = []
    pred_labels = []
    ntest = 0
    test_acc = 0
    loop = tqdm(dm.test_dataloader(), total=len(dm.test_dataloader()), leave=False, file=sys.stderr)
    for i, batch in enumerate(loop):
        try:
            batch = dm.on_before_batch_transfer(batch, i)
        except Exception():
            pass
        batch = prep_batch(batch, dtype)
        loss, accuracy, ypred = loss_fn(
            inference_params, inference_static, y0, batch, yinit_guess
        )
        test_acc += accuracy * len(batch[1])
        ntest += len(batch[1])
        preds = jnp.argmax(ypred, axis=-1)
        true_labels.extend(batch[1].tolist())
        pred_labels.extend(preds.tolist())
    test_acc /= ntest
    print(f"Version {args.version} with {dtype}: Total number of test samples: {ntest}. Total number of correct predictions: {test_acc * ntest}. Accuracy: {test_acc}")
    print("")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)


if __name__ == "__main__":
    main()
