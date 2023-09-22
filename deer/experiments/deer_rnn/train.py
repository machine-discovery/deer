import argparse
import os
import sys
from functools import partial
from typing import Tuple, Any, List
from glob import glob

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import prep_batch, count_params, get_datamodule, compute_metrics, grad_norm
from models import SingleScaleGRU


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
    return loss, accuracy


@partial(jax.jit, static_argnames=("static", "optimizer"))
def update_step(
    params: Any,
    static: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray,
    yinit_guess: jnp.ndarray,
) -> Tuple[optax.Params, Any, jnp.ndarray, jnp.ndarray]:
    """
    batch (nbatch, nsequence, ninp) (nbatch,)
    y0 (nbatch, nstate)
    yinit_guess (nbatch, nsequence, nstate)
    """
    (loss, accuracy), grad = jax.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True
    )(params, static, y0, batch, yinit_guess)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    gradnorm = grad_norm(grad)
    return new_params, opt_state, loss, accuracy, gradnorm


def main():
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--nepochs", type=int, default=999999999)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--ninps", type=int, default=6)
    parser.add_argument("--nstates", type=int, default=32)
    parser.add_argument("--nsequence", type=int, default=17984)
    parser.add_argument("--nclass", type=int, default=5)
    parser.add_argument("--nlayer", type=int, default=5)
    parser.add_argument("--nchannel", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--patience_metric", type=str, default="accuracy")
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--use_scan", action="store_true", help="Doing --use_scan sets it to True")
    parser.add_argument(
        "--dset", type=str, default="eigenworms",
        choices=[
            "eigenworms",
        ],
    )
    args = parser.parse_args()

    ninp = args.ninps
    nstate = args.nstates
    nsequence = args.nsequence
    nclass = args.nclass
    nlayer = args.nlayer
    nchannel = args.nchannel
    batch_size = args.batch_size
    patience = args.patience
    patience_metric = args.patience_metric
    use_scan = args.use_scan

    if args.precision == 32:
        dtype = jnp.float32
    elif args.precision == 64:
        dtype = jnp.float64
    else:
        raise ValueError("Only 32 or 64 accepted")
    print(f"dtype is {dtype}")
    print(f"use_scan is {use_scan}")
    print(f"patience_metric is {patience_metric}")

    # check the path
    logpath = "logs_instance_3"
    path = os.path.join(logpath, f"version_{args.version}")
    # if os.path.exists(path):
    #     raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    assert nchannel == 1, "currently only support 1 channel"
    model = SingleScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=key,
        use_scan=use_scan
    )
    model = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model)
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel)),
        dtype=dtype
    )  # (nbatch, nstate)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate / nchannel)),
        dtype=dtype
    )  # (nbatch, nsequence, nstate)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1),
        optax.adam(learning_rate=args.lr)
    )
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    print(f"Total parameter count: {count_params(params)}")

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    # training loop
    step = 0
    dm = get_datamodule(dset=args.dset, batch_size=args.batch_size)
    dm.setup()
    best_val_acc = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(args.nepochs), file=sys.stderr):
        loop = tqdm(dm.train_dataloader(), total=len(dm.train_dataloader()), leave=False, file=sys.stderr)
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype)
            params, opt_state, loss, accuracy, gradnorm = update_step(
                params=params,
                static=static,
                optimizer=optimizer,
                opt_state=opt_state,
                batch=batch,
                y0=y0,
                yinit_guess=yinit_guess,
            )
            summary_writer.add_scalar("train_loss", loss, step)
            summary_writer.add_scalar("train_accuracy", accuracy, step)
            summary_writer.add_scalar("gru_gradnorm", gradnorm, step)
            step += 1

        inference_model = eqx.combine(params, static)
        inference_model = eqx.tree_inference(inference_model, value=True)
        inference_params, inference_static = eqx.partition(inference_model, eqx.is_array)
        if epoch % 1 == 0:
            val_loss = 0
            nval = 0
            val_acc = 0
            loop = tqdm(dm.val_dataloader(), total=len(dm.val_dataloader()), leave=False, file=sys.stderr)
            for i, batch in enumerate(loop):
                try:
                    batch = dm.on_before_batch_transfer(batch, i)
                except Exception():
                    pass
                batch = prep_batch(batch, dtype)
                loss, accuracy = loss_fn(
                    inference_params, inference_static, y0, batch, yinit_guess
                )
                val_loss += loss * len(batch[1])
                val_acc += accuracy * len(batch[1])
                nval += len(batch[1])
            val_loss /= nval
            val_acc /= nval
            summary_writer.add_scalar("val_loss", val_loss, step)
            summary_writer.add_scalar("val_accuracy", val_acc, step)
            if patience_metric == "accuracy":
                if val_acc > best_val_acc:
                    patience = args.patience
                    best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(path, f"best_model_epoch_{epoch}_step_{step}.pkl")
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)
                else:
                    patience -= 1
                    if patience == 0:
                        print(f"The validation accuracy stopped improving, training ends here at epoch {epoch} and step {step}!")
                        break
            elif patience_metric == "loss":
                if val_loss < best_val_loss:
                    patience = args.patience
                    best_val_loss = val_loss
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                    for f in glob(f"{path}/best_model_epoch_*"):
                        os.remove(f)
                    checkpoint_path = os.path.join(path, f"best_model_epoch_{epoch}_step_{step}.pkl")
                    best_model = eqx.combine(params, static)
                    eqx.tree_serialise_leaves(checkpoint_path, best_model)
                else:
                    patience -= 1
                    if patience == 0:
                        print(f"The validation loss stopped improving at {best_val_loss} with accuracy {best_val_acc}, training ends here at epoch {epoch} and step {step}!")
                        break
            else:
                raise ValueError


if __name__ == "__main__":
    main()
