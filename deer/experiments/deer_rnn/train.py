import argparse
import os
import dill as pickle
import sys
from functools import partial
from typing import Tuple, Any, Optional, List
from glob import glob
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from flax import serialization
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import prep_batch, count_params, get_datamodule, compute_metrics, grad_norm
from models import MultiScaleGRU, SingleScaleGRU
from deer.seq1d import seq1d
import pdb

# # run on cpu
# jax.config.update('jax_platform_name', 'cpu')
# enable float 64
jax.config.update('jax_enable_x64', True)
jax.config.update("jax_debug_nans", True)


@partial(jax.jit, static_argnames=("model", "method"))
def rollout(
    model: eqx.Module,
    y0: jnp.ndarray,
    inputs: jnp.ndarray,
    yinit_guess: List[jnp.ndarray],
    method: str = "deer_rnn",
) -> jnp.ndarray:
    # roll out the model's predictions with y being the state
    # y0: (nstates,)
    # inputs: (ntpts,)
    # yinit_guess: (ntpts, nstates)
    # returns: (ntpts, nstates)

    if method == "multiscale_deer":
        # multiple channels from multiple scales -- each channel has its own params
        # do the same multiple times by reusing the same set of parameters
        out, yinit_guess = model(inputs, y0, yinit_guess)
        # pdb.set_trace()
        # jax.debug.print("{s}", s=out.shape)
        return out.mean(axis=0), yinit_guess
    else:
        raise NotImplementedError()


@partial(jax.jit, static_argnames=("static", "method"))
def loss_fn(
    params: Any,
    static: Any,
    y0: jnp.ndarray,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    yinit_guess: List[jnp.ndarray],
    method: str = "deer_rnn"
) -> jnp.ndarray:
    """
    y0 (nlayer, nchannel, nbatch, nstate)
    yinit_guess (nlayer, nchannel, nbatch, nsequence, nstate)
    batch (nbatch, nseq, ndim) (nbatch,)
    """
    # compute the loss
    # batch: (batch_size, ntpts, ninp), (batch_size, ntpts, nstates)
    # yinit_guess (ndata, ntpts, nstates)
    # weight: (ntpts,)
    model = eqx.combine(params, static)
    x, y = batch
    # (nlayer, nchannel, batch_size, nsequence, nstates)
    # TODO replace this with something more elegant

    # remove this line
    # y0 = yinit_guess[..., 0, :]

    # ypred: (batch_size, nclass)
    ypred, yinit_guess = jax.vmap(
        rollout, in_axes=(None, 0, 0, 0, None), out_axes=(0, 2)
    )(model, y0, x, yinit_guess, method)

    metrics = compute_metrics(ypred, y)
    loss, accuracy = metrics["loss"], metrics["accuracy"]
    # pdb.set_trace()
    return loss, (accuracy, yinit_guess)


@partial(jax.jit, static_argnames=("static", "optimizer", "method"))
def update_step(
    params: Any,
    static: Any,
    optimizer: optax.GradientTransformation,
    opt_state: Any,
    batch: Tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray,
    yinit_guess: jnp.ndarray,
    method: str = "deer_rnn"
) -> Tuple[optax.Params, Any, jnp.ndarray, jnp.ndarray]:
    """
    y0 (nlayer, nchannel, batch_size, nstates)
    yinit_guess (nlayer, nchannel, batch_size, nsequence, nstates)
    batch (nbatch, nseq, ndim) (nbatch,)
    """
    # params, static = eqx.partition(model, eqx.is_array)

    (loss, (accuracy, yinit_guess)), grad = jax.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True
    )(params, static, y0, batch, yinit_guess, method)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    gradnorm = grad_norm(grad)
    return new_params, opt_state, loss, accuracy, yinit_guess, gradnorm


def main():
    # set up argparse for the hyperparameters above
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--nepochs", type=int, default=999999999)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--method", type=str, default="deer_rnn")
    parser.add_argument("--ninps", type=int, default=1)
    parser.add_argument("--nstates", type=int, default=256)
    parser.add_argument("--nsequence", type=int, default=1024)
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--nlayer", type=int, default=5)
    parser.add_argument("--nchannel", type=int, default=4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument(
        "--dset", type=str, default="pathfinder32",
        choices=[
            "imdb",
            "pathfinder128",
            "pathfinder64",
            "pathfinder32",
            "cifar10",
            "cifar10grayscale",
            "listops",
            "aan",
            "sanity_check",
            "eigenworms",
            "ecg200",
            "rightwhalecalls"
        ],
    )
    args = parser.parse_args()

    method = args.method
    ninp = args.ninps
    nstate = args.nstates
    nsequence = args.nsequence
    nclass = args.nclass
    nlayer = args.nlayer
    nchannel = args.nchannel
    batch_size = args.batch_size
    patience = args.patience

    if args.precision == 32:
        dtype = jnp.float32
    elif args.precision == 64:
        dtype = jnp.float64
    else:
        raise ValueError("Only 32 or 64 accepted")

    # check the path
    logpath = "logs"
    path = os.path.join(logpath, f"version_{args.version}")
    # if os.path.exists(path):
    #     raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    key = jax.random.PRNGKey(args.seed)
    if nchannel > 1:
        model = MultiScaleGRU(
            ninp=ninp,
            nchannel=nchannel,
            nstate=nstate,
            nlayer=nlayer,
            nclass=nclass,
            key=key
        )
    elif nchannel == 1:
        model = SingleScaleGRU(
            ninp=ninp,
            nchannel=nchannel,
            nstate=nstate,
            nlayer=nlayer,
            nclass=nclass,
            key=key
        )
    else:
        raise ValueError("nchannnel must be a positive integer")
    model = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model)
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel)),
        dtype=dtype
    )  # (batch_size, nstates)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate / nchannel)),
        dtype=dtype
    )  # (batch_size, nsequence, nstates)

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=1),
        optax.adam(learning_rate=args.lr)
    )
    # checkpoint_path = os.path.join(path, "best_model.pkl")
    # model = eqx.tree_deserialise_leaves(checkpoint_path, model)
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
    for epoch in tqdm(range(args.nepochs), file=sys.stderr):
        loop = tqdm(dm.train_dataloader(), total=len(dm.train_dataloader()), leave=False, file=sys.stderr)
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype)
            # replace yinit_guess with _
            params, opt_state, loss, accuracy, _, gradnorm = update_step(
                params=params,
                static=static,
                optimizer=optimizer,
                opt_state=opt_state,
                batch=batch,
                y0=y0,
                yinit_guess=yinit_guess,
                method=method
            )
            # y0 = yinit_guess[:, 0, :]
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
                loss, (accuracy, _) = loss_fn(
                    inference_params, inference_static, y0, batch, yinit_guess, method
                )
                # loss, (accuracy, _) = loss_fn(
                #     params, static, y0, batch, yinit_guess, method
                # )
                val_loss += loss * len(batch[1])
                val_acc += accuracy * len(batch[1])
                nval += len(batch[1])
            val_loss /= nval
            val_acc /= nval
            summary_writer.add_scalar("val_loss", val_loss, step)
            summary_writer.add_scalar("val_accuracy", val_acc, step)
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

if __name__ == "__main__":
    main()
