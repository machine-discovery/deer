import argparse
import os
import sys
from functools import partial
from typing import Tuple, Any, Optional, List

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from flax import linen as nn
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils import prep_batch, count_params, get_datamodule, compute_metrics, grad_norm
from models import MultiScaleGRU
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

    y0 = yinit_guess[..., 0, :]

    # ypred: (batch_size, nclass)
    ypred, yinit_guess = jax.vmap(
        rollout, in_axes=(None, 2, 0, 2, None), out_axes=(0, 2)
    )(model, y0, x, yinit_guess, method)

    metrics = compute_metrics(ypred, y)
    loss, accuracy = metrics["loss"], metrics["accuracy"]
    # pdb.set_trace()
    return loss, (accuracy, yinit_guess)


@partial(jax.jit, static_argnames=("static", "optimizer", "method"))
def update_step(
    # model: eqx.Module,
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
    # pdb.set_trace()
    updates, opt_state = optimizer.update(grad, opt_state, params)
    # pdb.set_trace()
    params = optax.apply_updates(params, updates)
    # pdb.set_trace()
    gradnorm = grad_norm(grad)
    # pdb.set_trace()
    # model = eqx.combine(params, static)
    return params, opt_state, loss, accuracy, yinit_guess, gradnorm


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
    parser.add_argument("--ngru", type=int, default=10)
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
            "ecg200"
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
    dtype = jnp.float64
    batch_size = args.batch_size

    # check the path
    logpath = "logs"
    path = os.path.join(logpath, f"version_{args.version}")
    # if os.path.exists(path):
    #     raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # set up the model and optimizer
    # key = jax.random.PRNGKey(args.seed)
    # subkey1, subkey2, subkey3, subkey4, key = jax.random.split(key, 5)
    model = MultiScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=jax.random.PRNGKey(args.seed)
    )
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel))
    )  # (batch_size, nstates)
    # (nlayer, nchannel, batch_size, nstates)
    y0 = jnp.stack([jnp.stack([y0 for _ in range(nchannel)]) for _ in range(nlayer)])
    yinit_guess = jax.random.normal(
        jax.random.PRNGKey(1),
        (batch_size, nsequence, int(nstate / nchannel)),
    )
    # (nlayer, nchannel, batch_size, nsequence, nstates)
    yinit_guess = jnp.stack([jnp.stack([yinit_guess for _ in range(nchannel)]) for _ in range(nlayer)])

    optimizer = optax.chain(
        optax.clip_by_global_norm(max_norm=.1),
        optax.adam(learning_rate=args.lr)
    )
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)

    # count1 = count_params(combined_params["params"])
    # count2 = count_params(combined_params["mlp_params"])
    # print(count1, count2)

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    # training loop
    step = 0
    dm = get_datamodule(dset=args.dset, batch_size=args.batch_size)
    dm.setup()
    for epoch in tqdm(range(args.nepochs), file=sys.stderr):
        loop = tqdm(dm.train_dataloader(), total=len(dm.train_dataloader()), leave=False, file=sys.stderr)
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype)
            params, opt_state, loss, accuracy, yinit_guess, gradnorm = update_step(
                # model=model,
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

        val_loss = 0
        nval = 0
        val_acc = 0
        loop = tqdm(dm.val_dataloader(), total=len(dm.val_dataloader()), leave=False, file=sys.stderr)
        for i, batch in enumerate(loop):
            batch = dm.on_before_batch_transfer(batch, i)
            batch = prep_batch(batch, dtype)
            loss, (accuracy, _) = loss_fn(
                params, static, y0, batch, yinit_guess, method
            )
            val_loss += loss * len(batch[1])
            val_acc += accuracy * len(batch[1])
            nval += len(batch[1])
        val_loss /= nval
        val_acc /= nval
        summary_writer.add_scalar("val_loss", val_loss, step)
        summary_writer.add_scalar("val_accuracy", val_acc, step)


if __name__ == "__main__":
    main()