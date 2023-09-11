from typing import Any, Tuple, Dict
import os
import argparse
from tqdm import tqdm
from functools import partial
from tensorboardX import SummaryWriter
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import optax
from deer.experiments.rnn_classification.dsets import get_dataset
from deer.experiments.rnn_classification.models import RNNClassifier


# jax.config.update("jax_platform_name", "cpu")

def loss_fn(params, static, xs: jnp.ndarray, targets: jnp.ndarray, model_states: Dict) \
        -> Tuple[jnp.ndarray, Dict]:
    # xs: (batch_size, nsamples, ninputs)
    # targets: (batch_size,) int
    model = eqx.combine(params, static)
    # ys: (batch_size, noutputs)
    ys, model_states = jax.vmap(model, in_axes=(0, None))(xs, model_states)
    loss = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(ys, targets)  # (batch_size,)
    # jax.debug.print("ys: {ys}", ys=jnp.argmax(ys, axis=-1))
    # jax.debug.print("targets: {targets}", targets=targets)
    # jax.debug.print("loss: {loss}", loss=loss)
    # jax.debug.print("correct: {correct}", correct=jnp.equal(jnp.argmax(ys, axis=-1), targets))
    # jax.debug.print("accuracy: {accuracy}", accuracy=jnp.mean(jnp.equal(jnp.argmax(ys, axis=-1), targets)))
    return jnp.mean(loss), model_states

@partial(jax.jit, static_argnames=("static",))
def calc_accuracy(params, static, xs: jnp.ndarray, targets: jnp.ndarray, model_states: Dict) \
        -> Tuple[jnp.ndarray, Dict]:
    # preds: (batch_size, noutputs), new_yinit_guess: [nlayers] + (batch_size, nsamples, nhiddens)
    model = eqx.combine(params, static)
    preds, model_states = jax.vmap(model, in_axes=(0, None))(xs, model_states)
    idx_preds = jnp.argmax(preds, axis=-1)  # (batch_size,)
    # jax.debug.print("idx_preds: {idx_preds}, targets: {targets}", idx_preds=idx_preds, targets=targets)
    correct = jnp.equal(idx_preds, targets)  # (batch_size,)
    # jax.debug.print("correct: {correct}", correct=correct)
    accuracy = jnp.mean(correct)  # ()
    # jax.debug.print("accuracy: {accuracy}", accuracy=accuracy)
    return accuracy, model_states

@partial(jax.jit, static_argnames=("static", "optimizer"))
def update_step(params, static, optimizer: optax.GradientTransformation, opt_state: Any,
                xs: jnp.ndarray, targets: jnp.ndarray,
                model_states: Dict) -> Tuple[RNNClassifier, Any, jnp.ndarray, Dict]:
    # xs: (batch_size, nsamples, ninputs)
    # targets: (batch_size, noutputs)
    (loss, model_states), grad = \
        jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params, static, xs, targets, model_states)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, model_states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--model", type=str, default="gruseq")
    parser.add_argument("--dataset", type=str, default="urbansound")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=9999999)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)

    # check the path
    logpath = "logs"
    path = os.path.join(logpath, f"version_{args.version}")
    if os.path.exists(path):
        raise ValueError(f"Path {path} already exists!")
    os.makedirs(path, exist_ok=True)

    # get the dataset
    dset = get_dataset(args.dataset)

    # split the dataset into train, val, test
    ntrain, nval, ntest = dset.splits()
    train_dset = torch.utils.data.Subset(dset, range(ntrain))
    val_dset = torch.utils.data.Subset(dset, range(ntrain, ntrain + nval))

    # get the dataloader
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    val_dloader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=True)

    # get the number of inputs and outputs from the dataset
    ninputs = dset.ninputs if dset.nquantization is None else args.width
    nclasses = dset.nclasses

    # get the model
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    if args.model == "grudeer":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="gru",
            eval_method="deer", nquantization=dset.nquantization, key=subkey)
    elif args.model == "gruseq":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="gru",
            eval_method="sequential", nquantization=dset.nquantization, key=subkey)
    elif args.model == "lstmdeer":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="lstm",
            eval_method="deer", nquantization=dset.nquantization, key=subkey)
    elif args.model == "lstmseq":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="lstm",
            eval_method="sequential", nquantization=dset.nquantization, key=subkey)
    else:
        raise ValueError(f"Unknown model {args.model}")

    # initialize the optimizer
    optimizer = optax.chain(
        # optax.clip(1.0),
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=args.lr),
    )
    # optimizer = optax.adam(learning_rate=args.lr)
    params, static = eqx.partition(model, eqx.is_array)
    opt_state = optimizer.init(params)
    model_states = eqx.nn.State(model)

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    step = 0
    for i in range(args.max_epochs):
        # training phase
        for batch in tqdm(train_dloader):
            xs, targets = batch
            # convert to jax
            xs = jnp.array(xs.numpy())
            targets = jnp.array(targets.numpy())
            # update step
            params, opt_state, loss, model_states = \
                update_step(params, static, optimizer, opt_state, xs, targets, model_states)
            step += 1
            summary_writer.add_scalar("train_loss", loss, step)

        # validation phase
        val_acc = 0.0
        for batch in tqdm(val_dloader, leave=False):
            xs, targets = batch
            # convert to jax
            xs = jnp.array(xs.numpy())
            targets = jnp.array(targets.numpy())
            # calculate the accuracy
            accuracy, model_states = calc_accuracy(params, static, xs, targets, model_states)
            val_acc += accuracy * len(targets)
        val_acc /= len(val_dset) * 1.0

        summary_writer.add_scalar("val_accuracy", val_acc, step)
        summary_writer.flush()

if __name__ == "__main__":
    main()
