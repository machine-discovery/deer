from typing import Callable, Optional, List, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from deer.seq1d import seq1d
from utils import prep_batch, count_params, get_datamodule, compute_metrics, grad_norm
import sys
import pdb

def clscall(module: eqx.Module):
    return module.__class__.__call__

class LSTMWrapper(eqx.Module):
    lstm: eqx.nn.LSTMCell
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, *, key: jax.random.PRNGKey):
        self.lstm = eqx.nn.LSTMCell(input_size, hidden_size // 2, key=key)
        self.hidden_size = hidden_size

    def __call__(self, xinput: jnp.ndarray, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # xinput: (input_size,)
        # state: (hidden_size,)
        # split the state into two tensors of shape (hidden_size // 2,)
        state1, state2 = jnp.split(state, 2, axis=-1)
        state = (state1, state2)
        next_state = self.lstm(xinput, state)

        # concatenate the two tensors of shape (hidden_size // 2,) to (hidden_size,)
        next_state = jnp.concatenate(next_state, axis=-1)
        return next_state

class CellLayer(eqx.Module):
    cell: eqx.Module
    eval_method: str
    cell_type: str

    def __init__(self, ninputs: int, nhiddens: int, cell_type: str = "gru", eval_method: str = "deer", *,
                 key: jax.random.PRNGKey):
        if cell_type == "gru":
            self.cell = eqx.nn.GRUCell(input_size=ninputs, hidden_size=nhiddens, key=key)
        elif cell_type == "lstm":
            if eval_method == "deer":
                self.cell = LSTMWrapper(input_size=ninputs, hidden_size=nhiddens, key=key)
            else:
                self.cell = eqx.nn.LSTMCell(input_size=ninputs, hidden_size=nhiddens // 2, key=key)
        else:
            raise ValueError(f"Unknown cell: {cell_type}")
        self.cell_type = cell_type
        self.eval_method = eval_method

    def __call__(self, xs: jnp.ndarray, yinit_guess: Optional[jnp.ndarray] = None) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        # xs: (nsamples, ninputs)
        # outputs: (nsamples, nhiddens)

        # initialize the states
        if self.cell_type == "gru":
            init_state = jnp.zeros((self.cell.hidden_size,))
        elif self.cell_type == "lstm":
            if self.eval_method == "deer":
                init_state = jnp.zeros((self.cell.hidden_size,))
            else:
                init_state = (jnp.zeros((self.cell.hidden_size,)),) * 2
        else:
            raise RuntimeError(f"Should not be here")

        if self.eval_method == "deer":
            cell_fun = lambda state, xinput, cell: clscall(cell)(cell, xinput, state)
            outputs = seq1d(cell_fun, init_state, xs, self.cell, yinit_guess=yinit_guess)
        elif self.eval_method == "sequential":
            scan_fun = lambda state, xinput: (self.cell(xinput, state),) * 2
            final_state, outputs = jax.lax.scan(scan_fun, init_state, xs)
            if self.cell_type == "lstm":
                # outputs is a tuple of (hidden_state, cell_state), we need to concatenate it
                # to make it equivalent with deer's outputs
                outputs = jnp.concatenate(outputs, axis=-1)
        else:
            raise ValueError(f"Unknown eval_method: {self.eval_method}")
        yinit_guess = outputs
        return outputs, yinit_guess

class RNNClassifier(eqx.Module):
    cells: List[CellLayer]
    mlp: eqx.nn.MLP
    embedding: eqx.Module
    norms: List[eqx.Module]

    def __init__(self, ninputs: int, nhiddens: int, nlayers: int, noutputs: int, cell_type: str = "gru",
                 eval_method: str = "deer", nquantization: Optional[int] = None, *, key: jax.random.PRNGKey):
        cells = []
        key, *subkey = jax.random.split(key, nlayers + 3)
        for i in range(nlayers):
            ninput = ninputs if i == 0 else nhiddens
            cell = CellLayer(ninput, nhiddens, cell_type=cell_type, eval_method=eval_method, key=subkey[i])
            cells.append(cell)
        self.cells = cells
        self.mlp = eqx.nn.MLP(nhiddens, noutputs, nhiddens, depth=1, key=subkey[-1])
        if nquantization is not None:
            self.embedding = eqx.nn.Sequential([
                eqx.nn.Embedding(nquantization, ninputs, key=subkey[-2]),
                lambda x: x[:, 0, :]
            ])
        else:
            self.embedding = eqx.nn.Identity()
        self.norms = [eqx.nn.LayerNorm((nhiddens,), use_weight=False, use_bias=False) for i in range(nlayers)]

    def __call__(self, xs: jnp.ndarray, yinit_guess: Optional[List[jnp.ndarray]] = None) \
            -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
        # xs: (nsamples, ninputs)
        # outputs: (noutputs,)
        if yinit_guess is None:
            yinit_guess = [None] * len(self.cells)

        xs = self.embedding(xs)  # (nsamples, ninputs)
        new_yinit_guess = []
        for i, cell in enumerate(self.cells):
            xs_new, yinit_g = cell(xs, yinit_guess=yinit_guess[i])
            # if i != 0:
            #     xs = xs + xs_new
            # else:
            #     xs = xs_new
            xs = xs_new
            xs = jax.vmap(self.norms[i])(xs)
            # xs = xs[::2]
            new_yinit_guess.append(yinit_g)
        # xs: (nsamples, nhiddens)
        xlast = jnp.mean(xs, axis=0)  # (nhiddens,)
        # xlast = xs[-1]  # (nhiddens,)
        xs = self.mlp(xlast)  # (noutputs,)
        return xs, new_yinit_guess  # [nlayers] + (nsamples, nhiddens)


if __name__ == "__main__":
    from typing import Any, Tuple
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


# jax.config.update("jax_platform_name", "cpu")

def loss_fn(params, static, xs: jnp.ndarray, targets: jnp.ndarray, yinit_guess: Any = None) \
        -> Tuple[jnp.ndarray, Any]:
    # xs: (batch_size, nsamples, ninputs)
    # targets: (batch_size,) int
    model = eqx.combine(params, static)
    yinit_guess_axis = None if yinit_guess is None else 0
    # ys: (batch_size, noutputs), new_yinit_guess: [nlayers] + (batch_size, nsamples, nhiddens)
    ys, new_yinit_guess = jax.vmap(model, in_axes=(0, yinit_guess_axis))(xs, yinit_guess)
    # loss = jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(ys, targets)  # (batch_size,)
    metrics = compute_metrics(ys, targets)
    loss, accuracy = metrics["loss"], metrics["accuracy"]
    # jax.debug.print("ys: {ys}", ys=jnp.argmax(ys, axis=-1))
    # jax.debug.print("targets: {targets}", targets=targets)
    # jax.debug.print("loss: {loss}", loss=loss)
    # jax.debug.print("correct: {correct}", correct=jnp.equal(jnp.argmax(ys, axis=-1), targets))
    # jax.debug.print("accuracy: {accuracy}", accuracy=jnp.mean(jnp.equal(jnp.argmax(ys, axis=-1), targets)))
    return jnp.mean(loss), (accuracy, new_yinit_guess)

@partial(jax.jit, static_argnames=("static", "yinit_guess"))
def calc_accuracy(params, static, xs: jnp.ndarray, targets: jnp.ndarray, yinit_guess: Any = None) -> jnp.ndarray:
    yinit_gaxis = None if yinit_guess is None else 0
    # preds: (batch_size, noutputs), new_yinit_guess: [nlayers] + (batch_size, nsamples, nhiddens)
    model = eqx.combine(params, static)
    preds, new_yinit_guess = jax.vmap(model, in_axes=(0, yinit_gaxis))(xs, yinit_guess)
    idx_preds = jnp.argmax(preds, axis=-1)  # (batch_size,)
    # jax.debug.print("idx_preds: {idx_preds}, targets: {targets}", idx_preds=idx_preds, targets=targets)
    correct = jnp.equal(idx_preds, targets)  # (batch_size,)
    # jax.debug.print("correct: {correct}", correct=correct)
    accuracy = jnp.mean(correct)  # ()
    # jax.debug.print("accuracy: {accuracy}", accuracy=accuracy)
    return accuracy

@partial(jax.jit, static_argnames=("static", "optimizer", "yinit_guess"))
def update_step(params, static, optimizer: optax.GradientTransformation, opt_state: Any,
                xs: jnp.ndarray, targets: jnp.ndarray,
                yinit_guess: Any = None) -> Tuple[RNNClassifier, Any, jnp.ndarray, Any]:
    # xs: (batch_size, nsamples, ninputs)
    # targets: (batch_size, noutputs)
    (loss, (accuracy, new_yinit_guess)), grad = \
        jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params, static, xs, targets, yinit_guess)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, new_yinit_guess, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--model", type=str, default="gruseq")
    parser.add_argument("--dset", type=str, default="urbansound")
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

    # # get the dataset
    # dset = get_dataset(args.dataset)

    # # split the dataset into train, val, test
    # ntrain, nval, ntest = dset.splits()
    # train_dset = torch.utils.data.Subset(dset, range(ntrain))
    # val_dset = torch.utils.data.Subset(dset, range(ntrain, ntrain + nval))

    # # get the dataloader
    # train_dloader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
    # val_dloader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=True)

    # get the number of inputs and outputs from the dataset
    # ninputs = dset.ninputs if dset.nquantization is None else args.width
    # nclasses = dset.nclasses

    dm = get_datamodule(dset=args.dset, batch_size=args.batch_size)
    dm.setup()
    ninputs = 3
    nclasses = 10

    # get the model
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    if args.model == "grudeer":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="gru",
            eval_method="deer", nquantization=None, key=subkey)
    elif args.model == "gruseq":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="gru",
            eval_method="sequential", nquantization=None, key=subkey)
    elif args.model == "lstmdeer":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="lstm",
            eval_method="deer", nquantization=None, key=subkey)
    elif args.model == "lstmseq":
        model = RNNClassifier(
            ninputs=ninputs, nhiddens=args.width, nlayers=args.depth, noutputs=nclasses, cell_type="lstm",
            eval_method="sequential", nquantization=None, key=subkey)
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

    # get the summary writer
    summary_writer = SummaryWriter(log_dir=path)

    step = 0
    for epoch in tqdm(range(args.max_epochs), file=sys.stderr):
        loop = tqdm(dm.train_dataloader(), total=len(dm.train_dataloader()), leave=False, file=sys.stderr)
        # training phase
        for i, batch in enumerate(loop):
            try:
                batch = dm.on_before_batch_transfer(batch, i)
            except Exception():
                pass
            batch = prep_batch(batch, dtype=jnp.float32)
            xs, targets = batch
            # update step
            params, opt_state, loss, yinit_guess, accuracy = update_step(params, static, optimizer, opt_state, xs, targets)
            step += 1
            summary_writer.add_scalar("train_loss", loss, step)
            summary_writer.add_scalar("train_accuracy", accuracy, step)

        # # validation phase
        # val_acc = 0.0
        # for batch in tqdm(val_dloader, leave=False):
        #     xs, targets = batch
        #     # convert to jax
        #     xs = jnp.array(xs.numpy())
        #     targets = jnp.array(targets.numpy())
        #     # calculate the accuracy
        #     accuracy = calc_accuracy(params, static, xs, targets)
        #     val_acc += accuracy * len(targets)
        # val_acc /= len(val_dset) * 1.0

        # summary_writer.add_scalar("val_accuracy", val_acc, step)
        # summary_writer.flush()

if __name__ == "__main__":
    main()