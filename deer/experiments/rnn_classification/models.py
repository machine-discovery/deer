from typing import Callable, Optional, List, Tuple, Dict, Any
import math
from functools import partial
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import numpy as np
from deer.seq1d import seq1d


BATCH_AXIS_NAME = "batch"

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

    def __call__(self, xs: jnp.ndarray, model_states: Dict) \
            -> Tuple[jnp.ndarray, Dict]:
        # xs: (batch_size, nsamples, ninputs)
        # outputs: (batch_size, nsamples, nhiddens)

        # get the initial guess
        yinit_guess = None

        # initialize the states
        batch_size = xs.shape[0]
        if self.cell_type == "gru":
            init_state = jnp.zeros((batch_size, self.cell.hidden_size))
        elif self.cell_type == "lstm":
            if self.eval_method == "deer":
                init_state = jnp.zeros((batch_size, self.cell.hidden_size))
            else:
                init_state = (jnp.zeros((batch_size, self.cell.hidden_size)),) * 2
        else:
            raise RuntimeError(f"Should not be here")

        if self.eval_method == "deer":
            cell_fun = lambda state, xinput, cell: clscall(cell)(cell, xinput, state)
            # outputs: (batch_size, nsamples, noutputs)
            outputs = jax.vmap(seq1d, in_axes=(None, 0, 0, None, 0))(cell_fun, init_state, xs, self.cell, yinit_guess)
        elif self.eval_method == "sequential":
            scan_fun = lambda state, xinput: (jax.vmap(self.cell)(xinput, state),) * 2
            final_state, outputs = jax.lax.scan(scan_fun, init_state, xs)
            if self.cell_type == "lstm":
                # outputs is a tuple of (hidden_state, cell_state), we need to concatenate it
                # to make it equivalent with deer's outputs
                outputs = jnp.concatenate(outputs, axis=-1)
        else:
            raise ValueError(f"Unknown eval_method: {self.eval_method}")
        return outputs, model_states

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
        self.norms = [eqx.nn.LayerNorm((nhiddens,)) for i in range(nlayers)]

    def __call__(self, xs: jnp.ndarray, model_states: Dict) \
            -> Tuple[jnp.ndarray, Dict]:
        # xs: (batch_size, nsamples, ninputs)
        # outputs: (batch_size, noutputs)

        xs = jax.vmap(self.embedding)(xs)  # (batch_size, nsamples, ninputs)
        for i, cell in enumerate(self.cells):
            xs_new, model_states = cell(xs, model_states)
            # if i != 0:
            #     xs = xs + xs_new
            # else:
            #     xs = xs_new
            xs = xs_new
            xs = jax.vmap(self.norms[i])(xs)
            # xs = xs[::2]
        # xs: (nsamples, nhiddens)
        xlast = jnp.mean(xs, axis=0)  # (nhiddens,)
        # xlast = xs[-1]  # (nhiddens,)
        xs = self.mlp(xlast)  # (noutputs,)
        return xs, model_states  # [nlayers] + (nsamples, nhiddens)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 4)
    nfeatures = 32
    ninputs = 16
    nsamples = 100
    cell1 = RNNModule(ninputs, nfeatures, 3, nfeatures, eval_method="sequential", key=subkey[0])
    cell2 = RNNModule(ninputs, nfeatures, 3, nfeatures, eval_method="deer", key=subkey[0])

    xs = jax.random.normal(subkey[1], (nsamples, ninputs))
    y1, y1init_guess = cell1(xs)
    y2 = cell2(xs, y1init_guess)[0]
    print(y1 - y2)
    print(y1)
    print(y1.shape)
