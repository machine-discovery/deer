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
    state_index: eqx.nn.StateIndex
    initg_static: Any
    initg_optimizer: optax.GradientTransformation

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

        key, *subkey = jax.random.split(key, 2)

        # initialize the optimizer for initg_model
        self.initg_optimizer = optax.chain(
            optax.adam(learning_rate=1e-3),
        )

        def _create_initg_model():
            return InitGModel(ninputs, nhiddens, key=subkey[0])

        def _create_states(**_):
            initg_params, _ = eqx.partition(_create_initg_model(), eqx.is_array)
            opt_state = self.initg_optimizer.init(initg_params)
            return initg_params, opt_state

        self.cell_type = cell_type
        self.eval_method = eval_method
        self.state_index = eqx.nn.StateIndex(_create_states)
        _, self.initg_static = eqx.partition(_create_initg_model(), eqx.is_array)

    def __call__(self, xs: jnp.ndarray, model_states: Dict) \
            -> Tuple[jnp.ndarray, Dict]:
        # xs: (batch_size, nsamples, ninputs)
        # outputs: (batch_size, nsamples, nhiddens)

        # get the initial guess
        yinit_guess = None
        initg_params, initg_opt_state = model_states.get(self.state_index)
        initg_model = eqx.combine(initg_params, self.initg_static)
        yinit_guess = jax.vmap(initg_model)(xs)  # (batch_size, nsamples, nhiddens)

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

        # update the states by applying one update states for the initg_model
        initg_params, initg_opt_state = \
            initg_update_step(initg_params, self.initg_static, xs, outputs, self.initg_optimizer, initg_opt_state)
        model_states = model_states.set(self.state_index, (initg_params, initg_opt_state))
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
            xs = jax.vmap(jax.vmap(self.norms[i]))(xs)
            # xs = xs[::2]
        # xs: (batch_size, nsamples, nhiddens)
        xlast = jnp.mean(xs, axis=-2)  # (batch_size, nhiddens)
        xs = jax.vmap(self.mlp)(xlast)  # (batch_size, noutputs)
        return xs, model_states  # [nlayers] + (batch_size, nsamples, nhiddens)


class InitGModel(eqx.Module):
    model: eqx.Module

    def __init__(self, ninputs: int, noutputs: int, nhiddens: int = 64, *, key: jax.random.PRNGKey):
        subkey = jax.random.split(key, 5)
        self.model = eqx.nn.Sequential([
            VMapped(eqx.nn.MLP(ninputs, nhiddens, nhiddens, depth=1, key=subkey[0])),
            S4D(nhiddens, key=subkey[1]),
            VMapped(eqx.nn.MLP(nhiddens, nhiddens, nhiddens, depth=1, key=subkey[2])),
            S4D(nhiddens, key=subkey[3]),
            VMapped(eqx.nn.MLP(nhiddens, noutputs, nhiddens, depth=1, key=subkey[4])),
        ])

    def __call__(self, xs: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # xs: (nsamples, ninputs)
        # outputs: (nsamples, noutputs)
        return self.model(xs)

class VMapped(eqx.Module):
    model: eqx.Module
    n: int

    def __init__(self, model: eqx.Module, n: int = 1):
        self.model = model
        self.n = n

    def __call__(self, xs: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        model = self.model
        for i in range(self.n):
            model = jax.vmap(model)
        y = model(xs, key=key)
        return y

class S4D(eqx.Module):
    log_dt: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray
    log_A_real: jnp.ndarray
    A_imag: jnp.ndarray

    def __init__(self, d_model: int, N: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1, *,
                 key: jax.random.PRNGKey):
        subkey = jax.random.split(key, 3)
        H = d_model
        self.log_dt = jax.random.uniform(subkey[0], (H,)) * (
            math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        cdtype = jnp.complex64 if self.log_dt == jnp.float32 else jnp.complex128
        self.C = jax.random.normal(subkey[1], (H, N // 2), dtype=cdtype)  # (H, N // 2)

        # obtain the A matrix
        self.log_A_real = jnp.log(0.5 * jnp.ones((H, N // 2)))
        self.A_imag = math.pi * jnp.repeat(jnp.reshape(jnp.arange(N // 2), (1, -1)), H, axis=0)

        # get the D matrix for the skip connection
        self.D = jax.random.normal(subkey[2], (H, 1))  # (H, 1)

    def __call__(self, xs: jnp.ndarray, *, key: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
        # xs: (nsamples, H)
        # outputs: (nsamples, H)
        L = xs.shape[-2]

        # obtain the fft kernel
        # first, materialize the parameters
        dt = jnp.exp(self.log_dt)  # (H,)
        A = -jnp.exp(self.log_A_real) + 1j * self.A_imag  # (H, N // 2)

        # vandermonde multiplication
        dtA = A * dt[..., None]  # (H, N // 2)
        K = dtA[..., None] * jnp.arange(L)  # (H, N // 2, nsamples)
        C = self.C * (jnp.exp(dtA) - 1.0) / A  # (H, N // 2)
        kernel = 2 * jnp.einsum("...hn, ...hnl -> ...hl", C, jnp.exp(K)).real  # (H, nsamples)

        # convolution with the kernel
        xs2 = jnp.swapaxes(xs, -2, -1)  # (H, nsamples)
        k_f = jnp.fft.rfft(kernel, n=2 * L)
        u_f = jnp.fft.rfft(xs2, n=2 * L)
        y = jnp.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (H, nsamples)

        # skip connection
        y = y + xs2 * self.D  # (H, nsamples)
        y = jnp.swapaxes(y, -2, -1)  # (nsamples, H)
        return y

def initg_update_step(initg_params, initg_static, xs, outputs, initg_optimizer, initg_opt_state):
    grad = jax.grad(initg_loss)(initg_params, initg_static, xs, outputs)
    updates, initg_opt_state = initg_optimizer.update(grad, initg_opt_state, initg_params)
    initg_params = optax.apply_updates(initg_params, updates)
    initg_params = jax.lax.stop_gradient(initg_params)
    initg_opt_state = jax.lax.stop_gradient(initg_opt_state)
    return initg_params, initg_opt_state

def initg_loss(initg_params, initg_static, xs, outputs):
    # xs: (batch_size, nsamples, ninputs)
    # outputs: ()
    initg_model = eqx.combine(initg_params, initg_static)
    preds = jax.vmap(initg_model)(xs)
    loss = jnp.mean((preds - outputs) ** 2)  # ()
    return loss

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
