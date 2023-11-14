from typing import List, Tuple, Callable, Optional
import argparse
import time
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
from deer.seq1d import seq1d


def eval_rnn_seq(carry: jnp.ndarray, inputs: jnp.ndarray, rnn_params, rnn_static) \
        -> jnp.ndarray:
    # carry: (hidden_size,)
    # inputs: (length, input_size)
    # returns: (length, hidden_size)
    rnn = eqx.combine(rnn_params, rnn_static)

    def call_rnn1(carry: jnp.ndarray, inputs: jnp.ndarray):
        output = rnn(inputs, carry)
        return output, output

    _, outputs = jax.lax.scan(call_rnn1, carry, inputs)
    return outputs

def eval_rnn_deer(carry: jnp.ndarray, inputs: jnp.ndarray, rnn_params, rnn_static) \
        -> jnp.ndarray:
    # carry: (hidden_size,)
    # inputs: (length, input_size)
    # returns: (length, hidden_size)
    def call_gru2(carry: jnp.ndarray, inputs: jnp.ndarray, params):
        gru = eqx.combine(params, rnn_static)
        return gru(inputs, carry)

    # seq1dm = jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)
    outputs = seq1d(call_gru2, carry, inputs, rnn_params)
    return outputs

class DualStatesWrapper(eqx.Module):
    # wrapper for dual-states modules to make its states and outputs as one tensor, so the interface is the same as GRU
    module: eqx.Module

    def __init__(self, module: eqx.Module):
        super().__init__()
        self.module = module

    def __call__(self, input: jnp.ndarray, carry: jnp.ndarray) -> jnp.ndarray:
        carry1, carry2 = jnp.split(carry, indices_or_sections=2, axis=-1)
        out1, out2 = self.module(input, (carry1, carry2))
        return jnp.concatenate((out1, out2), axis=-1)

############# cells #############
class RNN(eqx.Module):
    rnn: eqx.Module
    hidden_size: int
    method: str

    def __init__(self, input_size: int, hidden_size: int, use_bias: bool = True, method: str = "deer",
                 rnn_type: str = "gru", *,
                 key: jax.random.PRNGKeyArray, **kwargs):
        self.hidden_size = hidden_size
        if rnn_type == "gru":
            self.rnn = eqx.nn.GRUCell(input_size, hidden_size, use_bias, key=key, **kwargs)
        elif rnn_type == "lstm":
            assert hidden_size % 2 == 0
            rnn = eqx.nn.LSTMCell(input_size, hidden_size // 2, use_bias, key=key, **kwargs)
            self.rnn = DualStatesWrapper(rnn)
        else:
            raise ValueError(f"Unknown rnn_type: '{rnn_type}'. Must be 'gru' or 'lstm'.")
        self.method = method

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: (length, input_size)
        # carry: (hidden_size)
        carry = jnp.zeros((self.hidden_size,), dtype=inputs.dtype)
        params, static = eqx.partition(self.rnn, eqx.is_inexact_array)
        # outputs: (batch_size, length, hidden_size)
        if self.method == "sequential":
            outputs = eval_rnn_seq(carry, inputs, params, static)
        elif self.method == "deer":
            outputs = eval_rnn_deer(carry, inputs, params, static)
        return outputs

class MultiScaleRNN(eqx.Module):
    rnns: List[eqx.Module]
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8, use_bias: bool = True,
                 method: str = "deer", rnn_type: str = "gru", *, 
                 key: jax.random.PRNGKeyArray, **kwargs):
        key, *subkey = jax.random.split(key, num_heads + 1)
        self.hidden_size = hidden_size
        self.rnns = [RNN(input_size, hidden_size, use_bias, method, rnn_type, key=subkey[i], **kwargs)
                     for i in range(num_heads)]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (length, input_size)
        # returns: (length, hidden_size)
        # def scan_func(x: jnp.ndarray, _: None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #     # x: (2 ** irnn, length // 2 ** irnn, input_size)
        #     # output: (length, hidden_size)
        #     output = jax.vmap(self.rnns[irnns])(x).reshape(-1, self.hidden_size)
        #     x = x.reshape(x.shape[0] * 2, -1, x.shape[-1])
        #     return x, output
        # irnns = jnp.arange(len(self.rnns), dtype=jnp.int64)
        # x = x.reshape(1, -1, x.shape[-1])
        # _, outputs = jax.lax.scan(scan_func, x, irnns)  # (num_heads, length, hidden_size)
        # # (length, num_heads, hidden_size), then (length, num_heads * hidden_size)
        # return jnp.moveaxis(outputs, 0, -2).reshape(outputs.shape[-2], -1)

        xshape = x.shape
        outputs = []
        for i, rnn in enumerate(self.rnns):
            xx = x.reshape(2 ** i, -1, xshape[-1])
            outputs.append(jax.vmap(rnn)(xx).reshape(-1, self.hidden_size))  # [num_heads] + (length, hidden_size)
        output = jnp.concatenate(outputs, axis=-1)  # (length, num_heads * hidden_size)
        return output

class Bidirectional(eqx.Module):
    rnn1: eqx.Module
    rnn2: eqx.Module

    def __init__(self, rnn1: eqx.Module, rnn2: eqx.Module):
        super().__init__()
        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # inputs: (length, input_size)
        # outputs: (length, hidden_size)
        outputs1 = self.rnn1(inputs)
        outputs2 = self.rnn2(inputs[::-1])[::-1]
        return (outputs1 + outputs2) * 0.5

class LinActLinGLU(eqx.Module):
    module: eqx.Module
    lin0: eqx.nn.Linear
    lin1: eqx.nn.Linear

    def __init__(self, module: eqx.Module, hidden_size: int, use_bias: bool = True, *,
                 key: jax.random.PRNGKeyArray):
        key, *subkey = jax.random.split(key, 3)
        self.module = module
        self.lin0 = eqx.nn.Linear(hidden_size, hidden_size, use_bias, key=subkey[0])
        self.lin1 = eqx.nn.Linear(hidden_size, 2 * hidden_size, use_bias, key=subkey[1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (length, input_size)
        # returns: (length, hidden_size)
        x = self.module(x)
        x = jax.nn.gelu(jax.vmap(self.lin0)(x))
        x = jax.nn.glu(jax.vmap(self.lin1)(x))
        return x

############# nets #############
class RNNNet(eqx.Module):
    embedding: Optional[eqx.Module]
    lin0: eqx.nn.Linear
    rnns: List[eqx.Module]
    ln0s: List[eqx.Module]
    ln1s: List[eqx.Module]
    mlps: List[eqx.Module]
    drop0s: List[eqx.Module]
    drop1s: List[eqx.Module]
    activation: Callable
    lin1: eqx.nn.Linear
    reduce_length: bool
    prenorm: bool = False
    final_mlp: Optional[eqx.Module] = None

    def __init__(self, num_inps: int, num_outs: int, with_embedding: bool, reduce_length: bool,
                 num_heads: int = 1, nhiddens: int = 64, nlayers: int = 8, nhiddens_mlp: int = 64,
                 method: str = "deer", rnn_type: str = "gru",
                 p_dropout: float = 0.0,
                 bidirectional: bool = False, bidirasymm: bool = False, rnn_wrapper: Optional[int] = None,
                 prenorm: bool = False, final_mlp: bool = False, *,
                 key: jax.random.PRNGKeyArray):

        # get the embedding
        if with_embedding:
            key, subkey = jax.random.split(key, 2)
            self.embedding = eqx.nn.Embedding(num_inps, nhiddens, key=subkey)
            num_inps = nhiddens
        else:
            self.embedding = None

        self.reduce_length = reduce_length
        key, *subkey = jax.random.split(key, 4)
        self.lin0 = eqx.nn.Linear(num_inps, nhiddens, key=subkey[0])
        self.lin1 = eqx.nn.Linear(nhiddens, num_outs, key=subkey[1])
        self.activation = jax.nn.gelu
        self.prenorm = prenorm
        self.final_mlp = eqx.nn.MLP(nhiddens, nhiddens, nhiddens_mlp, depth=1, activation=self.activation,
                                    key=subkey[2]) if final_mlp else None

        self.rnns = []
        self.ln0s = []
        self.ln1s = []
        self.mlps = []
        self.drop0s = []
        self.drop1s = []
        key, *subkey = jax.random.split(key, nlayers + 1)
        key, *subkey1 = jax.random.split(key, nlayers + 1)
        rnn_kwargs = {
            "input_size": nhiddens,
            "hidden_size": nhiddens,
            "method": method,
            "rnn_type": rnn_type,
        }
        rnn_class = MultiScaleRNN if num_heads > 1 else RNN
        if num_heads > 1:
            assert nhiddens % num_heads == 0
            rnn_kwargs["num_heads"] = num_heads
            rnn_kwargs["hidden_size"] = nhiddens // num_heads
        for i in range(nlayers):
            # get the rnn layer
            if bidirectional and not bidirasymm:
                # symmetric bidirectional
                rnn = rnn_class(**rnn_kwargs, key=subkey[i])
                rnn = Bidirectional(rnn, rnn)
            elif bidirectional and bidirasymm:
                # asymmetric bidirectional
                subk = jax.random.split(subkey[i], 2)
                rnn1 = rnn_class(**rnn_kwargs, key=subk[0])
                rnn2 = rnn_class(**rnn_kwargs, key=subk[1])
                rnn = Bidirectional(rnn1, rnn2)
            else:
                # unidirectional
                rnn = rnn_class(**rnn_kwargs, key=subkey[i])
            if rnn_wrapper == 1:
                key, subkey_wrapper = jax.random.split(key, 2)
                rnn = LinActLinGLU(rnn, nhiddens, key=subkey_wrapper)
            self.rnns.append(rnn)

            self.ln0s.append(eqx.nn.LayerNorm(nhiddens))
            self.mlps.append(eqx.nn.MLP(nhiddens, nhiddens, nhiddens_mlp, depth=1, activation=self.activation,
                                        final_activation=self.activation, key=subkey1[i]))
            self.ln1s.append(eqx.nn.LayerNorm(nhiddens))
            self.drop0s.append(eqx.nn.Dropout(p_dropout) if p_dropout > 0.0 else eqx.nn.Identity())
            self.drop1s.append(eqx.nn.Dropout(p_dropout) if p_dropout > 0.0 else eqx.nn.Identity())

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKeyArray, inference: bool) -> jnp.ndarray:
        # x: (length, num_inps) if not with_embedding else (length,) int
        # returns: (length, num_outs)
        if self.embedding is not None:
            x = jax.vmap(self.embedding)(x)

        x = jax.vmap(self.lin0)(x)

        key, *subkey = jax.random.split(key, len(self.rnns) * 2 + 1)
        for i in range(len(self.rnns)):
            rnn = self.rnns[i]
            mlp = jax.vmap(self.mlps[i])
            ln0 = jax.vmap(self.ln0s[i])
            ln1 = jax.vmap(self.ln1s[i])
            drop0 = self.drop0s[i]
            drop1 = self.drop1s[i]
            sublayer0 = lambda xx: drop0(self.activation(rnn(xx)), key=subkey[2 * i], inference=inference)
            sublayer1 = lambda xx: drop1(mlp(xx), key=subkey[2 * i + 1], inference=inference)
            if not self.prenorm:  # post-norm
                x = ln0(sublayer0(x) + x)
                # x = drop0(self.activation(rnn(x)), key=subkey[2 * i], inference=inference) + x
                # x = ln0(x)
                x = ln1(sublayer1(x) + x)
                # x = drop1(jax.vmap(mlp)(x), key=subkey[2 * i + 1], inference=inference) + x
                # x = ln1(x)
            else:
                x = sublayer0(ln0(x)) + x
                x = sublayer0(ln1(x)) + x
                # x = drop0(self.activation(rnn(jax.vmap(ln0)(x))), key=subkey[2 * i], inference=inference) + x
                # x = drop1(jax.vmap(mlp)(jax.vmap(ln1)(x)), key=subkey[2 * i + 1], inference=inference) + x

        x = jax.vmap(self.lin1)(x)  # (length, num_outs)
        if self.reduce_length:
            x = jnp.mean(x, axis=0)  # (num_outs,)
        if self.final_mlp is not None:
            x = self.activation(x)
            final_mlp = jax.vmap(self.final_mlp) if not self.reduce_length else self.final_mlp
            x = final_mlp(x)
        return x
