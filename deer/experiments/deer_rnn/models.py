import functools
from typing import Any, List, Tuple, Callable, Sequence, Optional

import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, xavier_uniform
import equinox as eqx
from flax import linen as nn
from jax._src import prng

from deer.seq1d import seq1d

import pdb


def he_uniform(key, shape, dtype):
    fan_in = shape[-1]
    bound = jnp.sqrt(6 / fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _uniform(key, shape, dtype):
    fan_in = shape[-1]
    bound = jnp.sqrt(1 / fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class TmpScaleGRU(nn.Module):
    nhidden: int
    dtype: Any
    scale: float

    def scaled_initializers(self, scale: float):
        def init_fn(key, shape):
            return jnp.log(jnp.ones(shape) * scale)
        return init_fn

    def setup(self):
        self.gru = nn.GRUCell(
            features=self.nhidden,
            dtype=self.dtype,
            param_dtype=self.dtype,
            # kernel_init=_uniform,
            # recurrent_kernel_init=_uniform,
        )
        self.log_s = self.param("log_s", self.scaled_initializers(self.scale), ())

    def initialize_carry(self, batch_size):
        return jnp.ones((batch_size, self.nhidden))

    @nn.compact
    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)
        s = jnp.exp(self.log_s)
        states, _ = self.gru(h0, inputs / s)
        states = (states - h0) / s + h0
        return states, states


class MLP1(nn.Module):
    nstates: int
    nout: int
    dtype: Any

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.nstates, dtype=self.dtype, kernel_init=he_uniform)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nout, dtype=self.dtype, kernel_init=he_uniform)(x)
        # x = nn.Dense(self.nstates, dtype=self.dtype, kernel_init=_uniform)(x)
        # x = nn.tanh(x)
        # x = nn.Dense(self.nout, dtype=self.dtype, kernel_init=_uniform)(x)
        return x


def vmap_to_shape(func: Callable, shape: Sequence[int]):
    rank = len(shape)
    for i in range(rank - 1):
        func = jax.vmap(func)
    return func


def custom_mlp(mlp: eqx.nn.MLP, key: prng.PRNGKeyArray, init_method: Optional[str] = "he_uniform") -> eqx.nn.MLP:
    """
    eqx.nn.MLP with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda m: [lin.bias for lin in m.layers]
    where_weight = lambda m: [lin.weight for lin in m.layers]

    mlp = eqx.tree_at(where=where_bias, pytree=mlp, replace_fn=jnp.zeros_like)

    if init_method is None:
        return mlp

    if init_method == "he_uniform":
        # get all the weights of the mlp model
        weights = where_weight(mlp)
        # split the random key into different subkeys for each layer
        subkeys = jax.random.split(key, len(weights))
        new_weights = [
            jax.nn.initializers.he_uniform()(subkey, weight.shape) for weight, subkey in zip(weights, subkeys)
        ]
        mlp = eqx.tree_at(where=where_weight, pytree=mlp, replace=new_weights)
    else:
        return NotImplementedError("only he_uniform is implemented")
    return mlp


def custom_gru(gru: eqx.nn.GRUCell, key: prng.PRNGKeyArray) -> eqx.nn.GRUCell:
    """
    eqx.nn.GRUCell with custom initialisation scheme using jax.nn.initializers
    """
    where_bias = lambda g: g.bias
    where_bias_n = lambda g: g.bias_n
    where_weight_ih = lambda g: g.weight_ih
    where_weight_hh = lambda g: g.weight_hh

    gru = eqx.tree_at(where=where_bias, pytree=gru, replace_fn=jnp.zeros_like)
    gru = eqx.tree_at(where=where_bias_n, pytree=gru, replace_fn=jnp.zeros_like)

    weight_ih = where_weight_ih(gru)
    weight_hh = where_weight_hh(gru)

    ih_key, hh_key = jax.random.split(key, 2)

    new_weight_ih = jax.nn.initializers.lecun_normal()(ih_key, weight_ih.shape)
    new_weight_hh = jax.nn.initializers.orthogonal()(hh_key, weight_hh.shape)

    gru = eqx.tree_at(where_weight_ih, gru, new_weight_ih)
    gru = eqx.tree_at(where_weight_hh, gru, new_weight_hh)
    return gru


class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, ninp: int, nstate: int, nout: int, key: prng.PRNGKeyArray):
        self.model = eqx.nn.MLP(
            in_size=ninp,
            out_size=nout,
            width_size=nstate,
            depth=1,
            activation=jax.nn.tanh,
            # final_activation=jax.nn.tanh,  # adding --> even smaller gradient
            key=key
        )
        self.model = custom_mlp(self.model, key)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return vmap_to_shape(self.model, x.shape)(x)


class ScaleGRU(eqx.Module):
    # check if everything defined here is meant to be trainable
    gru: eqx.Module
    log_s: jnp.ndarray  # check if this is trainable

    def __init__(self, ninp: int, nstate: int, scale: float, key: prng.PRNGKeyArray):
        self.gru = eqx.nn.GRUCell(
            input_size=ninp,
            hidden_size=nstate,
            key=key
        )
        self.gru = custom_gru(self.gru, key)
        self.log_s = jnp.log(jnp.ones((1,)) * scale)

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # h0.shape == (nbatch, nstate)
        # inputs.shape == (nbatch, ninp)
        assert len(inputs.shape) == len(h0.shape)

        s = jnp.exp(self.log_s)
        # jax.debug.print("{s}", s=s)
        states = vmap_to_shape(self.gru, inputs.shape)(inputs / s, h0)
        states = (states - h0) / s + h0
        return states


class MultiScaleGRU(eqx.Module):
    nchannel: int
    nlayer: int
    encoder: MLP
    scale_grus: List[List[ScaleGRU]]
    mlps: List[MLP]
    classifier: MLP
    norms: List[eqx.nn.LayerNorm]
    # dropout: eqx.nn.Dropout
    # dropout_key: prng.PRNGKeyArray

    def __init__(self, ninp: int, nchannel: int, nstate: int, nlayer: int, nclass: int, key: prng.PRNGKeyArray):
        keycount = 1 + (nchannel + 1) * nlayer + 1
        print(f"Keycount: {keycount}")
        keys = jax.random.split(key, keycount)

        self.nchannel = nchannel
        self.nlayer = nlayer

        assert nstate % nchannel == 0
        gru_nstate = int(nstate / nchannel)

        # encode inputs (or rather, project) to have nstates in the feature dimension
        self.encoder = MLP(ninp=ninp, nstate=nstate, nout=nstate, key=keys[0])

        # nlayers of (scale_gru + mlp) pair
        self.scale_grus = [[
            ScaleGRU(
                ninp=nstate,
                nstate=gru_nstate,
                scale=10 ** i,
                key=keys[int(1 + (nchannel * j) + i)]
            ) for i in range(nchannel)] for j in range(nlayer)
        ]
        self.mlps = [
            MLP(ninp=nstate, nstate=nstate, nout=nstate, key=keys[int(i + 1 + nchannel * nlayer)]) for i in range(nlayer)
        ]
        assert len(self.scale_grus) == nlayer
        assert len(self.scale_grus[0]) == nchannel
        assert len(self.mlps) == nlayer
        print(f"scale_grus random keys end at index {int(1 + (nchannel * (nlayer - 1)) + (nchannel - 1))}")
        print(f"mlps random keys end at index {int((nchannel * nlayer) + nlayer)}")

        # project nstates in the feature dimension to nclasses for classification
        self.classifier = MLP(ninp=nstate, nstate=nstate, nout=nclass, key=keys[int((nchannel + 1) * nlayer + 1)])

        self.norms = [eqx.nn.LayerNorm((nstate,), use_weight=False, use_bias=False) for i in range(nlayer * 2)]
        # self.dropout = eqx.nn.Dropout(p=0.2)
        # self.dropout_key = jax.random.PRNGKey(42)

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray, yinit_guess: jnp.ndarray):
        # encode (or rather, project) the inputs
        inputs = self.encoder(inputs)

        def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
            return model(inputs, carry)

        x_from_all_layers = []
        # TODO there should be a way to vmap the channel
        for i in range(self.nlayer):
            inputs = self.norms[i](inputs)

            x_from_all_channels = []
            for ch in range(self.nchannel):

                x = seq1d(
                    model_func,
                    h0[i][ch],
                    inputs,
                    self.scale_grus[i][ch],
                    yinit_guess[i][ch]
                )
                # think vmap should be removed later??
                # x = jax.vmap(seq1d, in_axes=(None, 0, 0, None, 0))(
                #     model_func,
                #     h0[i][ch],
                #     inputs,
                #     None,
                #     yinit_guess[i][ch]
                # )
                x_from_all_channels.append(x)
            x_from_all_layers.append(jnp.stack(x_from_all_channels))
            x = jnp.concatenate(x_from_all_channels, axis=-1)
            x = self.norms[i + 1](x + inputs)  # add and norm after multichannel GRU layer
            # x = self.dropout(x, key=jax.random.PRNGKey(42))
            x = self.mlps[i](x) + x  # add with norm added in the next loop
            inputs = x
        yinit_guess = jnp.stack(x_from_all_layers)
        return self.classifier(x), yinit_guess


if __name__ == "__main__":
    # this only works with jax.vmap(seq1d) in MultiScaleGRU
    ninp = 1
    nstate = 32
    nchannel = 4
    nclass = 2
    nlayer = 3
    # model = MultiScaleGRU(
    #     ninp=ninp,
    #     nchannel=nchannel,
    #     nstate=nstate,
    #     nlayer=nlayer,
    #     nclass=nclass,
    #     key=jax.random.PRNGKey(1)
    # )
    # mlp = MLP(
    #     ninp=ninp,
    #     nstate=nstate,
    #     nout=nstate,
    #     key=jax.random.PRNGKey(1)
    # )
    # scale_gru = ScaleGRU(
    #     ninp=nstate,
    #     nstate=nstate,
    #     scale=1,
    #     key=jax.random.PRNGKey(1)
    # )

    nseq = 69
    batch_size = 7
    # carry = jnp.zeros(
    #     (batch_size, int(nstate / nchannel))
    # )  # (batch_size, nstates)
    # inputs = jax.random.normal(
    #     jax.random.PRNGKey(1),
    #     (batch_size, nseq, ninp),
    # )  # (batch_size, nsequence, nstates)
    # yinit_guess = jax.random.normal(
    #     jax.random.PRNGKey(1),
    #     (batch_size, nseq, int(nstate / nchannel)),
    # )  # (batch_size, nsequence, nstates)
    # y = model(
    #     inputs,
    #     [[carry for _ in range(nchannel)] for _ in range(nlayer)],
    #     [[yinit_guess for _ in range(nchannel)] for _ in range(nlayer)]
    # )

    inputs = jnp.ones((batch_size, nseq, ninp))
    mlp = MLP(
        ninp=ninp,
        nstate=nstate,
        nout=nstate,
        key=jax.random.PRNGKey(1)
    )
    scale_gru = ScaleGRU(
        ninp=nstate,
        nstate=nstate,
        scale=1,
        key=jax.random.PRNGKey(1)
    )
