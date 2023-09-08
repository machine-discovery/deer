import functools
from typing import Any, List, Tuple, Callable, Sequence

import jax
import jax.numpy as jnp
# from jax.nn.initializers import glorot_uniform, xavier_uniform
import equinox as eqx
from flax import linen as nn
from jax._src import prng

from deer.seq1d import seq1d

import pdb


def he_uniform(key, shape, dtype):
    fan_in = shape[-1]
    bound = jnp.sqrt(6 / fan_in)
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
        )
        self.log_s = self.param("log_s", self.scaled_initializers(self.scale), ())

    def initialize_carry(self, batch_size):
        return jnp.ones((batch_size, self.nhidden))

    @nn.compact
    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)
        s = jnp.exp(self.log_s)
        # pdb.set_trace()
        states, _ = self.gru(h0, inputs / s)
        # pdb.set_trace()
        states = (states - h0) / s + h0
        # pdb.set_trace()
        return states, states


# class MLP(nn.Module):
#     nstates: int
#     nout: int
#     dtype: Any

#     @nn.compact
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         x = nn.Dense(self.nstates, dtype=self.dtype, kernel_init=he_uniform)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(self.nout, dtype=self.dtype, kernel_init=he_uniform)(x)
#         return x


# class ScaleGRU(nn.Module):
#     nstates: int
#     scale: float
#     dtype: Any

#     def scaled_initializers(self, scale: float):
#         def init_fn(key, shape):
#             return jnp.log(jnp.ones(shape) * scale)
#         return init_fn

#     def setup(self):
#         self.gru = nn.GRUCell(
#             features=self.nstates,
#             dtype=self.dtype,
#             param_dtype=self.dtype,
#         )
#         self.log_s = self.param("log_s", self.scaled_initializers(self.scale), ())

#     @nn.compact
#     def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):        
#         # h0.shape == (nbatch, nstates)
#         # inputs.shape == (nbatch, ninp)
#         s = jnp.exp(self.log_s)
#         states, _ = self.gru(h0, inputs / s)
#         states = (states - h0) / s + h0
#         return states, states


# def MultiScaleGRU(
#     nchannels: int,
#     nstates: int,
#     nlayers: int,
#     nclasses: int,
#     dtype: Any,
# ):
#     assert nstates % nchannels == 0
#     gru_nstates = int(nstates / nchannels)

#     # encode inputs (or rather, project) to have nstates in the feature dimension
#     encoder = MLP(nstates=nstates, nout=nstates, dtype=dtype)

#     # nlayers of (scale_gru + mlp) pair
#     scale_grus = [[
#         ScaleGRU(
#             nstates=gru_nstates,
#             scale=10 ** i,
#             dtype=dtype
#         ) for i in range(nchannels)] for _ in range(nlayers)
#     ]
#     mlps = [
#         MLP(nstates=nstates, nout=nstates, dtype=dtype) for i in range(nlayers)
#     ]
#     assert len(scale_grus) == nlayers
#     assert len(scale_grus[0]) == nchannels
#     assert len(mlps) == nlayers

#     # project nstates in the feature dimension to nclasses for classification
#     classifier = MLP(nstates=nstates, nout=nclasses, dtype=dtype)

#     return


# class MultiScaleGRU(nn.Module):
#     nchannels: int
#     nstates: int
#     nlayers: int
#     nclasses: int
#     dtype: Any

#     def initialize_carry(self, batch_size):
#         assert self.nstates % self.nchannels == 0
#         gru_nstates = int(self.nstates / self.nchannels)

#         return jnp.ones((batch_size, gru_nstates))

#     def setup(self):
#         assert self.nstates % self.nchannels == 0
#         gru_nstates = int(self.nstates / self.nchannels)

#         # encode inputs (or rather, project) to have nstates in the feature dimension
#         self.encoder = MLP(nstates=self.nstates, nout=self.nstates, dtype=self.dtype)

#         # nlayers of (scale_gru + mlp) pair
#         self.scale_grus = [[
#             ScaleGRU(
#                 nstates=gru_nstates,
#                 scale=10 ** i,
#                 dtype=self.dtype
#             ) for i in range(self.nchannels)] for _ in range(self.nlayers)
#         ]
#         self.mlps = [
#             MLP(nstates=self.nstates, nout=self.nstates, dtype=self.dtype) for i in range(self.nlayers)
#         ]
#         assert len(self.scale_grus) == self.nlayers
#         assert len(self.scale_grus[0]) == self.nchannels
#         assert len(self.mlps) == self.nlayers

#         # project nstates in the feature dimension to nclasses for classification
#         self.classifier = MLP(nstates=self.nstates, nout=self.nclasses, dtype=self.dtype)

#     def __call__(self, h0: List[jnp.ndarray], inputs: jnp.ndarray, yinit_guess: List[jnp.ndarray]):
#         # assert len(h0) == self.nchannels

#         # encode (or rather, project) the inputs
#         inputs = self.encoder(inputs)

#         for i in range(self.nlayers):
#             x_from_all_channels = []
#             for ch in range(self.nchannels):
#                 print(i, ch)
#                 print(inputs.shape)
#                 pdb.set_trace()

#                 # def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any):
#                 #     return self.scale_grus[i][ch](carry, inputs)
#                 def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any):
#                     return self.scale_grus[i][ch].apply({"params": params[f"scaled_grus_{i}"]}, carry, inputs)

#                 # x = seq1d(
#                 #     func=model_func,
#                 #     y0=h0,
#                 #     xinp=inputs,
#                 #     params=0,
#                 #     yinit_guess=yinit_guess  # yinit_guess[i]
#                 # )
#                 # x = jax.vmap(seq1d, in_axes=(None, 0, 0, None, 0))(
#                 #     model_func,
#                 #     h0,
#                 #     inputs,
#                 #     None,
#                 #     yinit_guess  # yinit_guess[i]
#                 # )
#                 x = nn.vmap(seq1d, in_axes=(0, 0, None, 0))(
#                     self.scale_grus[i][ch],
#                     h0,
#                     inputs,
#                     None,
#                     yinit_guess  # yinit_guess[i]
#                 )
#                 pdb.set_trace()
#                 x_from_all_channels.append(x)
#             x = jnp.concatenate(x_from_all_channels, axis=-1)
#             x = self.mlp[i](x)
#             inputs = x
#         return self.mlp[-1](x)


def vmap_to_shape(func: Callable, shape: Sequence[int]):
    rank = len(shape)
    for i in range(rank - 1):
        func = jax.vmap(func)
    return func


class MLP(eqx.Module):
    model: eqx.nn.MLP

    def __init__(self, ninp: int, nstate: int, nout: int, key: prng.PRNGKeyArray):
        self.model = eqx.nn.MLP(
            in_size=ninp,
            out_size=nout,
            width_size=nstate,
            depth=1,
            activation=jax.nn.tanh,
            key=key
        )

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
        self.log_s = jnp.log(jnp.ones(1) * scale)

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # h0.shape == (nbatch, nstate)
        # inputs.shape == (nbatch, ninp)
        assert len(inputs.shape) == len(h0.shape)

        s = jnp.exp(self.log_s)
        jax.debug.print("{s}", s=s)
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

    def __init__(self, ninp: int, nchannel: int, nstate: int, nlayer: int, nclass: int, key: prng.PRNGKeyArray):
        keycount = 1 + (nchannel + 1) * nlayer + 1
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

        # project nstates in the feature dimension to nclasses for classification
        self.classifier = MLP(ninp=nstate, nstate=nstate, nout=nclass, key=keys[int((nchannel + 1) * nlayer + 1)])

    def __call__(self, inputs: jnp.ndarray, h0: jnp.ndarray, yinit_guess: jnp.ndarray):
        # encode (or rather, project) the inputs
        inputs = self.encoder(inputs)

        x_from_all_layers = []
        # TODO there should be a way to vmap the channel
        for i in range(self.nlayer):
            x_from_all_channels = []
            for ch in range(self.nchannel):

                def model_func(carry: jnp.ndarray, inputs: jnp.ndarray, model: Any):
                    return model(inputs, carry)

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
            x = self.mlps[i](x)
            inputs = x
        yinit_guess = jnp.stack(x_from_all_layers)
        return self.mlps[-1](x), yinit_guess


if __name__ == "__main__":
    # this only works with jax.vmap(seq1d) in MultiScaleGRU
    ninp = 1
    nstate = 32
    nchannel = 4
    nclass = 2
    nlayer = 3
    model = MultiScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=jax.random.PRNGKey(1)
    )
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

    nseq = 69
    batch_size = 7
    carry = jnp.zeros(
        (batch_size, int(nstate / nchannel))
    )  # (batch_size, nstates)
    inputs = jax.random.normal(
        jax.random.PRNGKey(1),
        (batch_size, nseq, ninp),
    )  # (batch_size, nsequence, nstates)
    yinit_guess = jax.random.normal(
        jax.random.PRNGKey(1),
        (batch_size, nseq, int(nstate / nchannel)),
    )  # (batch_size, nsequence, nstates)
    y = model(
        inputs,
        [[carry for _ in range(nchannel)] for _ in range(nlayer)],
        [[yinit_guess for _ in range(nchannel)] for _ in range(nlayer)]
    )
