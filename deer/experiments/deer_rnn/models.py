from typing import Any
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, xavier_uniform
from flax import linen as nn
import pdb


def he_uniform(key, shape, dtype):
    fan_in = shape[-1]
    bound = jnp.sqrt(6 / fan_in)
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


class MLP(nn.Module):
    nout: int
    nstates: int
    dtype: Any

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.nstates, dtype=self.dtype, kernel_init=he_uniform)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.nout, dtype=self.dtype, kernel_init=he_uniform)(x)
        return x


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
#         log_s = 1
#         s = jnp.exp(log_s)
#         states, _ = self.gru(h0, inputs / s)
#         # pdb.set_trace()
#         states = (states - h0) / s + h0
#         # pdb.set_trace()
#         return states, states


# class MLP(nn.Module):
#     nstates: int
#     nout: int
#     dtype: Any

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(self.nstates, dtype=self.dtype, kernel_init=he_uniform)(x)
#         x = nn.tanh(x)
#         x = nn.Dense(self.nout, dtype=self.dtype, kernel_init=he_uniform)(x)
#         return x


# class MultiScaleGRU(nn.Module):
#     nchannel: int
#     nstates: int
#     dtype: Any
    
#     def initialize_carry(self, batch_size):
#         return jnp.ones((batch_size, self.nhidden))

#     def setup(self):
#         self.encoder = MLP(nstates=self.nstates, dtype=self.dtype)
#         self.scale_grus = 
#         self.mlps = [
            
#         ]
