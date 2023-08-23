from typing import Any
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    ndim: int
    dtype: Any

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.ndim, dtype=self.dtype)(x)
        return x


class StackedGRU(nn.Module):
    nlayer: int
    nhidden: int
    dtype: Any

    def setup(self):
        self.gru_cells = [nn.GRUCell(name=f"gru_{i}", features=self.nhidden, dtype=self.dtype, param_dtype=self.dtype) for i in range(self.nlayer)]

    def initialize_carry(self, batch_size):
        # no seed needed since it is constant
        return jnp.zeros((batch_size, self.nhidden))
        # return [jnp.zeros((batch_size, self.nhidden)) for _ in range(self.nlayer)]

    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        for i, cell in enumerate(self.gru_cells):
            states, _ = cell(h0, inputs)
            inputs = states
        # only the state in the last layer is returned, tuple for consistency
        # see GRUCell source code __call__
        return states, states
