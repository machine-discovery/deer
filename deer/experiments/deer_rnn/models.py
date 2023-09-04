from typing import Any
import jax.numpy as jnp
from jax.nn.initializers import glorot_uniform, xavier_uniform
# from jax.experimental.stax import glorot_uniform
from flax import linen as nn
import pdb
import jax


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


class StackedGRU(nn.Module):
    nlayer: int
    nhidden: int
    dtype: Any

    def setup(self):
        self.gru_cells = [
            nn.GRUCell(
                name=f"gru_{i}",
                features=self.nhidden,
                dtype=self.dtype,
                param_dtype=self.dtype,
                # kernel_init=glorot_uniform(),
                # recurrent_kernel_init=glorot_uniform(),
            ) for i in range(self.nlayer)
        ]
        # self.lstm_cells = [
        #     nn.LSTMCell(
        #         name=f"lstm_{i}",
        #         features=self.nhidden,
        #         dtype=self.dtype,
        #         param_dtype=self.dtype,
        #         kernel_init=glorot_uniform()
        #     ) for i in range(self.nlayer)
        # ]
        self.norms = [
            nn.LayerNorm(
                name=f"norm_{i}",
            ) for i in range(self.nlayer)
        ]
        # self.dropout = nn.Dropout(rate=0.1, deterministic=False)

    def initialize_carry(self, batch_size):
        # no seed needed since it is constant
        return jnp.zeros((batch_size, self.nhidden))
        # return [jnp.zeros((batch_size, self.nhidden)) for _ in range(self.nlayer)]

    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)
        # pdb.set_trace()
        for i, cell in enumerate(self.gru_cells):
        # for i, cell in enumerate(self.lstm_cells):
            states, _ = cell(h0, inputs)
            # states = self.dropout(states)
            # states = self.norms[i](states)
            # jax.debug.print("{states}", states=h0)
            inputs = states
        # only the state in the last layer is returned, tuple for consistency
        # see GRUCell source code __call__
        # jax.debug.print("{inputs}", inputs=inputs)
        # pdb.set_trace()
        return states, states


class UnitGRU(nn.Module):
    nlayer: int  # unused for now
    nhidden: int
    dtype: Any

    @nn.compact
    def __call__(self, h0: jnp.ndarray, x: jnp.ndarray):
        x, _ = nn.GRUCell(
            features=self.nhidden,
            dtype=self.dtype,
            param_dtype=self.dtype,
            kernel_init=glorot_uniform()
        )(h0, x)
        return x


class MultiScaleGRU(nn.Module):
    ngru: int
    nlayer: int
    nhidden: int
    dtype: Any

    def setup(self):
        self.grus = [
            StackedGRU(
                self.nlayer,
                self.nhidden,
                self.dtype,
                name=f"gru_{i}"
            ) for i in range(self.ngru)
        ]
        self.log_s = self.param("log_s", nn.initializers.ones, (self.ngru,))
        self.log_s = jnp.log(jnp.logspace(-1, 1, self.ngru))

    def logspace_init(self, start, stop, num):
        values = jnp.logspace(start, stop, num)

        def init_func(rng, shape, index):
            return values[index]
        return init_func

    def initialize_carry(self, batch_size):
        # no seed needed since it is constant
        return jnp.ones((batch_size, self.nhidden))

    @nn.compact
    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)
        outputs = []

        for i in range(self.ngru):
            # log_s = self.param(f"log_scale_gru_{i}", self.custom_scalar_init, ())
            # log_s = self.param(f"log_scale_gru_{i}", self.logspace_init(-3, 3, self.ngru), (), index=i)
            log_s = self.log_s[i]
            # log_s = self.param(f"log_scale_gru_{i}", nn.initializers.ones, ())
            s = jnp.exp(log_s)
            # jax.debug.print("{s}", s=s)
            # states = UnitGRU(self.nlayer, self.nhidden, self.dtype, name=f"gru_{i}")(h0, inputs * s)
            states, _ = self.grus[i](h0, inputs * s)
            # jax.debug.print("{delta}", delta=jnp.mean(states - h0))
            states = (states - h0) * s + h0
            outputs.append(states)

        states = jnp.mean(jnp.stack(outputs), axis=0)

        return states, states

    @staticmethod
    def custom_scalar_init(rng, shape):
        scalars = -3 + 6 * jax.random.uniform(rng)
        # log_increment = 0.1
        # scalars = jnp.array([(i + 1) * log_increment for i in range(shape[0])])
        return scalars

    # def init_params(self, rng, h0, inputs):
    #     params = self.init({"params": rng}, h0, inputs)

    #     log_increment = 0.1
    #     scalars = self.custom_scalar_init(rng, (self.num_layers,), log_increment)
    #     for i in range(self.num_layers):
    #         params["params"][f"scalar_{i}"] = scalars[i]
    #     return params

# class MultiScaleGRU(nn.Module):
#     nlayer: int
#     nhidden: int
#     dtype: Any

#     def setup(self):
#         self.gru_cells = [
#             nn.GRUCell(
#                 name=f"gru_{i}",
#                 features=self.nhidden,
#                 dtype=self.dtype,
#                 param_dtype=self.dtype,
#                 kernel_init=glorot_uniform()
#             ) for i in range(self.nlayer)
#         ]

#     def initialize_carry(self, batch_size):
#         # no seed needed since it is constant
#         return jnp.zeros((batch_size, self.nhidden))

#     @nn.compact
#     def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
#         # h0.shape == (nbatch, nstates)
#         # inputs.shape == (nbatch, ninp)
#         log_s = self.param("log_scale", nn.initializers.ones, ())
#         s = jnp.exp(log_s)
#         inputs *=  s
#         for i, cell in enumerate(self.gru_cells):
#             states, _ = cell(h0, inputs)
#             inputs = states
#         states = h0 + (states - h0) * s
#         return states, states
