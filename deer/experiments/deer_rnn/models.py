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
        self.norms = [
            nn.LayerNorm(
                name=f"norm_{i}",
            ) for i in range(self.nlayer)
        ]

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

    def initialize_carry(self, batch_size):
        # no seed needed since it is constant
        return jnp.ones((batch_size, self.nhidden))

    @nn.compact
    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)
        outputs = []

        for i in range(self.ngru):
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


class ScaleGRU(nn.Module):
    nhidden: int
    dtype: Any

    def setup(self):
        self.gru = nn.GRUCell(
            features=self.nhidden,
            dtype=self.dtype,
            param_dtype=self.dtype,
        )

    def initialize_carry(self, batch_size):
        return jnp.ones((batch_size, self.nhidden))

    @nn.compact
    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray, log_s: float):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)

        s = jnp.exp(log_s)
        states, _ = self.gru(h0, inputs / s)
        states = (states - h0) / s + h0
        return states, states


class StackedScaleGRU(nn.Module):
    nlayer: int
    nhidden: int
    dtype: Any

    def logspace_init(self, start: float, stop: float, num: int):
        values = jnp.log(jnp.logspace(start, stop, num, base=10))

        def init_fn(rng, shape):
            return values

        return init_fn

    def setup(self):
        self.encoder = MLP(
            name="linear_encoder",
            nout=self.nhidden,
            nstates=self.nhidden,
            dtype=self.dtype,
        )
        self.pre_gru_norm = nn.LayerNorm(
            name="pre_gru_norm",
        )
        self.gru_cells = [
            ScaleGRU(
                name=f"scale_gru_{i}",
                nhidden=self.nhidden,
                dtype=self.dtype,
            ) for i in range(self.nlayer)
        ]
        self.norms = [
            nn.LayerNorm(
                name=f"norm_{i}",
            ) for i in range(self.nlayer)
        ]
        self.mlp = [
            MLP(
                name=f"mlp_{i}",
                nout=self.nhidden,
                nstates=self.nhidden,
                dtype=self.dtype,
            ) for i in range(self.nlayer - 1)
        ]
        self.log_s = self.param(
            "log_s",
            self.logspace_init(start=0, stop=self.nlayer - 1, num=self.nlayer),
            # self.logspace_init(start=-1, stop=0, num=self.nlayer),
            (self.nlayer,)
        )

    def initialize_carry(self, batch_size):
        return jnp.zeros((batch_size, self.nhidden))

    def __call__(self, h0: jnp.ndarray, inputs: jnp.ndarray):
        # h0.shape == (nbatch, nstates)
        # inputs.shape == (nbatch, ninp)

        # inputs = self.encoder(inputs)
        # inputs = self.pre_gru_norm(inputs)
        # concat_states = []

        for i, cell in enumerate(self.gru_cells):
            states, _ = cell(h0, inputs, self.log_s[i])
            # states, _ = cell(h0[i * self.nhidden:(i + 1) * self.nhidden], inputs, self.log_s[i])
            # concat_states.append(states)
            if i < self.nlayer - 1:
                states = self.mlp[i](states)
                # states = self.norms[i](states)
            inputs = states
        # states += a

        jax.debug.print("{log_s}", log_s=self.log_s)

        # only the state in the last layer is returned, tuple for consistency
        # see GRUCell source code __call__

        # states = jnp.concatenate(concat_states, axis=-1)

        return states, states


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
