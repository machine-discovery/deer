from typing import Any, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from deer.seq1d import seq1d


class MFNSine(nn.Module):
    ninputs: int
    noutputs: int
    nhiddens: int
    nlayers: int
    input_scale: float = 256.0
    dtype: Any = jnp.float32

    def setup(self):
        # initialize the linear part
        lin_weight_scale = 1.0
        linscale = (lin_weight_scale / self.nhiddens) ** 0.5
        lin_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        self.lins = [
            nn.Dense(self.nhiddens,
                     kernel_init=lin_kernel_init,
                     dtype=self.dtype, param_dtype=self.dtype)
            for _ in range(self.nlayers)
        ]
        self.linout = nn.Dense(self.noutputs, dtype=self.dtype, param_dtype=self.dtype)

        # initialize the filter part
        filt_weight_scale = self.input_scale / (self.nlayers + 1) ** 0.5
        filt_weight_bound = filt_weight_scale / (self.nhiddens ** 0.5)
        filt_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-filt_weight_bound, maxval=filt_weight_bound)
        filt_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-np.pi, maxval=np.pi)
        self.filters = [
            nn.Sequential([
                nn.Dense(self.nhiddens,
                         kernel_init=filt_kernel_init, bias_init=filt_bias_init,
                         dtype=self.dtype, param_dtype=self.dtype),
                jnp.sin
            ]) for _ in range(self.nlayers + 1)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., ninputs)
        # returns (..., noutputs)
        out = self.filters[0](x)  # (..., nhiddens)
        for i in range(1, self.nlayers + 1):
            out = self.filters[i](x) * self.lins[i - 1](out)
        out = self.linout(out)
        return out

class MFNSineLong(nn.Module):
    # The implementation of MFN Sine for ultra-long but thin network
    # The evaluation is using the DEER framework
    ninputs: int
    noutputs: int
    nhiddens: int
    nlayers: int
    input_scale: float = 256.0
    dtype: Any = jnp.float32

    def setup(self) -> jnp.ndarray:
        # initialize the linear part
        lin_weight_scale = 1.0
        linscale = (lin_weight_scale / self.nhiddens) ** 0.5
        lin_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        lin_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        # initialize the parameters of the linear part
        # (nlayers, nhiddens, nhiddens)
        self.lin_weights = self.param(
            "lin_weights", lin_kernel_init, (self.nlayers, self.nhiddens, self.nhiddens), self.dtype)
        # (nlayers, nhiddens)
        self.lin_biases = self.param(
            "lin_biases", lin_bias_init, (self.nlayers, self.nhiddens), self.dtype)
        self.linout = nn.Dense(self.noutputs, dtype=self.dtype, param_dtype=self.dtype)

        # initialize the filter part
        filt_weight_scale = self.input_scale / (self.nlayers + 1) ** 0.5
        filt_weight_bound = filt_weight_scale / (self.nhiddens ** 0.5)
        filt_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-filt_weight_bound, maxval=filt_weight_bound)
        filt_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-np.pi, maxval=np.pi)
        # (nlayers + 1, ninputs, nhiddens)
        self.filt_weights = self.param(
            "filt_weights", filt_kernel_init, (self.nlayers + 1, self.ninputs, self.nhiddens), self.dtype)
        # (nlayers + 1, nhiddens)
        self.filt_biases = self.param(
            "filt_biases", filt_bias_init, (self.nlayers + 1, self.nhiddens), self.dtype)

    def __call__(self, xcoord: jnp.ndarray) -> jnp.ndarray:
        # x: (ninputs,)
        # returns (noutputs,)
        out0 = self.filter(xcoord, self.filt_weights[0], self.filt_biases[0])  # (nhiddens,)

        def func_next(y: jnp.ndarray, x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                      params: jnp.ndarray) -> jnp.ndarray:
            # y: (nhiddens,)
            # x: (lin_weights, lin_biases, filt_weights, filt_biases)
            # x.shapes: (nlayers, nhiddens, nhiddens), (nlayers, nhiddens),
            #           (nlayers, ninputs, nhiddens), (nlayers, nhiddens)
            # params: xcoord (ninputs,)
            # ynext: (nhiddens,)
            lin_weight, lin_bias, filt_weight, filt_bias = x
            xcoord = params
            filt_out = self.filter(xcoord, filt_weight, filt_bias)  # (nhiddens,)
            lin_out = y @ lin_weight + lin_bias  # (nhiddens,)
            out = filt_out * lin_out  # (nhiddens,)
            # out = out + y  # skip connection
            return out

        # evaluate the depth in parallel
        wbs = (self.lin_weights, self.lin_biases, self.filt_weights[1:], self.filt_biases[1:])
        outs = seq1d(func_next, out0, wbs, xcoord)  # (nlayers, nhiddens)
        # out1 = outs[-1]  # obtain the output at the last layer
        out1 = jnp.reshape(outs, -1)  # (nlayers * nhiddens)
        return self.linout(out1)
    
    def filter(self, xcoord: jnp.ndarray, filt_weight: jnp.ndarray, filt_bias: jnp.ndarray) -> jnp.ndarray:
        # x: (ninputs,)
        # returns (noutputs,)
        return jnp.sin(xcoord @ filt_weight + filt_bias)

class MFNGaborLong(nn.Module):
    # The implementation of MFN Gabor for ultra-long but thin network
    # The evaluation is using the DEER framework
    ninputs: int
    noutputs: int
    nhiddens: int
    nlayers: int
    input_scale: float = 256.0
    dtype: Any = jnp.float32

    def setup(self) -> jnp.ndarray:
        # initialize the linear part
        lin_weight_scale = 1.0
        linscale = (lin_weight_scale / self.nhiddens) ** 0.5
        lin_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        lin_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        # initialize the parameters of the linear part
        # (nlayers, nhiddens, nhiddens)
        self.lin_weights = self.param(
            "lin_weights", lin_kernel_init, (self.nlayers, self.nhiddens, self.nhiddens), self.dtype)
        # (nlayers, nhiddens)
        self.lin_biases = self.param(
            "lin_biases", lin_bias_init, (self.nlayers, self.nhiddens), self.dtype)
        self.linout = nn.Dense(self.noutputs, dtype=self.dtype, param_dtype=self.dtype)

        # initialize the filter part
        filt_weight_scale = self.input_scale / (self.nlayers + 1) ** 0.5
        filt_weight_bound = filt_weight_scale / (self.nhiddens ** 0.5)
        filt_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-filt_weight_bound, maxval=filt_weight_bound)
        filt_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-np.pi, maxval=np.pi)
        filt_gamma_init = lambda rng, shape, dtype: \
            jax.random.gamma(rng, 1.0, shape, dtype=dtype) * 1.0
        filt_mu_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-1.0, maxval=1.0)
        # (nlayers + 1, ninputs, nhiddens)
        self.filt_weights = self.param(
            "filt_weights", filt_kernel_init, (self.nlayers + 1, self.ninputs, self.nhiddens), self.dtype)
        # (nlayers + 1, nhiddens)
        self.filt_biases = self.param(
            "filt_biases", filt_bias_init, (self.nlayers + 1, self.nhiddens), self.dtype)
        self.filt_gammas = self.param(
            "filt_gammas", filt_gamma_init, (self.nlayers + 1, self.nhiddens), self.dtype)
        self.filt_mus = self.param(
            "filt_mus", filt_mu_init, (self.nlayers + 1, self.nhiddens, self.ninputs), self.dtype)

    def __call__(self, xcoord: jnp.ndarray) -> jnp.ndarray:
        # x: (ninputs,)
        # returns (noutputs,)
        out0 = self.filter(xcoord, self.filt_weights[0], self.filt_biases[0],
                           self.filt_gammas[0], self.filt_mus[0])  # (nhiddens,)

        def func_next(y: jnp.ndarray, x: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
                      params: jnp.ndarray) -> jnp.ndarray:
            # y: (nhiddens,)
            # x: (lin_weights, lin_biases, filt_weights, filt_biases, filt_gammas, filt_mus)
            # x.shapes: (nlayers, nhiddens, nhiddens), (nlayers, nhiddens),
            #           (nlayers, ninputs, nhiddens), (nlayers, nhiddens),
            #           (nlayers, nhiddens), (nlayers, nhiddens, ninputs)
            # params: xcoord (ninputs,)
            # ynext: (nhiddens,)
            lin_weight, lin_bias, filt_weight, filt_bias, filt_gamma, filt_mu = x
            xcoord = params
            filt_out = self.filter(xcoord, filt_weight, filt_bias, filt_gamma, filt_mu)  # (nhiddens,)
            lin_out = y @ lin_weight + lin_bias  # (nhiddens,)
            out = filt_out * lin_out  # (nhiddens,)
            # out = out + y  # skip connection
            return out

        # evaluate the depth in parallel
        wbs = (self.lin_weights, self.lin_biases,
               self.filt_weights[1:], self.filt_biases[1:], self.filt_gammas[1:], self.filt_mus[1:])
        outs = seq1d(func_next, out0, wbs, xcoord)  # (nlayers, nhiddens)
        out1 = jnp.reshape(outs, -1)  # (nlayers * nhiddens)
        return self.linout(out1)
    
    def filter(self, xcoord: jnp.ndarray, filt_weight: jnp.ndarray, filt_bias: jnp.ndarray,
               filt_gamma: jnp.ndarray, filt_mu: jnp.ndarray) -> jnp.ndarray:
        # xcoord: (ninputs,)
        # returns (noutputs,)
        d = jnp.sum(xcoord ** 2) + jnp.sum(filt_mu ** 2, axis=-1) - 2.0 * (filt_mu @ xcoord)  # (nhiddens,)
        return jnp.sin(xcoord @ filt_weight + filt_bias) * jnp.exp(-0.5 * d * filt_gamma)  # (nhiddens,)

class SIRENLong(nn.Module):
    # The implementation of SIREN for ultra-long but thin network
    # The evaluation is using the DEER framework
    ninputs: int
    noutputs: int
    nhiddens: int
    nlayers: int
    w0: float = 10.0
    dtype: Any = jnp.float32
    method: str = "sequential"

    def setup(self) -> jnp.ndarray:
        # initialize the linear part
        linscale = (6.0 / self.nhiddens) ** 0.5
        lin_kernel_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-linscale, maxval=linscale)
        lin_bias_init = lambda rng, shape, dtype: \
            jax.random.uniform(rng, shape, dtype=dtype, minval=-np.pi, maxval=np.pi)
        # initialize the parameters of the linear part
        # (nlayers, nhiddens, nhiddens)
        self.lin_weights = self.param(
            "lin_weights", lin_kernel_init, (self.nlayers, self.nhiddens, self.nhiddens), self.dtype)
        # (nlayers, nhiddens)
        self.lin_biases = self.param(
            "lin_biases", lin_bias_init, (self.nlayers, self.nhiddens), self.dtype)
        self.lininp = nn.Dense(self.nhiddens, dtype=self.dtype, param_dtype=self.dtype)
        self.linout = nn.Dense(self.noutputs, dtype=self.dtype, param_dtype=self.dtype, use_bias=True)

    def __call__(self, xcoord: jnp.ndarray) -> jnp.ndarray:
        # x: (ninputs,)
        # returns (noutputs,)
        # act = jnp.sin
        act = jnp.sin
        y0 = act(self.lininp(xcoord) * self.w0)  # (nhiddens,)

        def func_next(y: jnp.ndarray, x: Tuple[jnp.ndarray, jnp.ndarray], params: None) -> jnp.ndarray:
            # y: (nhiddens,)
            # x: (lin_weights, lin_biases)
            # x.shapes: (nlayers, nhiddens, nhiddens), (nlayers, nhiddens)
            # params: None
            # ynext: (nhiddens,)
            lin_weight, lin_bias = x
            out = act(y @ lin_weight + lin_bias)  # (nhiddens,)
            # out = out + y  # skip connection
            return out

        def scan_func(y: jnp.ndarray, x: Tuple[jnp.ndarray, jnp.ndarray]) \
                -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
            lin_weight, lin_bias = x
            out = act(y @ lin_weight + lin_bias)  # (nhiddens,)
            return out, out

        # evaluate the depth in parallel
        wbs = (self.lin_weights, self.lin_biases)
        if self.method == "sequential":
            _, outs = jax.lax.scan(scan_func, y0, wbs)
            out1 = jnp.reshape(outs, -1)
        else:
            outs = seq1d(func_next, y0, wbs, None, max_iter=1000)  # (nlayers, nhiddens)
        out1 = jnp.reshape(outs, -1)
        return self.linout(out1)
