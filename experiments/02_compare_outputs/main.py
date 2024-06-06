from typing import Tuple, Callable, Sequence, Any
import itertools
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import matplotlib.pyplot as plt
from deer.fseq1d import seq1d


def compare_outputs(
        nh: int = 32,
        nsequence: int = 10000,
        seed: int = 0,
        batch_size: int = 1,
        dtype: Any = jnp.float32):

    key = jax.random.PRNGKey(seed)
    key, *subkey = jax.random.split(key, 3)
    gru = eqx.nn.GRUCell(nh, nh, key=subkey[0])
    # split the module into parameters and static parts
    gru_params, gru_static = eqx.partition(gru, eqx.is_array)
    gru_params = jax.tree_util.tree_map(lambda x: x.astype(dtype) if x is not None else x, gru_params)

    key = jax.random.PRNGKey(seed)
    key, *subkey = jax.random.split(key, 3)

    carry = jnp.zeros((batch_size, nh), dtype=dtype)
    inputs = jax.random.normal(subkey[1], (nsequence, batch_size, nh), dtype=dtype)  # (nsequence, batch_size, nh)

    @jax.jit
    def func1(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        gru = eqx.combine(gru_params, gru_static)
        gru_method = jax.vmap(gru, in_axes=0, out_axes=0)

        def call_gru1(carry: jnp.ndarray, inputs: jnp.ndarray):
            output = gru_method(inputs, carry)
            return output, output

        _, outputs = jax.lax.scan(call_gru1, carry, inputs)
        return outputs  # (nsequence, batch_size, nh)

    @jax.jit
    def func2(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        def call_gru2(carry: jnp.ndarray, inputs: jnp.ndarray, params):
            gru = eqx.combine(params, gru_static)
            return gru(inputs, carry)

        seq1dm = jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)
        outputs = seq1dm(call_gru2, carry, inputs, gru_params)
        return outputs

    # compile
    _ = func1(carry, inputs, gru_params)
    _ = func2(carry, inputs, gru_params)
    y1 = func1(carry, inputs, gru_params)[:, 0, 0]  # (nsequence,)
    y2 = func2(carry, inputs, gru_params)[:, 0, 0]  # (nsequence,)

    label_fontsize = 14
    legend_fontsize = 12
    ticks_fontsize = 12
    plt.figure(figsize=(12, 3.5))
    plt.subplot(1, 2, 1)
    x = np.arange(nsequence)
    nlast = 200
    plt.plot(x[-nlast:], y1[-nlast:], label="Sequential")
    plt.plot(x[-nlast:], y2[-nlast:], label="DEER")
    plt.xlabel("Sequence index\n(a)", fontsize=label_fontsize)
    plt.title(f"GRU outputs for the last {nlast} indices", fontsize=label_fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.subplot(1, 2, 2)
    plt.plot(y1 - y2)
    plt.title("Difference between sequential and DEER outputs", fontsize=label_fontsize)
    plt.xlabel("Sequence index\n(b)", fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()
    plt.savefig("gru-outputs.png")

if __name__ == "__main__":
    compare_outputs()
