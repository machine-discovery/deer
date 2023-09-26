from typing import Tuple, Callable, Sequence, Any
import itertools
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen
import matplotlib.pyplot as plt
from deer.seq1d import seq1d

def compare_outputs(
        nh: int = 32,
        nsequence: int = 10000,
        seed: int = 0,
        dtype: Any = jnp.float32):

    gru = flax.linen.GRUCell(features=nh, dtype=dtype, param_dtype=dtype)
    key = jax.random.PRNGKey(seed)
    key, *subkey = jax.random.split(key, 3)

    carry = gru.initialize_carry(subkey[0], (1, nh))  # (batch_size, nh)
    inputs = jax.random.normal(subkey[1], (nsequence, 1, nh), dtype=dtype)  # (nsequence, batch_size, nh)
    params = gru.init(key, carry, inputs[0])

    @jax.jit
    def func1(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        carry, outputs = jax.lax.scan(partial(gru.apply, params), carry, inputs)
        return outputs  # (nsequence, batch_size, nh)

    @jax.jit
    def func2(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]
        return jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)(gru_func, carry, inputs, params)

    # compile
    _ = func1(carry, inputs, params)
    _ = func2(carry, inputs, params)
    y1 = func1(carry, inputs, params)[:, 0, 0]  # (nsequence,)
    y2 = func2(carry, inputs, params)[:, 0, 0]  # (nsequence,)

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
