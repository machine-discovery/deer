from typing import Tuple, Callable, Sequence, Any
import itertools
from functools import partial
from time import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen
from deer.seq1d import seq1d


# jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

def benchmark_seq1d_gru(
        nh: int = 8,
        batch_size: int = 16,
        nsequence: int = 100000,
        seed: int = 0,
        dtype: Any = jnp.float32):

    gru = flax.linen.GRUCell(features=nh, dtype=dtype, param_dtype=dtype)
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2, key = jax.random.split(key, 3)

    # initialize the model and get the first carry
    # carry = gru.initialize_carry(subkey1, (nh,))  # (nh,)
    # inputs = jax.random.normal(subkey2, (nsequence, nh), dtype=dtype)  # (nsequence, nh)
    carry = gru.initialize_carry(subkey1, (batch_size, nh))  # (batch_size, nh)
    inputs = jax.random.normal(subkey2, (nsequence, batch_size, nh), dtype=dtype)  # (nsequence, batch_size, nh)
    params = gru.init(key, carry, inputs[0])

    def func1(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        carry, outputs = jax.lax.scan(partial(gru.apply, params), carry, inputs)
        return outputs

    def func2(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any) -> jnp.ndarray:
        gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]
        return jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)(gru_func, carry, inputs, params)
        # return seq1d(gru_func, carry, inputs, params)

    func_benchmark(
        func1, func2, (carry, inputs, params),
        func1_name="Sequential GRU", func2_name="DEER GRU")

def func_benchmark(
        func1: Callable, func2: Callable, args: Sequence, with_jit: bool = True,
        func1_name: str = "func1", func2_name: str = "func2"):

    if with_jit:
        func1 = jax.jit(func1)
        func2 = jax.jit(func2)

    nwarmups = 5
    nreps = 20

    # warmup
    for _ in range(nwarmups):
        x1 = func1(*args)
        jax.block_until_ready(x1)

    # benchmark func1
    t0 = time()
    for _ in (range(nreps)):
        x1 = func1(*args)
        jax.block_until_ready(x1)
    t1 = time()
    time1_tots = (t1 - t0) / nreps
    print(f"{func1_name} time: {time1_tots:.3e} s")

    # warmup
    for _ in range(nwarmups):
        x2 = func2(*args)
        jax.block_until_ready(x2)

    # benchmark func2
    t0 = time()
    for _ in (range(nreps)):
        x2 = func2(*args)
        jax.block_until_ready(x2)
    t1 = time()
    time2_tots = (t1 - t0) / nreps

    print(f"{func2_name} time: {time2_tots:.3e} s")
    print(f"Speedup of {func2_name} over {func1_name}:", time1_tots / time2_tots)
    print("Max relative error:", jnp.max(jnp.abs((x1 - x2) / x1.at[x1 == 0.0].set(1e-8))))
    print("Max absolute error:", jnp.max(jnp.abs((x1 - x2))))
    print("Max and min of x1:", jnp.max(x1), jnp.min(x1))
    # assert jnp.allclose(x1, x2), "outputs are not close"

if __name__ == "__main__":
    for (nh, nsequence) in itertools.product([1, 2, 4, 8, 16, 32], [1000, 3000, 10000, 30000, 100000, 300000, 1000000]):
        for seed in range(5):
            print("nh:", nh, "nsequence:", nsequence, "seed:", seed)
            benchmark_seq1d_gru(nh=nh, nsequence=nsequence, seed=seed)
            print("--------")
