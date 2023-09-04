import sys
from typing import Tuple, Callable, Sequence, Any
import itertools
from functools import partial
from time import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen
from deer.seq1d import seq1d
import pdb

# jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)


def loss_fn(
    logits: jnp.ndarray,
    labels: jnp.ndarray
) -> float:
    return jnp.mean((logits - labels) ** 2)


def benchmark_seq1d_gru(
        nh: int = 8,
        batch_size: int = 16,
        nsequence: int = 100000,
        seed: int = 0,
        dtype: Any = jnp.float32):

    gru = flax.linen.GRUCell(features=nh, dtype=dtype, param_dtype=dtype)
    key = jax.random.PRNGKey(seed)
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)

    # initialize the model and get the first carry
    # carry = gru.initialize_carry(subkey1, (nh,))  # (nh,)
    # inputs = jax.random.normal(subkey2, (nsequence, nh), dtype=dtype)  # (nsequence, nh)
    carry = gru.initialize_carry(subkey1, (batch_size, nh))  # (batch_size, nh)
    inputs = jax.random.normal(subkey2, (nsequence, batch_size, nh), dtype=dtype)  # (nsequence, batch_size, nh)
    labels = jax.random.normal(subkey3, (nsequence, batch_size, nh), dtype=dtype)  # (nsequence, batch_size, nh)
    params = gru.init(key, carry, inputs[0])


    def func1(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any, labels: jnp.ndarray) -> jnp.ndarray:
        carry, outputs = jax.lax.scan(partial(gru.apply, params), carry, inputs)
        # what should the output be
        loss = loss_fn(outputs, labels)
        return loss, outputs

    def func2(carry: jnp.ndarray, inputs: jnp.ndarray, params: Any, labels: jnp.ndarray) -> jnp.ndarray:
        gru_func = lambda carry, inputs, params: gru.apply(params, carry, inputs)[0]
        # check this line
        outputs = jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)(gru_func, carry, inputs, params)
        loss = loss_fn(outputs, labels)
        return loss, outputs

    func_benchmark(
        func1, func2, (carry, inputs, params, labels),
        func1_name="Sequential GRU", func2_name="DEER GRU")


def func_benchmark(
        func1: Callable, func2: Callable, args: Sequence, with_jit: bool = True,
        func1_name: str = "func1", func2_name: str = "func2"):

    if with_jit:
        func1 = jax.jit(jax.value_and_grad(func1, argnums=2, has_aux=True))
        func2 = jax.jit(jax.value_and_grad(func2, argnums=2, has_aux=True))

    nwarmups = 5
    nreps = 20

    # warmup
    for _ in tqdm(range(nwarmups), file=sys.stderr):
        (loss1, x1), grad1 = func1(*args)
        jax.block_until_ready(loss1)
        jax.block_until_ready(x1)
        jax.block_until_ready(grad1)

    # benchmark func1
    t0 = time()
    for _ in tqdm(range(nreps), file=sys.stderr):
        (loss1, x1), grad1 = func1(*args)
        jax.block_until_ready(loss1)
        jax.block_until_ready(x1)
        jax.block_until_ready(grad1)
    t1 = time()
    time1_tots = (t1 - t0) / nreps
    print(f"{func1_name} time: {time1_tots:.3e} s")

    # warmup
    for _ in tqdm(range(nwarmups), file=sys.stderr):
        (loss2, x2), grad2 = func2(*args)
        jax.block_until_ready(loss2)
        jax.block_until_ready(x2)
        jax.block_until_ready(grad2)

    # benchmark func2
    t0 = time()
    for _ in tqdm(range(nreps), file=sys.stderr):
        (loss2, x2), grad2 = func2(*args)
        jax.block_until_ready(x2)
        jax.block_until_ready(loss2)
        jax.block_until_ready(x2)
        jax.block_until_ready(grad2)

    t1 = time()
    time2_tots = (t1 - t0) / nreps

    print(f"{func2_name} time: {time2_tots:.3e} s")
    print(f"Speedup of {func2_name} over {func1_name}:", time1_tots / time2_tots)
    print("Max relative error:", jnp.max(jnp.abs((x1 - x2) / x1.at[x1 == 0.0].set(1e-8))))
    print("Max absolute error:", jnp.max(jnp.abs((x1 - x2))))
    print("Max and min of x1:", jnp.max(x1), jnp.min(x1))
    rel_errs = jax.tree_map(compute_rel_error, grad1, grad2)
    max_rel_err_per_tensor = jax.tree_map(jnp.max, rel_errs)
    max_rel_err = jax.tree_util.tree_reduce(jnp.maximum, max_rel_err_per_tensor)
    print("Max grad relative error:", max_rel_err)
    abs_errs = jax.tree_map(compute_abs_error, grad1, grad2)
    max_abs_err_per_tensor = jax.tree_map(jnp.max, abs_errs)
    max_abs_err = jax.tree_util.tree_reduce(jnp.maximum, max_abs_err_per_tensor)
    print("Max grad absolute error:", max_abs_err)
    concat_grads = jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), grad1, grad2)
    print("Max and min of grad1 and grad2:", tree_max(concat_grads), tree_min(concat_grads))
    # assert jnp.allclose(x1, x2), "outputs are not close"


def compute_abs_error(
    x: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    return jnp.abs(x - y)


def compute_rel_error(
    x: jnp.ndarray,
    y: jnp.ndarray
) -> jnp.ndarray:
    return jnp.abs(x - y) / (1e-10 + jnp.maximum(jnp.abs(x), jnp.abs(y)))


def tree_max(tree):
    return jax.tree_util.tree_reduce(jnp.maximum, jax.tree_map(jnp.max, tree))


def tree_min(tree):
    return jax.tree_util.tree_reduce(jnp.minimum, jax.tree_map(jnp.min, tree))


if __name__ == "__main__":
    # executed with "python seq1d.py > report.txt"
    batch_size = 16
    for (nh, nsequence) in itertools.product([1, 2, 4, 8, 16, 32, 64], [1000, 3000, 10000, 30000, 100000, 300000, 1000000]):
        for seed in range(5):
            print("nh:", nh, "nsequence:", nsequence, "seed:", seed)
            try:
                benchmark_seq1d_gru(nh=nh, nsequence=nsequence, seed=seed, batch_size=batch_size)
            except:
                print("Fail")
            print("--------")
    # for (nh, nsequence) in itertools.product([2], [1000]):
    #     for seed in range(1):
    #         print("nh:", nh, "nsequence:", nsequence, "seed:", seed)
    #         benchmark_seq1d_gru(nh=nh, nsequence=nsequence, seed=seed, batch_size=batch_size)
