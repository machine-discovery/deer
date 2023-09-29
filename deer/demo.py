import argparse
import time
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
from deer.seq1d import seq1d


@functools.partial(jax.jit, static_argnames=("method", "gru_static"))
def eval_gru(carry: jnp.ndarray, inputs: jnp.ndarray, gru_params, gru_static, method: str = "sequential") \
        -> jnp.ndarray:
    # carry: (batch_size, hidden_size)
    # inputs: (length, batch_size, input_size)
    # outputs: (length, batch_size, hidden_size)
    if method == "sequential":
        gru = eqx.combine(gru_params, gru_static)
        gru_method = jax.vmap(gru, in_axes=0, out_axes=0)

        def call_gru1(carry: jnp.ndarray, inputs: jnp.ndarray):
            output = gru_method(inputs, carry)
            return output, output

        _, outputs = jax.lax.scan(call_gru1, carry, inputs)

    elif method == "deer":
        def call_gru2(carry: jnp.ndarray, inputs: jnp.ndarray, params):
            gru = eqx.combine(params, gru_static)
            return gru(inputs, carry)

        seq1dm = jax.vmap(seq1d, in_axes=(None, 0, 1, None), out_axes=1)
        outputs = seq1dm(call_gru2, carry, inputs, gru_params)

    else:
        raise ValueError(f"Unknown method: '{method}'. Must be 'sequential' or 'deer'.")

    return outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--inputsize", type=int, default=1, help="The number of input features")
    parser.add_argument("--batchsize", type=int, default=16, help="Batch size")
    parser.add_argument("--length", type=int, default=10000, help="Sequence length")
    args = parser.parse_args()

    # problem setup
    seed = args.seed
    input_size = args.inputsize
    batch_size = args.batchsize
    length = args.length

    print("===========================")
    print("Problem setup")
    print("---------------------------")
    print(f"* Batch size: {batch_size}")
    print(f"* Input size: {input_size}")
    print(f"* Sequence length: {length}")
    print("===========================")

    # initialize the random seed
    key = jax.random.PRNGKey(seed)
    key, *subkey = jax.random.split(key, 3)

    # create a GRUCell
    hidden_size = input_size
    gru = eqx.nn.GRUCell(input_size, hidden_size, key=subkey[0])
    gru_params, gru_static = eqx.partition(gru, eqx.is_array)

    # initialize the random inputs and the initial states of the GRUCell
    x = jax.random.normal(subkey[1], (length, batch_size, input_size))
    carry = jnp.zeros((batch_size, hidden_size))

    # warm up for sequential method
    print("Warming up sequential method", end="\r")
    outputs = eval_gru(carry, x, gru_params, gru_static, method="sequential")

    # benchmark for sequential method
    print("Benchmarking sequential method", end="\r")
    start = time.time()
    outputs1 = eval_gru(carry, x, gru_params, gru_static, method="sequential")
    end = time.time()
    seq_time = end - start
    print(f"Benchmarking sequential method: {seq_time:.5f} seconds")

    # warm up
    print("Warming up DEER", end="\r")
    outputs = eval_gru(carry, x, gru_params, gru_static, method="deer")

    # benchmark
    print("Benchmarking DEER", end="\r")
    start = time.time()
    outputs2 = eval_gru(carry, x, gru_params, gru_static, method="deer")
    end = time.time()
    deer_time = end - start
    print(f"Benchmarking DEER: {deer_time:.5f} seconds")

    print(f"DEER GRU speed up over sequential GRU: {seq_time / deer_time:.3f}x")

    # calculate the error
    dev = jnp.abs(outputs1 - outputs2).max()
    maxout = outputs1.max()
    minout = outputs1.min()
    print(f"Maximum absolute deviation: {dev:.3e} where output range: {minout:3e} to {maxout:3e}")

if __name__ == "__main__":
    main()
