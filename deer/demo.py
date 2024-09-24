import argparse
import time
import functools
import jax
import jax.numpy as jnp
import equinox as eqx
from deer.fseq1d import seq1d


jax.config.update("jax_enable_x64", True)

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
    parser.add_argument("--cell", type=str, default="gru", help="Cell type, either 'gru' or 'lstm'")
    parser.add_argument("--inputsize", type=int, default=2, help="The number of input features")
    parser.add_argument("--batchsize", type=int, default=16, help="Batch size")
    parser.add_argument("--length", type=int, default=10000, help="Sequence length")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type, either 'float32' or 'float64'")
    args = parser.parse_args()

    # problem setup
    seed = args.seed
    input_size = args.inputsize
    batch_size = args.batchsize
    length = args.length
    if args.dtype.lower() == "float32":
        dtype = jnp.float32
    elif args.dtype.lower() == "float64":
        dtype = jnp.float64
    else:
        raise ValueError(f"Unknown dtype: '{args.dtype}'. Must be 'float32' or 'float64'.")
    a = jnp.zeros((1,), dtype=dtype)

    print("=========================================")
    print("Problem setup")
    print("-----------------------------------------")
    print(f"* Random seed: {seed}")
    print(f"* Cell: {args.cell.upper()}")
    print(f"* Input size: {input_size}")
    print(f"* Batch size: {batch_size}")
    print(f"* Sequence length: {length}")
    print(f"* Data type: {a.dtype} with eps = {jnp.finfo(dtype).eps:.3e}")
    print("=========================================")
    print("You can change the problem setup by passing arguments to this script.")
    print("To see the list of arguments, run with --help.")
    print("")

    # initialize the random seed
    key = jax.random.PRNGKey(seed)
    key, *subkey = jax.random.split(key, 3)

    # create a GRUCell
    hidden_size = input_size
    if args.cell.lower() == "gru":
        gru = eqx.nn.GRUCell(input_size, hidden_size, key=subkey[0])
    elif args.cell.lower() == "lstm":
        assert hidden_size % 2 == 0, f"hidden_size must be even for LSTM, got {hidden_size}"
        gru = LSTMWrapper(eqx.nn.LSTMCell(input_size, hidden_size // 2, key=subkey[0]))
    else:
        raise ValueError(f"Unknown cell type: '{args.cell}'. Must be 'gru' or 'lstm'.")
    
    # split the module into parameters and static parts
    gru_params, gru_static = eqx.partition(gru, eqx.is_array)
    gru_params = jax.tree_util.tree_map(lambda x: x.astype(dtype) if x is not None else x, gru_params)

    # initialize the random inputs and the initial states of the GRUCell
    x = jax.random.normal(subkey[1], (length, batch_size, input_size), dtype=dtype)
    carry = jnp.zeros((batch_size, hidden_size), dtype=dtype)

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
    dev = jnp.abs(outputs1 - outputs2.value).max()
    maxout = outputs1.max()
    minout = outputs1.min()
    print(f"Maximum absolute deviation: {dev:.3e} where output range: {minout:3e} to {maxout:3e}")

class LSTMWrapper(eqx.Module):
    # wrapper for LSTM to make its states and outputs as one tensor, so the interface is the same as GRU
    lstm: eqx.nn.LSTMCell

    def __init__(self, lstm: eqx.nn.LSTMCell):
        super().__init__()
        self.lstm = lstm

    def __call__(self, input: jnp.ndarray, carry: jnp.ndarray) -> jnp.ndarray:
        carry1, carry2 = jnp.split(carry, indices_or_sections=2, axis=-1)
        out1, out2 = self.lstm(input, (carry1, carry2))
        return jnp.concatenate((out1, out2), axis=-1)

if __name__ == "__main__":
    main()
