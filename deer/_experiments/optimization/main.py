from typing import Tuple
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
import optax
from deer.seq1d import seq1d


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

def func(x: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    # x: (nx,)
    # coeffs: (ncoeffs,)
    # return: (nx,)
    return jnp.sin(coeffs[0] * x + coeffs[1]) * coeffs[2] ** 2 * 3 / (1.0 + (coeffs[3] * x) ** 2)

def loss_fn(x: jnp.ndarray, coeffs: jnp.ndarray, ytrue: jnp.ndarray) -> jnp.ndarray:
    # x: (nx,)
    # coeffs: (ncoeffs)
    # ytrue: (nx,)
    # return: ()
    ypred = func(x, coeffs)  # (nx,)
    return jnp.mean((ypred - ytrue) ** 2)

def merge_states(*args: jnp.ndarray) -> jnp.ndarray:
    flatten = jax.tree_util.tree_flatten(args)[0]
    return jnp.concatenate([arr.ravel() for arr in flatten], axis=0)  # (nstates,)

def split_states(states: jnp.ndarray, tree):
    # states: (nstates,)
    shapes = jax.tree_util.tree_map(lambda x: x.shape, tree)
    numels, tree_struct = jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: x.size, tree))
    cumsum = list(np.cumsum(numels))
    states_lst = jnp.split(states, cumsum[:-1])
    states = jax.tree_util.tree_unflatten(tree_struct, states_lst)
    states = jax.tree_util.tree_map(lambda x, s: x.reshape(s), states, shapes)
    return states

def main():
    ncoeffs = 4
    nx = 1000

    # initialize the data
    key = jax.random.PRNGKey(0)
    key, *subkey = jax.random.split(key, 4)
    true_coeffs = jax.random.normal(subkey[0], shape=(ncoeffs,))
    x = jnp.linspace(-1, 1, nx)
    ydata = func(x, true_coeffs) + jax.random.normal(subkey[1], shape=(nx,)) * 1e-2

    # initialize the coefficients
    coeffs = jax.random.normal(subkey[2], shape=(ncoeffs,))
    optimizer = optax.rmsprop(learning_rate=1e-5)
    opt_state = optimizer.init(coeffs)

    states = merge_states(coeffs, opt_state)
    split_states_fcn = lambda states: split_states(states, (coeffs, opt_state))
    nt = 10000

    @jax.jit
    def update_state(states: jnp.ndarray, xinp: jnp.ndarray, params: Tuple) -> jnp.ndarray:
        # state: (nstates,)

        # x: (nx,)
        # ytrue: (nx,)
        x, ytrue = params
        # coeffs: (ncoeffs,)
        coeffs, opt_state = split_states_fcn(states)

        # calculate loss and the gradient
        loss, grad = jax.value_and_grad(loss_fn, argnums=1)(x, coeffs, ytrue)

        # update the coeffs
        updates, opt_state = optimizer.update(grad, opt_state, coeffs)
        new_coeffs = optax.apply_updates(coeffs, updates)

        # merge the states
        new_states = merge_states(new_coeffs, opt_state)
        return new_states

    def get_losses(all_states_t: jnp.ndarray) -> jnp.ndarray:
        coeffs_t, _ = jax.vmap(split_states_fcn)(all_states_t)  # (nt, ncoeffs)
        loss_t = jax.vmap(loss_fn, in_axes=(None, 0, None))(x, coeffs_t, ydata)  # (nt,)
        return loss_t

    seq1d2 = jax.jit(seq1d, static_argnames=("func", "yinit_guess", "max_iter"))

    xinp = jnp.zeros(nt)
    all_states_t = seq1d2(
        func=update_state, y0=states, xinp=xinp, params=(x, ydata), max_iter=10000)

    t0 = time.time()
    # all_states_t: (nt, nstates)
    all_states_t = seq1d2(
        func=update_state, y0=states, xinp=xinp, params=(x, ydata), max_iter=10000)
    t1 = time.time()
    print(t1 - t0)

    loss_t = get_losses(all_states_t)  # (nt,)

    # do the same thing with a for loop
    _ = update_state(states, xinp[0], (x, ydata))
    all_states = [states]
    t1 = time.time()
    for i in range(nt):
        states = update_state(states, xinp[i], (x, ydata))
        all_states.append(states)
    t2 = time.time()
    print(t2 - t1)

    all_states_t = jnp.stack(all_states, axis=0)  # (nt, nstates)
    loss_t2 = get_losses(all_states_t)  # (nt,)

    import matplotlib.pyplot as plt
    plt.plot(loss_t)
    plt.plot(loss_t2)
    plt.xlabel("Iteration")
    plt.title("Loss")
    plt.savefig("loss.png")

if __name__ == "__main__":
    main()
