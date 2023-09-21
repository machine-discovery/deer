from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
from models import SingleScaleGRU


jax.config.update('jax_enable_x64', True)


def test_scan_deer_match():
    ninp = 6
    nstate = 32
    nsequence = 1000
    nclass = 5
    nlayer = 5
    nchannel = 1
    batch_size = 1
    dtype = jnp.float64
    key = jax.random.PRNGKey(42)

    model_deer = SingleScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=key,
        use_scan=False
    )
    model_scan = SingleScaleGRU(
        ninp=ninp,
        nchannel=nchannel,
        nstate=nstate,
        nlayer=nlayer,
        nclass=nclass,
        key=key,
        use_scan=True
    )

    model_deer = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model_deer)
    model_scan = jax.tree_util.tree_map(lambda x: x.astype(dtype) if eqx.is_array(x) else x, model_scan)
    y0 = jnp.zeros(
        (batch_size, int(nstate / nchannel)),
        dtype=dtype
    )  # (batch_size, nstates)
    yinit_guess = jnp.zeros(
        (batch_size, nsequence, int(nstate / nchannel)),
        dtype=dtype
    )  # (batch_size, nsequence, nstates)
    x = jax.random.normal(
        jax.random.PRNGKey(0),
        (batch_size, nsequence, ninp),
        dtype
    )

    def rollout(
        model: eqx.Module,
        y0: jnp.ndarray,
        inputs: jnp.ndarray,
        yinit_guess: List[jnp.ndarray],
        method: str = "deer_rnn",
    ):
        out, yinit_guess = model(inputs, y0, yinit_guess)
        return out

    out_deer, _ = jax.vmap(
        model_deer, in_axes=(0, 0, 0), out_axes=(0, 2)
    )(x, y0, yinit_guess)

    out_scan, _ = jax.vmap(
        model_scan, in_axes=(0, 0, 0), out_axes=(0, 2)
    )(x, y0, yinit_guess)

    assert out_deer.shape == (batch_size, nsequence, nclass)
    assert out_scan.shape == (batch_size, nsequence, nclass)

    assert jnp.allclose(out_deer, out_scan)
