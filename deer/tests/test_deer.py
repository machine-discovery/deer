from typing import Any, Tuple
import pytest
import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp as solve_ivp_scipy
from deer.seq1d import solve_ivp, seq1d, matmul_recursive


jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

def test_matmul_recursive():
    nsamples = 100
    ny = 4

    # generate random matrix with shape (nsamples - 1, ny, ny)
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    mats = jax.random.normal(subkey1, (nsamples - 1, ny, ny), dtype=jnp.float64) / 3
    vecs = jax.random.normal(subkey2, (nsamples - 1, ny), dtype=jnp.float64)
    y0 = jax.random.normal(subkey3, (ny,), dtype=jnp.float64)

    # generate the result using for loop
    result = jnp.zeros((nsamples, ny), dtype=jnp.float64)
    result = result.at[0].set(y0)
    for i in range(nsamples - 1):
        result = result.at[i + 1].set(mats[i] @ result[i] + vecs[i])
    # generate the result using matmul_recursive
    result2 = matmul_recursive(mats, vecs, y0)
    assert jnp.allclose(result, result2)

def test_solve_ivp():
    ny = 4
    dtype = jnp.float64
    key = jax.random.PRNGKey(0)
    subkey1, subkey2, subkey3 = jax.random.split(key, 3)
    A0 = (jax.random.uniform(subkey1, shape=(ny, ny), dtype=dtype) * 2 - 1) / ny ** 0.5
    A1 = jax.random.uniform(subkey2, shape=(ny, ny), dtype=dtype) / ny ** 0.5
    npts = 10000  # TODO: investigate why npts=1000 make nans
    tpts = jnp.linspace(0, 1.0, npts, dtype=dtype)  # (ntpts,)
    y0 = jax.random.uniform(subkey3, shape=(ny,), dtype=dtype)

    A0_np = np.array(A0)
    A1_np = np.array(A1)
    tpts_np = np.array(tpts)
    y0_np = np.array(y0)

    def func(y: jnp.ndarray, x: jnp.ndarray, params) -> jnp.ndarray:
        # x: (1,) is time
        # y: (ny,)
        # returns: (ny,)
        A0, A1 = params
        yy = jnp.tanh(A1 @ jax.nn.relu(A0 @ y))
        dfdy = -6 * yy - 10 * y ** 3 + 30 * jnp.sin(6 * x) + 3
        return dfdy

    def func_np(t: np.ndarray, y: np.ndarray, A0_np: np.ndarray, A1_np: np.ndarray) -> np.ndarray:
        yy = np.tanh(A1_np @ np.maximum(A0_np @ y, 0))
        dfdy = -6 * yy - 10 * y ** 3 + 30 * np.sin(6 * t) + 3
        return dfdy

    params = (A0, A1)
    params_np = (A0_np, A1_np)
    yt = solve_ivp(func, y0, tpts[..., None], params, tpts)  # (ntpts, ny)
    yt_np = solve_ivp_scipy(func_np, (tpts_np[0], tpts_np[-1]), y0_np, t_eval=tpts_np, args=params_np, rtol=1e-10, atol=1e-10).y.T

    # import matplotlib.pyplot as plt
    # plt.plot(tpts, yt[..., 0])
    # plt.plot(tpts, yt_np[..., 0])
    # # plt.plot(tpts, (yt - yt_np)[..., 0])
    # plt.savefig("test.png")
    # plt.close()

    assert jnp.allclose(yt, yt_np, atol=1e-6)

    # check the gradients
    def get_loss(y0, params):
        yt = solve_ivp(func, y0, tpts[..., None], params, tpts)  # (ntpts, ny)
        return jnp.sum(yt ** 2, axis=0)  # only sum over time
    jax.test_util.check_grads(
        get_loss, (y0, params), order=1, modes=['rev'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)

@pytest.mark.parametrize("jit", [True, False])
def test_rnn(jit: bool):
    # test the rnn with the DEER framework using GRU
    def gru_func(hprev: jnp.ndarray, xinp: jnp.ndarray, params: Any) -> jnp.ndarray:
        # hprev: (nh,)
        # xinp: (nx,)
        # params: Wir, Whr, bhr, Wiz, Whz, bhz, Win, Whn, bhn
        # returns: (nh,)
        Wir, Whr, bhr, Wiz, Whz, bhz, Win, Whn, bhn = params
        r = jax.nn.sigmoid(Wir @ xinp + Whr @ hprev + bhr)  # (nh,)
        z = jax.nn.sigmoid(Wiz @ xinp + Whz @ hprev + bhz)  # (nh,)
        n = jnp.tanh(Win @ xinp + r * (Whn @ hprev + bhn))  # (nh,)
        h = (1 - z) * n + z * hprev
        return h

    # generate random parameters
    dtype = jnp.float64
    key = jax.random.PRNGKey(0)
    nh, nx = 5, 3
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    Wir = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5
    Whr = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5
    bhr = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    Wiz = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5
    Whz = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5
    bhz = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    Win = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5
    Whn = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5
    bhn = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    params = (Wir, Whr, bhr, Wiz, Whz, bhz, Win, Whn, bhn)

    # generate random inputs and the initial condition
    nsteps = 100
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    xinp = jax.random.normal(subkey1, shape=(nsteps, nx), dtype=dtype)
    h0 = jax.random.normal(subkey2, shape=(nh,), dtype=dtype)

    # calculate the output states using seq1d
    if jit:
        func = jax.jit(seq1d, static_argnums=(0,))
    else:
        func = seq1d
    hseq = func(gru_func, h0, xinp, params)  # (nsteps, nh)

    # calculate the output states using a for loop
    hfor_list = [h0]
    for i in range(xinp.shape[0]):
        hfor = gru_func(hfor_list[-1], xinp[i], params)
        hfor_list.append(hfor)
    hfor = jnp.stack(hfor_list[1:], axis=0)  # (nsteps, nh)

    # import matplotlib.pyplot as plt
    # print(hfor.shape, hseq.shape)
    # plt.plot(hseq[:, 0])
    # plt.plot(hfor[:, 0])
    # plt.savefig("test.png")

    # check the outputs
    assert jnp.allclose(hseq, hfor, atol=1e-6)

def test_rnn_derivs():
    # test the rnn with the DEER framework using simple RNN function
    def rnn_func(hprev: jnp.ndarray, xinp: jnp.ndarray, params: Any) -> jnp.ndarray:
        # hprev: (nh,)
        # xinp: (nx,)
        # params: Wh (nh, nh), Wx (nh, nx)
        # returns: (nh,)
        Wh, Wx = params
        return jnp.tanh(Wx @ xinp + Wh @ hprev)

    nsteps = 5
    dtype = jnp.float64
    key = jax.random.PRNGKey(0)
    nh, nx = 3, 2
    # generate the matrices Wh and Wx
    subkey1, subkey2, key = jax.random.split(key, 3)
    Wh = (jax.random.uniform(subkey1, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5
    Wx = (jax.random.uniform(subkey2, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5
    # generate the inputs (nsteps, nx) and the initial condition (nh,)
    subkey1, subkey2, key = jax.random.split(key, 3)
    xinp = jax.random.normal(subkey1, shape=(nsteps, nx), dtype=dtype)
    h0 = jax.random.normal(subkey2, shape=(nh,), dtype=dtype)
    params = (Wh, Wx)

    def get_loss(h0: jnp.ndarray, xinp: jnp.ndarray, params: Any) -> jnp.ndarray:
        hseq = seq1d(rnn_func, h0, xinp, params)  # (nsteps, nh)
        return hseq

    jax.test_util.check_grads(
        get_loss, (h0, xinp, params), order=1, modes=['rev'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)

def test_input_in_a_tree():
    # recurrence of z_i = sigma(b_i + z_{i-1} * w_i)

    # generate random parameters
    dtype = jnp.float64
    key = jax.random.PRNGKey(0)
    nh = 2
    ndepths = 1000

    # generate the random parameters
    key, *subkey = jax.random.split(key, 4)
    b = (jax.random.uniform(subkey[0], (ndepths, nh), dtype=dtype) * 2 - 1) / nh ** 0.5
    w = (jax.random.uniform(subkey[1], (ndepths, nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5

    # initial values
    z0 = jax.random.uniform(subkey[2], (nh,), dtype=dtype) * 2 - 1

    def func_next_seq(zi: jnp.ndarray, xinp: Tuple[jnp.ndarray, jnp.ndarray], _: None) -> jnp.ndarray:
        # zi: (nh,)
        # xinp: (b_i, w_i): (nh,), (nh, nh)
        # returns: (nh,)
        bi, wi = xinp
        return jax.nn.log_sigmoid(bi + jnp.dot(zi, wi))

    xinps = (b, w)
    zk = seq1d(func_next_seq, z0, xinps, None)  # (ndepths, nh)

    # TODO: check the result
    zk_trues = []
    z = z0
    for i in range(ndepths):
        z = func_next_seq(z, (b[i], w[i]), None)  # (nh,)
        zk_trues.append(z)
    zk_true = jnp.stack(zk_trues, axis=0)  # (ndepths, nh)

    # check the outputs
    assert jnp.allclose(zk, zk_true, atol=1e-6)
