from typing import Any, Tuple
import pytest
import itertools
import functools
import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp as solve_ivp_scipy
from deer.maths import matmul_recursive
from deer import solve_ivp, solve_idae, seq1d


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

@pytest.mark.parametrize("method", [
    solve_ivp.DEER()
])
def test_solve_ivp(method):
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
    res = solve_ivp(func, y0, tpts[..., None], params, tpts, method=method)  # (ntpts, ny)
    yt = res.value
    yt_np = solve_ivp_scipy(func_np, (tpts_np[0], tpts_np[-1]), y0_np, t_eval=tpts_np, args=params_np, rtol=1e-10, atol=1e-10).y.T

    # check if res.success has the same shape as yt
    assert res.success.shape == yt.shape
    # check if it all success
    assert jnp.all(res.success)

    # import matplotlib.pyplot as plt
    # plt.plot(tpts, yt[..., 0])
    # plt.plot(tpts, yt_np[..., 0])
    # # plt.plot(tpts, (yt - yt_np)[..., 0])
    # plt.savefig("test.png")
    # plt.close()

    assert jnp.allclose(yt, yt_np, atol=1e-6)

    # check the gradients
    def get_loss(y0, params):
        yt = solve_ivp(func, y0, tpts[..., None], params, tpts, method=method).value  # (ntpts, ny)
        return jnp.sum(yt ** 2, axis=0)  # only sum over time
    jax.test_util.check_grads(
        get_loss, (y0, params), order=1, modes=['rev', 'fwd'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)

@pytest.mark.parametrize("method", [
    solve_idae.BwdEulerDEER(),
    solve_idae.BwdEuler(),
    ])
def test_solve_idae(method):
    dtype = jnp.float64

    gval = 10.0
    theta = np.pi / 2
    x0 = np.sin(theta)
    y0 = -np.cos(theta)
    u0 = v0 = 0.0
    T0 = -gval * y0
    vr0 = jnp.array([x0, y0, u0, v0, T0], dtype=dtype)

    g = jnp.array(gval, dtype=dtype)
    params = g
    npts = 10000
    tpts = jnp.linspace(0, 2.0, npts, dtype=dtype)  # (ntpts,)
    res = solve_idae(dae_pendulum, vr0, tpts[..., None], params, tpts, method=method)  # (ntpts, ny)
    vrt = res.value

    # check if res.success has the same shape as yt
    assert res.success.shape == vrt.shape
    # check if it all success
    assert jnp.all(res.success)

    # evaluate with numpy, but recast the problem into ODE because numpy does not have DAE solver
    def func_np(t: np.ndarray, vr: np.ndarray) -> np.ndarray:
        theta, dtheta = vr
        theta_dot = dtheta
        dtheta_dot = -gval * np.sin(theta)
        return np.array([theta_dot, dtheta_dot])

    sol_np = solve_ivp_scipy(func_np, (tpts[0], tpts[-1]), np.array([theta, 0.0]), t_eval=tpts, method="BDF").y.T
    # get the x, y, u, v, T
    x_np = np.sin(sol_np[:, 0])
    y_np = -np.cos(sol_np[:, 0])
    u_np = np.cos(sol_np[:, 0]) * sol_np[:, 1]
    v_np = np.sin(sol_np[:, 0]) * sol_np[:, 1]
    T_np = gval * np.cos(sol_np[:, 0]) + sol_np[:, 1] ** 2
    vrt_np = np.stack([x_np, y_np, u_np, v_np, T_np], axis=1)  # (ntpts, 5)

    # import matplotlib.pyplot as plt
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     if i == 5:
    #         plt.plot(tpts, vrt[..., 0] ** 2 + vrt[..., 1] ** 2 - 1)
    #     else:
    #         plt.plot(tpts, vrt[..., i])
    #         plt.plot(tpts, vrt_np[..., i])
    #         # plt.plot(tpts, vrt_np[..., i] - vrt[..., i])
    # plt.tight_layout()
    # plt.savefig("test.png")
    # plt.close()

    # solve_idae can satisfy the constraints very well
    assert jnp.allclose(vrt[..., 0] ** 2 + vrt[..., 1] ** 2 - 1, 0, atol=1e-12)
    # check the outputs (relatively high rel error because of different ways to compute)
    assert jnp.all((vrt[:, 0] - vrt_np[:, 0]) / jnp.max(jnp.abs(vrt_np[:, 0])) < 1e-2)
    assert jnp.all((vrt[:, 1] - vrt_np[:, 1]) / jnp.max(jnp.abs(vrt_np[:, 1])) < 1e-2)
    assert jnp.all((vrt[:, 2] - vrt_np[:, 2]) / jnp.max(jnp.abs(vrt_np[:, 2])) < 1e-2)
    assert jnp.all((vrt[:, 3] - vrt_np[:, 3]) / jnp.max(jnp.abs(vrt_np[:, 3])) < 2e-2)
    assert jnp.all((vrt[:, 4] - vrt_np[:, 4]) / jnp.max(jnp.abs(vrt_np[:, 4])) < 1e-2)

@pytest.mark.parametrize("method", [
    solve_idae.BwdEulerDEER(),
    solve_idae.BwdEuler(),
    ])
def test_solve_idae_derivs(method):
    dtype = jnp.float64

    gval = 10.0
    theta = np.pi / 2
    x0 = np.sin(theta)
    y0 = -np.cos(theta)
    u0 = v0 = 0.0
    T0 = -gval * y0
    vr0 = jnp.array([x0, y0, u0, v0, T0], dtype=dtype)

    g = jnp.array(gval, dtype=dtype)
    params = g
    npts = 1000
    tpts = jnp.linspace(0, 2.0, npts, dtype=dtype)  # (ntpts,)

    def get_loss(vr0, tpts, params: Any) -> jnp.ndarray:
        # (nsteps, nh)
        hseq = solve_idae(dae_pendulum, vr0, jnp.zeros_like(tpts[..., None]), params, tpts, method=method).value
        return hseq

    jax.test_util.check_grads(
        get_loss, (vr0, tpts, params), order=1, modes=['rev', 'fwd'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)

@pytest.mark.parametrize(
        "jit, difficult, method",
        itertools.product([True, False], [True, False],
                          [seq1d.DEER(),
                           seq1d.Sequential()]))
def test_rnn(jit: bool, difficult: bool, method):
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
    dtype = jnp.float32 if difficult else jnp.float64
    key = jax.random.PRNGKey(0)
    nh, nx = (2, 2) if difficult else (5, 3)
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    m = 100 if difficult else 1  # difficult makes the temporary value nans
    Wir = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5 * m
    Whr = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5 * m
    bhr = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    Wiz = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5 * m
    Whz = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5 * m
    bhz = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    Win = (jax.random.uniform(subkey1, (nh, nx), dtype=dtype) * 2 - 1) / nx ** 0.5 * m
    Whn = (jax.random.uniform(subkey2, (nh, nh), dtype=dtype) * 2 - 1) / nh ** 0.5 * m
    bhn = (jax.random.uniform(subkey3, (nh,), dtype=dtype) * 2 - 1) / nh ** 0.5
    params = (Wir, Whr, bhr, Wiz, Whz, bhz, Win, Whn, bhn)

    # generate random inputs and the initial condition
    nsteps = 100
    subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
    xinp = jax.random.normal(subkey1, shape=(nsteps, nx), dtype=dtype) / m
    h0 = jax.random.normal(subkey2, shape=(nh,), dtype=dtype)

    # calculate the output states using seq1d
    def func0(gru_func, h0, xinp, params):
        return seq1d(gru_func, h0, xinp, params, method=method)

    if jit:
        func = jax.jit(func0, static_argnums=(0,))
    else:
        func = func0
    res = func(gru_func, h0, xinp, params)  # (nsteps, nh)
    hseq = res.value

    # check if res.success has the same shape as yt
    assert res.success.shape == hseq.shape
    # check if it all success
    assert jnp.all(res.success)

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
        # (nsteps, nh)
        hseq = seq1d(rnn_func, h0, xinp, params, method=seq1d.DEER()).value
        return hseq

    jax.test_util.check_grads(
        get_loss, (h0, xinp, params), order=1, modes=['rev', 'fwd'],
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
    zk = seq1d(func_next_seq, z0, xinps, None).value  # (ndepths, nh)

    # compute the true values
    zk_trues = []
    z = z0
    for i in range(ndepths):
        z = func_next_seq(z, (b[i], w[i]), None)  # (nh,)
        zk_trues.append(z)
    zk_true = jnp.stack(zk_trues, axis=0)  # (ndepths, nh)

    # check the outputs
    assert jnp.allclose(zk, zk_true, atol=1e-6)

## helper functions ##
def dae_pendulum(vrdot: jnp.ndarray, vr: jnp.ndarray, t: jnp.ndarray, params) -> jnp.ndarray:
    # pendulum problem:
    # x' = u
    # y' = v
    # u' = -lambda * x
    # v' = -lambda * y - g
    # 0 = x ** 2 + y ** 2 - 1
    # vrdot, vr: both (5,)
    # x: (1,) is time
    # params: (g,)
    # returns: (5,)
    g = params
    x, y, u, v, T = jnp.split(vr, 5)
    xdot, ydot, udot, vdot, Tdot = jnp.split(vrdot, 5)
    f0 = xdot - u
    f1 = ydot - v
    f2 = udot + T * x
    f3 = vdot + T * y + g
    # you can select which constraints to be put, and each constraint corresponds to an index of the DAE
    # f4 = -Tdot - 3 * g * v  # index-0
    # f4 = u ** 2 + v ** 2 - T * (x ** 2 + y ** 2) - g * y  # index-1
    # f4 = x * u + y * v  # index-2
    f4 = x ** 2 + y ** 2 - 1  # index-3
    return jnp.concatenate([f0, f1, f2, f3, f4])

if __name__ == "__main__":
    test_solve_idae()
