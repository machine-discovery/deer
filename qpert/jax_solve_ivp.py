from typing import Callable, Optional, Tuple, Sequence, Any
from functools import partial
import jax
import jax.numpy as jnp


def mv(mat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    # matrix vector multiplication
    return jnp.einsum("...ab, ...b -> ...a", mat, vec)

def binary_operator(element_i: Tuple[jnp.ndarray, jnp.ndarray],
                    element_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    return gtj @ gti, mv(gtj, hti) + htj

def conv_gt(gt: jnp.ndarray, ht: jnp.ndarray, y0: jnp.ndarray,
            reverse: bool = False) -> jnp.ndarray:
    # gt: (nt - 1, ny, ny) | ht: (nt - 1, ny) | y0: (ny,)
    # returns: (nt, ny)
    # shift the elements by one index
    eye = jnp.eye(gt.shape[-1], dtype=gt.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, gt) if not reverse else (gt, eye), axis=0)  # (nt, ny, ny)
    second_elem = jnp.concatenate((y0[None], ht) if not reverse else (ht, y0[None]), axis=0)  # (nt, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems, reverse=reverse)
    return yt

@partial(jax.jit, static_argnums=(0,))
# @jax.custom_vjp
def solve_ivp(func: Callable[[jnp.ndarray, jnp.ndarray, Sequence[jnp.ndarray]], jnp.ndarray], tpts: jnp.ndarray,
              y0: jnp.ndarray, params: Sequence[jnp.ndarray] = [],
              yt_init: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    # func: (t, y, params) -> dy/dt with shape ((,), (ny,), ...) -> (ny,)
    # tpts: (nt,) | y0: (ny,)
    # returns: (nt, ny)
    ny = y0.shape[-1]
    nt = tpts.shape[-1]
    dtype = tpts.dtype

    # get the initial guess of yt
    yt = jnp.zeros((nt, ny), dtype=dtype) if yt_init is None else yt_init
    dt = tpts[1:] - tpts[:-1]  # (nt - 1,)

    # get the function to obtain the jacobian matrix
    jac_func_t = jax.vmap(jax.jacfwd(func, argnums=1), in_axes=(0, 0, None))  # func with output (nt, ny, ny)
    func_t = jax.vmap(func, in_axes=(0, 0, None))
    eye = jnp.eye(ny, dtype=yt.dtype)  # (1, ny, ny)

    def iter_fun(err: jnp.ndarray, yt: jnp.ndarray,
                 resid: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # compute the terms to be convolved
        ft = func_t(tpts, yt, params)  # (nt, ny)
        jact = jac_func_t(tpts, yt, params)  # (nt, ny, ny)
        ht = ft - mv(jact, yt)  # (nt, ny)
        gt = -jact

        # get the mid value to increase the accuracy
        gtmid = 0.5 * (gt[:-1] + gt[1:])  # (nt - 1, ny, ny)
        htmid = 0.5 * (ht[:-1] + ht[1:])  # (nt - 1, ny)

        # get the matrices and vectors to be convolved
        gtbar = jax.scipy.linalg.expm(-gtmid * dt[..., None, None])  # (nt - 1, ny, ny)
        htbar = jnp.linalg.solve(gtmid, mv(eye - gtbar, htmid)[..., None])[..., 0]  # (nt - 1, ny)

        # compute the convolution
        yt_next = conv_gt(gtbar, htbar, y0)  # (nt, ny)

        # check the convergence
        err = jnp.max(jnp.abs(yt - yt_next))
        yt = yt_next
        print(err)
        resid = (gtbar, tpts)
        return err, yt, resid

    # iter_inp: (err, yt, resid)
    def scan_fun(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]], _):
        return jax.lax.cond(iter_inp[0] > 1e-7, iter_fun, lambda *iter_inp: iter_inp, *iter_inp), None

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    # resid: (gtbar, tpts)
    resid = (jnp.zeros((nt - 1, ny, ny), dtype=dtype), tpts)
    (err, yt, resid), _ = jax.lax.scan(scan_fun, (err, yt, resid), None, length=100)
    # return yt, resid
    return yt

def solve_ivp_fwd(func: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray], tpts: jnp.ndarray,
                  y0: jnp.ndarray, params: Sequence[jnp.ndarray] = [],
                  yt_init: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    yt, resid = solve_ivp(func, tpts, y0, yt_init)
    return yt, resid + (func,)

def solve_ivp_bwd(resid: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Callable],
                  grad_yt: jnp.ndarray) -> Tuple[None, jnp.ndarray, jnp.ndarray, Sequence[jnp.ndarray], None]:
    gtbar, tpts, func = resid
    # gtbar: (nt - 1, ny, ny) | tpts: (nt,) | func: (t, y, params) -> dy/dt with shape ((,), (ny,), ...) -> (ny,)
    # grad_yt: (nt, ny)
    nt, ny = grad_yt.shape
    dtype = grad_yt.dtype

    # get the gt-convolution of grad_yt
    grad_yt_mid = 0.5 * (grad_yt[:-1] + grad_yt[1:])  # (nt - 1, ny)
    grad_ytbar = conv_gt(gtbar, grad_yt_mid, jnp.zeros((ny,), dtype=dtype), reverse=True)  # (nt, ny)

if __name__ == "__main__":
    import time
    import numpy as np
    from scipy.integrate import solve_ivp as solve_ivp_scipy
    import matplotlib.pyplot as plt

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def func(t: jnp.ndarray, y: jnp.ndarray, params=[]) -> jnp.ndarray:
        # t: (,)
        # y: (ny,)
        # returns: (ny,)
        dfdy = -6 * y - 10 * y ** 3 + 30 * jnp.sin(600 * t) + 3
        return dfdy
    
    def func_np(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        dfdy = -6 * y - 10 * y ** 3 + 30 * np.sin(600 * t) + 3
        return dfdy

    npts = 1000000
    ny = 1
    dtype = jnp.float64
    tpts = jnp.linspace(0, 1, npts, dtype=dtype)  # (ntpts,)
    tpts2 = tpts * 10
    y0 = jnp.zeros((ny,), dtype=dtype)

    # warm-up jax
    yt = solve_ivp(func, tpts, y0)
    jax.block_until_ready(yt)
    yt = solve_ivp(func, tpts, y0)
    jax.block_until_ready(yt)
    print(yt.shape)

    t0 = time.time()
    yt = solve_ivp(func, tpts2, y0)
    jax.block_until_ready(yt)
    t1 = time.time()
    print("JAX:", t1 - t0)

    # warm-up scipy
    tpts_np = np.array(tpts)
    y0_np = np.array(y0)
    tpts_np = np.array(tpts)
    tpts2_np = np.array(tpts2)
    yt_np = solve_ivp_scipy(func_np, (tpts_np[0], tpts_np[-1]), y0_np, t_eval=tpts_np)
    yt_np = solve_ivp_scipy(func_np, (tpts_np[0], tpts_np[-1]), y0_np, t_eval=tpts_np)

    t2 = time.time()
    # need higher accuracy to get the same result
    yt_np = solve_ivp_scipy(func_np, (tpts2_np[0], tpts2_np[-1]), y0_np, t_eval=tpts2_np, atol=1e-7, rtol=1e-7)
    t3 = time.time()
    print("Numpy:", t3 - t2)
    print("Speedup:", (t3 - t2) / (t1 - t0))

    # plt.plot(tpts, yt[..., 0])
    # plt.plot(tpts_np, yt_np.y[0])
    plt.plot(tpts_np, yt_np.y[0] - yt[..., 0])
    plt.savefig("test.png")
    plt.close()
