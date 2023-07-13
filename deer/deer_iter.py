from typing import Callable, Any, Optional, Tuple
from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 5, 6))
def deer_iteration(
        inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        params: Any,  # gradable
        xinput: jnp.ndarray,  # gradable
        inv_lin_params: Any,  # gradable
        yinit_guess: jnp.ndarray,
        max_iter: int = 100,
        ) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework.

    Arguments
    ---------
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Inverse of the linear operator.
        Takes the G-matrix (nsamples, ny, ny), the right hand side of the equation (nsamples, ny), and
        the parameters in a tree.
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        The non-linear function.
        Function that takes the y (output, (ny,)), x (input, (nx,)), and parameters (any structure).
        Returns the output of the function.
    params: Any
        The parameters of the function ``func``.
    xinput: jnp.ndarray
        The external input signal of shape (nsamples, nx)
    inv_lin_params: tree structure of jnp.ndarray
        The parameters of the function ``inv_lin``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    return deer_iteration_helper(
        inv_lin=inv_lin, func=func, params=params, xinput=xinput, inv_lin_params=inv_lin_params,
        yinit_guess=yinit_guess, max_iter=max_iter)[0]

def deer_iteration_helper(
        inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        params: Any,  # gradable
        xinput: jnp.ndarray,  # gradable
        inv_lin_params: Any,  # gradable
        yinit_guess: jnp.ndarray,
        max_iter: int = 100,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, Callable]:
    # a helper function that returns all the residuals as well

    # make the function works on additional (nsamples,) axis
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    def iter_func(err: jnp.ndarray, yt: jnp.ndarray, gt_: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # gt_ is not used, but it is needed to return at the end of scan iteration
        gt = -jacfunc(yt, xinput, params)  # (nsamples, ny, ny)
        rhs = func2(yt, xinput, params) + jnp.einsum("...ij,...j->...i", gt, yt)  # (nsamples, ny)
        yt_next = inv_lin(gt, rhs, inv_lin_params)  # (nsamples, ny)
        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next, gt

    # iter_inp: (err, yt, gt)
    def scan_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _):
        return jax.lax.cond(iter_inp[0] > 1e-7, iter_func, lambda *iter_inp: iter_inp, *iter_inp), None

    err = jnp.array(1e10, dtype=xinput.dtype)  # initial error should be very high
    gt = jnp.zeros((xinput.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]), dtype=xinput.dtype)
    (err, yt, gt), _ = jax.lax.scan(scan_func, (err, yinit_guess, gt), None, length=max_iter)
    return yt, gt, func2

def deer_iteration_eval(
        inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        params: Any,  # gradable
        xinput: jnp.ndarray,  # gradable
        inv_lin_params: Any,  # gradable
        yinit_guess: Optional[jnp.ndarray] = None,
        max_iter: int = 100) -> jnp.ndarray:
    yt, gt, func2 = deer_iteration_helper(inv_lin, func, params, xinput, inv_lin_params, yinit_guess, max_iter)
    # the function must be wrapped as a partial to be used in the reverse mode
    return yt, (yt, gt, xinput, params, inv_lin_params, jax.tree_util.Partial(inv_lin), jax.tree_util.Partial(func2))

def deer_iteration_bwd(
        # collect non-gradable inputs first
        inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        yinit_guess: jnp.ndarray,
        max_iter: int,
        # the meaningful arguments
        resid: Any,
        grad_yt: jnp.ndarray):
    yt, gt, xinput, params, inv_lin_params, inv_lin, func = resid
    # gt: (nsamples, ny, ny)
    # func2: (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    rhs0 = jnp.zeros_like(gt[..., 0])  # (nsamples, ny)
    _, inv_lin_dual = jax.vjp(inv_lin, gt, rhs0, inv_lin_params)
    _, grad_rhs, grad_inv_lin_params = inv_lin_dual(grad_yt)
    # grad_rhs: (nsamples, ny)
    _, func_vjp = jax.vjp(func, yt, xinput, params)
    _, grad_xinput, grad_params = func_vjp(grad_rhs)
    return grad_params, grad_xinput, grad_inv_lin_params

deer_iteration.defvjp(deer_iteration_eval, deer_iteration_bwd)
