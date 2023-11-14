from typing import Callable, Any, Tuple, List, Optional
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 9, 10, 11))
def deer_iteration(
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray], List[jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable as 0
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework.

    Arguments
    ---------
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray]
        Inverse of the linear operator.
        Takes the list of G-matrix (nsamples, ny, ny) (p-elements),
        the right hand side of the equation (nsamples, ny), and the inv_lin parameters in a tree.
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]
        The non-linear function.
        Function that takes the list of y [output: (ny,)] (p elements), x [input: (*nx)] (in a pytree),
        and parameters (any structure of pytree).
        Returns the output of the function.
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]]
        The function that shifts the input signal.
        It takes the signal of shape (nsamples, ny) and produces a list of shifted signals of shape (nsamples, ny).
    p_num: int
        Number of how many dependency on values of ``y`` at different places the function ``func`` has
    params: Any
        The parameters of the function ``func``.
    xinput: Any
        The external input signal of in a pytree with shape (nsamples, *nx).
    inv_lin_params: tree structure of jnp.ndarray
        The parameters of the function ``inv_lin``.
    shifter_func_params: tree structure of jnp.ndarray
        The parameters of the function ``shifter_func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    memory_efficient: bool
        If True, then do not save the Jacobian matrix for the backward pass.
        This can save memory, but the backward pass will be slower due to recomputation of
        the Jacobian matrix.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    # TODO: handle the batch size in the implementation, because vmapped lax.cond is converted to lax.select
    # which is less efficient than lax.cond
    return deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        shifter_func_params=shifter_func_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=clip_ytnext)[0]


def deer_iteration_helper(
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4

    # def iter_func(err, yt, gt_, iiter):
    def iter_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) \
            -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [-gt for gt in jacfunc(ytparams, xinput, params)]  # [p_num] + (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) see train.py L41
        rhs += sum([jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)])
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        # workaround for rnn
        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, a_min=-clip, a_max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
            # jax.debug.print("{iiter}", iiter=iiter)
            # jax.debug.print("gteival: {gteival}", gteival=jnp.max(jnp.abs(jnp.real(jnp.linalg.eigvals(gts[0])))))

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        # jax.debug.print("iiter: {iiter}, err: {err}", iiter=iiter, err=err)
        return err, yt_next, gts, iiter + 1

    def cond_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros((yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]), dtype=dtype)
    gts = [gt] * p_num
    iiter = jnp.array(0, dtype=jnp.int32)
    err, yt, gts, iiter = jax.lax.while_loop(cond_func, iter_func, (err, yinit_guess, gts, iiter))
    # (err, yt, gts, iiter), _ = jax.lax.scan(scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter)
    if memory_efficient:
        gts = None
    return yt, gts, func


def deer_iteration_eval(
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable as 0
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> jnp.ndarray:
    # compute the iteration
    yt, gts, func = deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        shifter_func_params=shifter_func_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=clip_ytnext,
        )
    # the function must be wrapped as a partial to be used in the reverse mode
    resid = (yt, gts, xinput, params, inv_lin_params, shifter_func_params,
             jax.tree_util.Partial(inv_lin), jax.tree_util.Partial(func),
             jax.tree_util.Partial(shifter_func),
             )
    return yt, resid


def deer_iteration_bwd(
        # collect non-gradable inputs first
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        max_iter: int,
        memory_efficient: bool,
        clip_ytnext: bool,
        # the meaningful arguments
        resid: Any,
        grad_yt: jnp.ndarray):
    yt, gts, xinput, params, inv_lin_params, shifter_func_params, inv_lin, func, shifter_func = resid
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    if gts is None:
        jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
        # recompute gts
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [-gt for gt in jacfunc(ytparams, xinput, params)]

    # gts: [p_num] + (nsamples, ny, ny)
    # func2: (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    rhs0 = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    _, inv_lin_dual = jax.vjp(inv_lin, gts, rhs0, inv_lin_params)
    _, grad_rhs, grad_inv_lin_params = inv_lin_dual(grad_yt)
    # grad_rhs: (nsamples, ny)
    ytparams = shifter_func(yt, shifter_func_params)
    _, func_vjp = jax.vjp(func2, ytparams, xinput, params)
    _, grad_xinput, grad_params = func_vjp(grad_rhs)
    # TODO: think about how to compute the gradient of the shifter_func_params?
    grad_shifter_func_params = None
    return grad_params, grad_xinput, grad_inv_lin_params, grad_shifter_func_params, None


deer_iteration.defvjp(deer_iteration_eval, deer_iteration_bwd)
