from typing import Callable, Any, Tuple, List, Optional
from functools import partial
import jax
import jax.numpy as jnp
from deer.utils import Result, while_loop_scan
from deer.linesearch import LineSearch


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16))
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
        convergence_func: Optional[Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]] = None,
        max_iter: int = 100,
        clip_ytnext: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        linesearch: Optional[LineSearch] = None,
        lin_func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,  # optional, only needed for linesearch
        max_dev: Optional[float] = None,
        ) -> jnp.ndarray:
    r"""
    Perform the iteration from the DEER framework on equations with the form

    .. math::

        L[y(r)] = f(y(r-s_1), y(r-s_2), ..., y(r-s_p), x(r); \theta)

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
    convergence_func: Optional[Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]]
        The function to check the convergence of the iteration. This takes the same arguments as ``func``.
        If None, it will use the default convergence check.
    max_iter: int
        The maximum number of iterations to perform.
    clip_ytnext: bool
        Whether to clip the output of the next iteration to avoid inf and nans.
    atol: Optional[float]
        The absolute tolerance for the convergence. If None, it will be set to 1e-6 for float64 and 1e-4 for float32.
    rtol: Optional[float]
        The relative tolerance for the convergence. If None, it will be set to 1e-4 for float64 and 1e-3 for float32.

    Returns
    -------
    res: Result
        A ``Result`` object where ``.value`` is the output signal as the solution of the non-linear differential
        equations ``(nsamples, ny)`` and ``.success`` is the boolean array of the success status of the iterations.
    """
    yt, is_converged, _, _ = deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        convergence_func=convergence_func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        shifter_func_params=shifter_func_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        clip_ytnext=clip_ytnext,
        atol=atol,
        rtol=rtol,
        linesearch=linesearch,
        lin_func=lin_func,
        max_dev=max_dev,
    )
    return Result(yt, success=is_converged)

def deer_iteration_full(
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray], List[jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable as 0
        convergence_func: Optional[Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]] = None,
        max_iter: int = 100,
        clip_ytnext: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        linesearch: Optional[LineSearch] = None,
        lin_func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        max_dev: Optional[float] = None,
        ) -> jnp.ndarray:
    # this is like deer_iteration, but it also returns the intermediate results during the iterations
    # it can be used, for example, if one wants to optimize the process itself.
    # yt: (max_iter, nsamples, ny) and is_converged: (max_iter, 1, 1)
    ytiter, is_converged_iter, _, _ = deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        convergence_func=convergence_func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        shifter_func_params=shifter_func_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        clip_ytnext=clip_ytnext,
        return_full=True,
        atol=atol,
        rtol=rtol,
        linesearch=linesearch,
        lin_func=lin_func,
        max_dev=max_dev,
    )
    return Result(ytiter, success=is_converged_iter)

def deer_iteration_helper(
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        convergence_func: Optional[Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable
        max_iter: int = 100,
        clip_ytnext: bool = False,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        return_full: bool = False,
        linesearch: Optional[LineSearch] = None,
        lin_func: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        max_dev: Optional[float] = None,
        ) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    # tol = 1e-7 if dtype == jnp.float64 else 1e-4
    atol = (1e-6 if dtype == jnp.float64 else 1e-4) if atol is None else atol
    rtol = (1e-4 if dtype == jnp.float64 else 1e-3) if rtol is None else rtol

    def resid_func(yt: jnp.ndarray, all_params):
        xinput, shifter_func_params, params = all_params
        ytshift = shifter_func(yt, shifter_func_params)
        return lin_func(yt) - func2(ytshift, xinput, params)

    def iter_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) \
            -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, tol, yt, gt_, iiter = iter_inp
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        gts = [-gt for gt in jacfunc(ytparams, xinput, params)]  # [p_num] + (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        rhs = func2(ytparams, xinput, params)  # (carry, input, params) see train.py L41
        rhs += sum([jnp.einsum("...ij,...j->...i", gt, ytp) for gt, ytp in zip(gts, ytparams)])
        yt_next = inv_lin(gts, rhs, inv_lin_params)  # (nsamples, ny)

        dev0 = yt_next - yt
        dev = dev0
        if max_dev is not None:
            dev = jnp.clip(dev, min=-max_dev, max=max_dev)
            yt_next = yt + dev

        # workaround for rnn
        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, min=-clip, max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
            # jax.debug.print("{iiter}", iiter=iiter)
            # jax.debug.print("gteival: {gteival}", gteival=jnp.max(jnp.abs(jnp.real(jnp.linalg.eigvals(gts[0])))))

        # conditions for convergence checking
        err = jnp.abs(dev0)  # (nsamples, ny)
        tol = atol + rtol * jnp.abs(yt_next)

        if linesearch is not None:
            yt_next = linesearch.forward(yt, yt_next, resid_func, (xinput, shifter_func_params, params))
        # jax.debug.print("iiter: {iiter}, err: {err}", iiter=iiter, err=jnp.max(err))
        return err, tol, yt_next, gts, iiter + 1

    def cond_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) -> bool:
        err, tol, _, _, iiter = iter_inp
        return jnp.logical_and(jnp.any(err > tol), iiter < max_iter)

    tol = jnp.zeros_like(yinit_guess, dtype=dtype)
    err = tol + 1e10  # initial error should be very high
    gt = jnp.zeros((yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]), dtype=dtype)
    gts = [gt] * p_num
    iiter = jnp.array(0, dtype=jnp.int32)
    if return_full:
        # return all the intermediate values during the iterations as well
        (err, tol, yt, gts, iiter), (erriter, toliter, ytiter, _, _) = while_loop_scan(
            cond_func, iter_func, (err, tol, yinit_guess, gts, iiter), max_iter=max_iter)
        # (max_iter, 1, 1)
        is_converged_iter = jnp.all(erriter <= toliter, axis=(-1, -2), keepdims=True)
        if convergence_func is not None:
            def recalculate_func(convergence_func, yt, shifter_func_params, xinput, params, tol, is_converged_iter):
                # recalculate the convergence using the convergence function
                # yt: (max_iter, nsamples, ny)
                # xinput: (nsamples, *nx)
                # ytparams: [p] + (max_iter, nsamples, ny)
                ytparams = jax.vmap(shifter_func, in_axes=(0, None))(ytiter, shifter_func_params)
                cf = convergence_func
                cf = jax.vmap(cf, in_axes=(0, 0, None))  # broadcast to nsamples
                cf = jax.vmap(cf, in_axes=(0, None, None))  # broadcast to max_iter
                convergence_err = cf(ytparams, xinput, params)
                convergence = jnp.broadcast_to(
                    jnp.all(convergence_err <= atol, axis=-1, keepdims=True), convergence_err.shape)
                return convergence

            def broadcast_func(convergence_func, yt, shifter_func_params, xinput, params, tol, is_converged_iter):
                # just broadcast is_converged to the shape of tol
                return jnp.broadcast_to(is_converged_iter, tol.shape)

            conv_params = (jax.tree_util.Partial(convergence_func), ytiter, shifter_func_params, xinput, params,
                           toliter, is_converged_iter)
            is_converged_iter = jax.lax.cond(is_converged_iter[-1, 0, 0], broadcast_func, recalculate_func,
                                             *conv_params)

        return ytiter, is_converged_iter, gts, func
    else:
        # not using while_loop_scan here to keep it fast when vmapped
        # err, tol: (nsamples, ny)
        # yt: (nsamples, ny)
        err, tol, yt, gts, iiter = jax.lax.while_loop(cond_func, iter_func, (err, tol, yinit_guess, gts, iiter))
        # (err, yt, gts, iiter), _ = jax.lax.scan(scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter)
        is_converged = jnp.all(err <= tol)
        if convergence_func is not None:
            def recalculate_func(convergence_func, yt, shifter_func_params, xinput, params, tol, is_converged):
                # recalculate the convergence using the convergence function
                ytparams = shifter_func(yt, shifter_func_params)  # (nsamples, ny)
                convergence_err = jax.vmap(convergence_func, in_axes=(0, 0, None))(ytparams, xinput, params)
                convergence = jnp.broadcast_to(
                    jnp.all(convergence_err <= atol, axis=-1, keepdims=True), convergence_err.shape)
                return convergence

            def broadcast_func(convergence_func, yt, shifter_func_params, xinput, params, tol, is_converged):
                # just broadcast is_converged to the shape of tol
                return jnp.broadcast_to(is_converged, tol.shape)

            conv_params = (jax.tree_util.Partial(convergence_func), yt, shifter_func_params, xinput, params,
                           tol, is_converged)
            is_converged = jax.lax.cond(is_converged, broadcast_func, recalculate_func, *conv_params)
        # masking out the non-converged gts to avoid gradient become nan
        is_converged = jnp.broadcast_to(is_converged, yt.shape)  # (nsamples, ny)
        is_converged_mask = jnp.broadcast_to(is_converged[..., None], gts[0].shape)  # (nsamples, ny, ny)
        gts = [jnp.where(is_converged_mask, gt, jnp.eye(gt.shape[-1])) for gt in gts]
        return yt, is_converged, gts, func

@deer_iteration.defjvp
def deer_iteration_jvp(
        # collect non-gradable inputs first
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        convergence_func: Optional[Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]],
        max_iter: int,
        clip_ytnext: bool,
        atol: Optional[float],
        rtol: Optional[float],
        linesearch: Optional[LineSearch],
        lin_func: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
        max_dev: Optional[float],
        # the meaningful arguments
        primals, tangents):
    params, xinput, inv_lin_params, shifter_func_params, yinit_guess = primals
    grad_params, grad_xinput, grad_inv_lin_params, grad_shifter_func_params, grad_yinit_guess = tangents

    # compute the iteration
    # yt: (nsamples, ny)
    yt, is_converged, gts, func = deer_iteration_helper(
        inv_lin=inv_lin,
        func=func,
        shifter_func=shifter_func,
        convergence_func=convergence_func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        shifter_func_params=shifter_func_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        clip_ytnext=clip_ytnext,
        atol=atol,
        rtol=rtol,
        linesearch=linesearch,
        lin_func=lin_func,
        max_dev=max_dev,
        )

    # func2: (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    func2 = jax.vmap(func, in_axes=(0, 0, None))  # vmap for y & x

    ytparams = shifter_func(yt, shifter_func_params)
    # gts: [p_num] + (nsamples, ny, ny)

    # compute df (grad_func)
    func2_params_xinput = partial(func2, ytparams)
    _, grad_func = jax.jvp(func2_params_xinput, (xinput, params), (grad_xinput, grad_params))

    # apply L_G^{-1} to the df
    rhs0 = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    inv_lin2 = partial(inv_lin, gts)
    _, grad_yt = jax.jvp(inv_lin2, (rhs0, inv_lin_params), (grad_func, grad_inv_lin_params))

    # Create the tangent for is_converged
    is_converged_tangent = jnp.zeros_like(is_converged, dtype=jax.dtypes.float0)

    result = Result(yt, success=is_converged)
    grad_result = Result(grad_yt, success=is_converged_tangent)
    return result, grad_result
