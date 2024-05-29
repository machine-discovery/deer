from typing import Callable, Any, Tuple, List, Optional
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 4, 10, 11, 12))
def deer_mode2_iteration(
        lin: Callable[[jnp.ndarray, Any], jnp.ndarray],
        inv_lin: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, List[jnp.ndarray], Any, Any], jnp.ndarray],
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
    Perform the iteration from the DEER framework in solving the following equations
    
    .. math::

        f(L[y(r)], y(r-s_1), y(r-s_2), ..., y(r-s_p), x(r); \theta) = 0

    Arguments
    ---------
    lin: Callable[[jnp.ndarray, Any], jnp.ndarray]
        Operation of the linear operator :math:`L[y]`, by taking the input signal :math:`y` ``(nsamples, ny)``
        and the parameters from ``inv_lin_params``.
    inv_lin: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray]
        Inverse of the linear operator ``ML[y] + Gy = z``.
        Takes: (1) the Jacobian matrix of ``func`` w.r.t. :math:`L[y(r)]`,
        (2) the list of G-matrix (nsamples, ny, ny) (p-elements),
        (3) the right hand side of the equation (nsamples, ny), and
        (4) the ``inv_lin`` parameters in a tree (``inv_lin_params``).
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[jnp.ndarray, List[jnp.ndarray], Any, Any], jnp.ndarray]
        The non-linear function.
        Function that takes:
        (1) :math:`L[y(r)]`,
        (2) the list of shifted y [output: ``(ny,)``] (``p`` elements),
        (3) :math:`x` [input: ``(*nx)``] (in a pytree), and
        (4) parameters (any structure of pytree).
        Returns the output of the function.
    shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]]
        The function that shifts the input signal.
        It takes the signal of shape ``(nsamples, ny)`` and produces a list of shifted signals of shape
        ``(nsamples, ny)``.
    p_num: int
        Number of how many dependency on values of ``y`` at different places the function ``func`` has
    params: Any
        The parameters of the function ``func``.
    xinput: Any
        The external input signal of in a pytree with shape ``(nsamples, *nx)``.
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
    return deer_mode2_iteration_helper(
        lin=lin,
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


def deer_mode2_iteration_helper(
        lin: Callable[[jnp.ndarray, Any], jnp.ndarray],
        inv_lin: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, List[jnp.ndarray], Any, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        shifter_func_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    vmap_axes = (0, 0, 0, None)
    jacsfunc = jax.vmap(jax.jacfwd(func, argnums=(0, 1)), in_axes=vmap_axes)
    func2 = jax.vmap(func, in_axes=vmap_axes)

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-7 if dtype == jnp.float64 else 1e-4

    # def iter_func(err, yt, gt_, iiter):
    def iter_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) \
            -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, jacs_, iiter = iter_inp
        Ly = lin(yt, inv_lin_params)  # (nsamples, ny)
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        ytparams = shifter_func(yt, shifter_func_params)
        jacs = jacsfunc(Ly, ytparams, xinput, params)  # both (nsamples, ny, ny)
        jacLy, gts = jacs  # jacLy: (nsamples, ny, ny), gts: [p_num] + (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        yf = func2(Ly, ytparams, xinput, params)
        jy = jnp.einsum("...ij, ...j -> ...i", jacLy, Ly) + \
            sum([jnp.einsum("...ij, ...j -> ...i", gt, ytp) for gt, ytp in zip(gts, ytparams)])
        # _, jy = jax.jvp(func2_partialy, (Ly, ytparams), (Ly, ytparams))
        rhs = jax.tree_util.tree_map(lambda x, y: x - y, jy, yf)
        yt_next = inv_lin(jacLy, gts, rhs, inv_lin_params)  # (nsamples, ny)

        # workaround for rnn
        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, min=-clip, max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
            # jax.debug.print("{iiter}", iiter=iiter)
            # jax.debug.print("gteival: {gteival}", gteival=jnp.max(jnp.abs(jnp.real(jnp.linalg.eigvals(gts[0])))))

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next, (jacLy, gts), iiter + 1

    def cond_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    gt = jnp.zeros((yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]), dtype=dtype)
    gts = [gt] * p_num
    jacLy = gt
    jacs = (jacLy, gts)
    iiter = jnp.array(0, dtype=jnp.int32)
    err, yt, jacs, iiter = jax.lax.while_loop(cond_func, iter_func, (err, yinit_guess, jacs, iiter))
    # (err, yt, gts, iiter), _ = jax.lax.scan(scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter)
    if memory_efficient:
        jacs = None
    return yt, jacs


@deer_mode2_iteration.defjvp
def deer_mode2_iteration_jvp(
        # collect non-gradable inputs first
        lin: Callable[[jnp.ndarray, Any], jnp.ndarray],
        inv_lin: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        shifter_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        p_num: int,
        max_iter: int,
        memory_efficient: bool,
        clip_ytnext: bool,
        # the meaningful arguments
        primals, tangents):
    params, xinput, inv_lin_params, shifter_func_params, yinit_guess = primals
    grad_params, grad_xinput, grad_inv_lin_params, grad_shifter_func_params, grad_yinit_guess = tangents

    # compute the iteration
    # yt: (nsamples, ny)
    yt, jacs = deer_mode2_iteration_helper(
        lin=lin,
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

    # func2: (nsamples, ny) + (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    func2 = jax.vmap(func, in_axes=(0, 0, 0, None))  # vmap for y & x

    Ly = lin(yt, inv_lin_params)  # (nsamples, ny)
    ytparams = shifter_func(yt, shifter_func_params)
    if jacs is None:
        # recompute gts and jacLy
        jacsfunc = jax.vmap(jax.jacfwd(func, argnums=(0, 1)), in_axes=(0, 0, 0, None))
        jacs = jacsfunc(Ly, ytparams, xinput, params)  # both (nsamples, ny, ny)
    jacLy, gts = jacs

    # gts: [p_num] + (nsamples, ny, ny)

    # compute df (grad_func)
    func2_params_xinput = partial(func2, Ly, ytparams)
    _, grad_func = jax.jvp(func2_params_xinput, (xinput, params), (grad_xinput, grad_params))
    grad_func = jax.tree_util.tree_map(lambda x: -x, grad_func)

    # apply L_G^{-1} to the df
    rhs0 = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    inv_lin2 = partial(inv_lin, jacLy, gts)
    _, grad_yt = jax.jvp(inv_lin2, (rhs0, inv_lin_params), (grad_func, grad_inv_lin_params))
    # grad_yt = jax.tree_util.tree_map(lambda x: -x, grad_yt_neg)

    return yt, grad_yt
