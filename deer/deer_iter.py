from typing import Callable, Any, Tuple, List, Optional
from functools import partial
import jax
import jax.numpy as jnp

from typing import Callable, Any, Tuple, List, Optional
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 9, 10, 11))
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
    memory_efficient: bool
        If True, then do not save the Jacobian matrix for the backward pass.
        This can save memory, but the backward pass will be slower due to recomputation of
        the Jacobian matrix.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    
    Example
    ---
    >>> import jax
    >>> import jax.numpy as jnp

    >>> def lin_func(yt, inv_lin_params):
    ...     return [yt]

    >>> def inv_lin(jacLs, rhs, inv_lin_params):
    ...     return jnp.linalg.solve(jacLs[0], rhs[..., None])[..., 0]

    >>> def func(Lys, xinput, params):
    ...     return Lys[0] + xinput - params

    >>> yinit_guess = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    >>> params = jnp.array(0.5, dtype=jnp.float32)
    >>> xinput = jnp.array([[0.1], [0.2]], dtype=jnp.float32)
    >>> inv_lin_params = None
    >>> p_num = 1

    >>> result = deer_mode2_iteration(
    ...     lin_func=lin_func,
    ...     inv_lin=inv_lin,
    ...     func=func,
    ...     p_num=p_num,
    ...     params=params,
    ...     xinput=xinput,
    ...     inv_lin_params=inv_lin_params,
    ...     yinit_guess=yinit_guess,
    ...     max_iter=100,
    ...     memory_efficient=False,
    ...     clip_ytnext=False
    ... )

    >>> print(result)
    [[0.4]
     [0.3]]
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
        yinit_guess: jnp.ndarray,  # gradable
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
            yt_next = jnp.clip(yt_next, min=-clip, max=clip)
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


@deer_iteration.defjvp
def deer_iteration_jvp(
        # collect non-gradable inputs first
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
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

    # func2: (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    func2 = jax.vmap(func, in_axes=(0, 0, None))  # vmap for y & x

    ytparams = shifter_func(yt, shifter_func_params)
    if gts is None:
        jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
        # recompute gts
        gts = [-gt for gt in jacfunc(ytparams, xinput, params)]
    # gts: [p_num] + (nsamples, ny, ny)

    # compute df (grad_func)
    func2_params_xinput = partial(func2, ytparams)
    _, grad_func = jax.jvp(func2_params_xinput, (xinput, params), (grad_xinput, grad_params))

    # apply L_G^{-1} to the df
    rhs0 = jnp.zeros_like(gts[0][..., 0])  # (nsamples, ny)
    inv_lin2 = partial(inv_lin, gts)
    _, grad_yt = jax.jvp(inv_lin2, (rhs0, inv_lin_params), (grad_func, grad_inv_lin_params))

    return yt, grad_yt


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3, 8, 9, 10))
def deer_mode2_iteration(
        lin_func: Callable[[jnp.ndarray, Any], jnp.ndarray],
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable as 0
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework in solving the following equations
    
    .. math::

        f(L_1[y(r)], L_2[y(r)], ..., L_p[y(r)], x(r); \theta) = 0

    Although the original DEER can be captured with this form, the original DEER might be slightly faster
    because there is a term that can analytically be eliminated, removing the need to doing the computation.
    Unless JAX can optimize this, the original DEER might be preferred for problems that fit within the form.

    Arguments
    ---------
    lin_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]]
        Operation of the linear operators :math:`L*[y]`, by taking the input signal :math:`y` ``(nsamples, ny)``
        and the parameters from ``inv_lin_params`` and then returns the output of linearly transformed :math:`y`.
    inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray]
        Inverse of the linear operator ``ML[y] + Gy = z``.
        Takes:
        (1) the list of Jacobian matrices of ``func`` w.r.t. :math:`L_*[y(r)]` (p-elements, 
            each ``(nsamples, ny, ny)``),
        (2) the right hand side of the equation of shape ``(nsamples, ny)``, and
        (3) the ``inv_lin`` parameters in a tree (``inv_lin_params``).
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray]
        The non-linear function.
        Function that takes:
        (1) the list of linearly transformed output values :math:`L_*[y(r)]`,
        (2) :math:`x` [input: ``(*nx)``] (in a pytree), and
        (3) parameters (any structure of pytree).
        Returns the output of the function.
    p_num: int
        Number of how many dependency on values of :math:`L_*[y]` at different places the function ``func`` has.
    params: Any
        The parameters of the function ``func``.
    xinput: Any
        The external input signal of in a pytree with shape ``(nsamples, *nx)``.
    inv_lin_params: tree structure of jnp.ndarray
        The parameters of the function ``inv_lin``.
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
    return deer_mode2_iteration_helper(
        lin_func=lin_func,
        inv_lin=inv_lin,
        func=func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=clip_ytnext)[0]


def deer_mode2_iteration_helper(
        lin_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        inv_lin: Callable[[jnp.ndarray, List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], Any, Any], jnp.ndarray],
        p_num: int,
        params: Any,  # gradable
        xinput: Any,  # gradable
        inv_lin_params: Any,  # gradable
        yinit_guess: jnp.ndarray,  # gradable
        max_iter: int = 100,
        memory_efficient: bool = False,
        clip_ytnext: bool = False,
        ) -> Tuple[jnp.ndarray, Optional[List[jnp.ndarray]], Callable]:
    # obtain the functions to compute the jacobians and the function
    vmap_axes = (0, 0, None)
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=vmap_axes)
    func2 = jax.vmap(func, in_axes=vmap_axes)

    dtype = yinit_guess.dtype
    # set the tolerance to be 1e-4 if dtype is float32, else 1e-7 for float64
    tol = 1e-6 if dtype == jnp.float64 else 1e-4

    # def iter_func(err, yt, gt_, iiter):
    def iter_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) \
            -> Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]:
        err, yt, _, iiter = iter_inp
        Lys = lin_func(yt, inv_lin_params)  # (nsamples, ny)
        # gt_ is not used, but it is needed to return at the end of scan iteration
        # yt: (nsamples, ny)
        jacLs = jacfunc(Lys, xinput, params)  # both (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        yf = func2(Lys, xinput, params)
        jy = sum([jnp.einsum("...ij, ...j -> ...i", jacL, Ly) for jacL, Ly in zip(jacLs, Lys)])
        # _, jy = jax.jvp(func2_partialy, (Ly, ytparams), (Ly, ytparams))
        rhs = jax.tree_util.tree_map(lambda x, y: x - y, jy, yf)
        yt_next = inv_lin(jacLs, rhs, inv_lin_params)  # (nsamples, ny)

        # workaround to avoid nans
        if clip_ytnext:
            clip = 1e8
            yt_next = jnp.clip(yt_next, min=-clip, max=clip)
            yt_next = jnp.where(jnp.isnan(yt_next), 0.0, yt_next)
            # jax.debug.print("{iiter}", iiter=iiter)
            # jax.debug.print("gteival: {gteival}", gteival=jnp.max(jnp.abs(jnp.real(jnp.linalg.eigvals(gts[0])))))

        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        # jax.debug.print("iiter: {iiter}, err: {err}", iiter=iiter, err=err)
        return err, yt_next, jacLs, iiter + 1

    def cond_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray, List[jnp.ndarray], jnp.ndarray]) -> bool:
        err, _, _, iiter = iter_inp
        return jnp.logical_and(err > tol, iiter < max_iter)

    err = jnp.array(1e10, dtype=dtype)  # initial error should be very high
    jacL = jnp.zeros((yinit_guess.shape[0], yinit_guess.shape[-1], yinit_guess.shape[-1]), dtype=dtype)
    jacLs = [jacL] * p_num
    iiter = jnp.array(0, dtype=jnp.int32)
    err, yt, jacLs, iiter = jax.lax.while_loop(cond_func, iter_func, (err, yinit_guess, jacLs, iiter))
    # (err, yt, gts, iiter), _ = jax.lax.scan(scan_func, (err, yinit_guess, gts, iiter), None, length=max_iter)
    if memory_efficient:
        jacLs = None
    return yt, jacLs


@deer_mode2_iteration.defjvp
def deer_mode2_iteration_jvp(
        # collect non-gradable inputs first
        lin_func: Callable[[jnp.ndarray, Any], List[jnp.ndarray]],
        inv_lin: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        func: Callable[[List[jnp.ndarray], jnp.ndarray, Any], jnp.ndarray],
        p_num: int,
        max_iter: int,
        memory_efficient: bool,
        clip_ytnext: bool,
        # the meaningful arguments
        primals, tangents):
    params, xinput, inv_lin_params, yinit_guess = primals
    grad_params, grad_xinput, grad_inv_lin_params, grad_yinit_guess = tangents

    # compute the iteration
    # yt: (nsamples, ny)
    yt, jacLs = deer_mode2_iteration_helper(
        lin_func=lin_func,
        inv_lin=inv_lin,
        func=func,
        p_num=p_num,
        params=params,
        xinput=xinput,
        inv_lin_params=inv_lin_params,
        yinit_guess=yinit_guess,
        max_iter=max_iter,
        memory_efficient=memory_efficient,
        clip_ytnext=clip_ytnext,
    )

    # func2: (nsamples, ny) + (nsamples, ny) + (nsamples, ny) + any -> (nsamples, ny)
    vmap_axes = (0, 0, None)
    func2 = jax.vmap(func, in_axes=vmap_axes)  # vmap for y & x

    Lys = lin_func(yt, inv_lin_params)  # [p_num] + (nsamples, ny)
    if jacLs is None:
        # recompute jacLs
        jacsfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=vmap_axes)
        jacLs = jacsfunc(Lys, xinput, params)  # (nsamples, ny, ny)

    # gts: [p_num] + (nsamples, ny, ny)

    # compute df (grad_func)
    func2_params_xinput = partial(func2, Lys)
    _, grad_func = jax.jvp(func2_params_xinput, (xinput, params), (grad_xinput, grad_params))
    grad_func = jax.tree_util.tree_map(lambda x: -x, grad_func)

    # apply L_G^{-1} to the df
    rhs0 = jnp.zeros_like(Lys[0])  # (nsamples, ny)
    inv_lin2 = partial(inv_lin, jacLs)
    _, grad_yt = jax.jvp(inv_lin2, (rhs0, inv_lin_params), (grad_func, grad_inv_lin_params))
    # grad_yt = jax.tree_util.tree_map(lambda x: -x, grad_yt_neg)

    return yt, grad_yt
