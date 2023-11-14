from typing import Callable, Tuple, Optional, Any, List
import jax
import jax.numpy as jnp
from deer.deer_iter import deer_iteration
# 1D sequence: RNN or ODE


def solve_ivp(func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
              y0: jnp.ndarray, xinp: jnp.ndarray, params: Any,
              tpts: jnp.ndarray,
              yinit_guess: Optional[jnp.ndarray] = None,
              max_iter: int = 10000,
              memory_efficient: bool = False,
              ) -> jnp.ndarray:
    """
    Solve the initial value problem dy/dt = func(y, x, params) with y(0) = y0.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        Function to evaluate the derivative of y with respect to t. The
        arguments are: output signal y (ny,), input signal x (nx,), and parameters.
        The return value is the derivative of y with respect to t (ny,).
    y0: jnp.ndarray
        Initial condition on y (ny,).
    xinp: jnp.ndarray
        The external input signal of shape (nsamples, nx)
    params: Any
        The parameters of the function ``func``.
    tpts: jnp.ndarray
        The time points to evaluate the solution (nsamples,).
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    # set the default initial guess
    if yinit_guess is None:
        yinit_guess = jnp.zeros((xinp.shape[0], y0.shape[-1]), dtype=xinp.dtype)

    def func2(ylist: List[jnp.ndarray], x: jnp.ndarray, params: Any) -> jnp.ndarray:
        return func(ylist[0], x, params)

    def shifter_func(y: jnp.ndarray, params: Any) -> List[jnp.ndarray]:
        # y: (nsamples, ny)
        return [y]

    # perform the deer iteration
    inv_lin_params = (tpts, y0)
    yt = deer_iteration(
        inv_lin=solve_ivp_inv_lin, p_num=1, func=func2, shifter_func=shifter_func, params=params, xinput=xinp,
        inv_lin_params=inv_lin_params, shifter_func_params=(), yinit_guess=yinit_guess, max_iter=max_iter,
        memory_efficient=memory_efficient)
    return yt


def seq1d(func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
          y0: jnp.ndarray, xinp: Any, params: Any,
          yinit_guess: Optional[jnp.ndarray] = None,
          max_iter: int = 10000,
          memory_efficient: bool = False,
          ) -> jnp.ndarray:
    """
    Solve the discrete sequential equation, y[i + 1] = func(y[i], x[i], params) with the DEER framework.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray]
        Function to evaluate the next output signal y[i + 1] from the current output signal y[i].
        The arguments are: output signal y (ny,), input signal x (*nx,) in a pytree, and parameters.
        The return value is the next output signal y[i + 1] (ny,).
    y0: jnp.ndarray
        Initial condition on y (ny,).
    xinp: Any
        The external input signal in a pytree of shape (nsamples, *nx)
    params: Any
        The parameters of the function ``func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation (nsamples, ny),
        excluding the initial states.
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros((xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype)  # (nsamples, ny)

    def func2(ylist: List[jnp.ndarray], x: Any, params: Any) -> jnp.ndarray:
        # ylist: (ny,)
        return func(ylist[0], x, params)

    def shifter_func(y: jnp.ndarray, shifter_params: Any) -> List[jnp.ndarray]:
        # y: (nsamples, ny)
        # shifter_params = (y0,)
        y0, = shifter_params
        y = jnp.concatenate((y0[None, :], y[:-1, :]), axis=0)  # (nsamples, ny)
        return [y]

    # perform the deer iteration
    yt = deer_iteration(
        inv_lin=seq1d_inv_lin, p_num=1, func=func2, shifter_func=shifter_func, params=params, xinput=xinp,
        inv_lin_params=(y0,), shifter_func_params=(y0,),
        yinit_guess=yinit_guess, max_iter=max_iter, memory_efficient=memory_efficient, clip_ytnext=True)
    return yt


def solve_ivp_inv_lin(gmat: List[jnp.ndarray], rhs: jnp.ndarray,
                      inv_lin_params: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the initial value problem.
    dy/dt + G(t) y = rhs(t), y(0) = y0.

    Arguments
    ---------
    gmat: list of jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny, ny).
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray, jnp.ndarray]
        The parameters of the linear operator.
        The first element is the time points (nsamples,),
        and the second element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    # extract the parameters
    tpts, y0 = inv_lin_params
    gmat = gmat[0]  # (nsamples, ny, ny)

    eye = jnp.eye(gmat.shape[-1], dtype=gmat.dtype)  # (ny, ny)

    # taking the mid-point of gmat and rhs
    half_dt = 0.5 * (tpts[1:] - tpts[:-1])  # (nsamples - 1,)
    gtmid_dt = (gmat[1:] + gmat[:-1]) * half_dt[..., None, None]  # (nsamples - 1, ny, ny)
    htmid_dt = (rhs[1:] + rhs[:-1]) * half_dt[..., None]  # (nsamples - 1, ny)

    # get the matrices and vectors to be convolved
    gtmid_dt2 = gtmid_dt @ gtmid_dt  # (nt - 1, ny, ny)
    gtmid_dt3 = gtmid_dt @ gtmid_dt2  # (nt - 1, ny, ny)
    htbar_helper = eye - gtmid_dt / 2 + gtmid_dt2 / 6 - gtmid_dt3 / 24
    gtbar = (eye - htbar_helper @ gtmid_dt)  # (nt - 1, ny, ny) # approximate expm(-gtmid_dt)
    htbar = jnp.einsum("...ij,...j->...i", htbar_helper, htmid_dt)

    # compute the recursive matrix multiplication
    yt = matmul_recursive(gtbar, htbar, y0)  # (nt - 1, ny)
    return yt


def seq1d_inv_lin(gmat: List[jnp.ndarray], rhs: jnp.ndarray,
                  inv_lin_params: Tuple[jnp.ndarray]) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the discrete sequential equation.
    y[i + 1] + G[i] y[i] = rhs[i], y[0] = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The list of 1 G-matrix of shape (nsamples, ny, ny).
    rhs: jnp.ndarray
        The right hand side of the equation of shape (nsamples, ny).
    inv_lin_params: Tuple[jnp.ndarray]
        The parameters of the linear operator.
        The first element is the initial condition (ny,).

    Returns
    -------
    y: jnp.ndarray
        The solution of the linear equation of shape (nsamples, ny).
    """
    # extract the parameters
    y0, = inv_lin_params
    gmat = gmat[0]

    # compute the recursive matrix multiplication and drop the first element
    yt = matmul_recursive(-gmat, rhs, y0)[1:]  # (nsamples, ny)
    return yt


def binary_operator(element_i: Tuple[jnp.ndarray, jnp.ndarray],
                    element_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    a = gtj @ gti
    b = jnp.einsum("...ij,...j->...i", gtj, hti) + htj
    # clip = 1e9
    # a = jnp.clip(a, a_min=-clip, a_max=clip)
    # b = jnp.clip(b, a_min=-clip, a_max=clip)
    return a, b


def matmul_recursive(mats: jnp.ndarray, vecs: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
    """
    Perform the matrix multiplication recursively, y[i + 1] = mats[i] @ y[i] + vec[i].

    Arguments
    ---------
    mats: jnp.ndarray
        The matrices to be multiplied, shape (nsamples - 1, ny, ny)
    vecs: jnp.ndarray
        The vector to be multiplied, shape (nsamples - 1, ny)
    y0: jnp.ndarray
        The initial condition, shape (ny,)

    Returns
    -------
    result: jnp.ndarray
        The result of the matrix multiplication, shape (nsamples, ny)
    """
    # shift the elements by one index
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)[None]  # (1, ny, ny)
    first_elem = jnp.concatenate((eye, mats), axis=0)  # (nsamples, ny, ny)
    second_elem = jnp.concatenate((y0[None], vecs), axis=0)  # (nsamples, ny)

    # perform the scan
    elems = (first_elem, second_elem)
    _, yt = jax.lax.associative_scan(binary_operator, elems)
    # jax.debug.print("{nan} {inf}", nan=jnp.any(jnp.isnan(yt)), inf=jnp.any(jnp.isinf(yt)))
    return yt  # (nsamples, ny)
