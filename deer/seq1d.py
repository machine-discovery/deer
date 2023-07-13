from typing import Callable, Tuple, Optional, Union, Any
import jax
import jax.numpy as jnp
from deer.deer_iter import deer_iteration

# 1D sequence: RNN or ODE

def solve_ivp(func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
              y0: jnp.ndarray, xinp: jnp.ndarray, params: Any,
              tpts: jnp.ndarray,
              yinit_guess: Optional[jnp.ndarray] = None,
              max_iter: int = 100) -> jnp.ndarray:
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

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    # set the default initial guess
    if yinit_guess is None:
        yinit_guess = jnp.zeros((xinp.shape[0], y0.shape[-1]), dtype=xinp.dtype)

    # perform the deer iteration
    inv_lin_params = (tpts, y0)
    yt = deer_iteration(
        inv_lin=solve_ivp_inv_lin, func=func, params=params, xinput=xinp, inv_lin_params=inv_lin_params,
        yinit_guess=yinit_guess, max_iter=max_iter)
    return yt

def solve_ivp_inv_lin(gmat: jnp.ndarray, rhs: jnp.ndarray,
                      inv_lin_params: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    """
    Inverse of the linear operator for solving the initial value problem.
    dy/dt + G(t) y = rhs(t), y(0) = y0.

    Arguments
    ---------
    gmat: jnp.ndarray
        The G-matrix of shape (nsamples, ny, ny).
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

def binary_operator(element_i: Tuple[jnp.ndarray, jnp.ndarray],
                    element_j: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # associative operator for the scan
    gti, hti = element_i
    gtj, htj = element_j
    return gtj @ gti, jnp.einsum("...ij,...j->...i", gtj, hti) + htj

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
    return yt  # (nsamples, ny)
