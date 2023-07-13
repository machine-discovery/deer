from typing import Callable, Any, Optional, Tuple
import jax
import jax.numpy as jnp


def deer_iteration(
        inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        params: Any,
        xinput: jnp.ndarray,
        rsample_pts: jnp.ndarray,
        ybound: jnp.ndarray,
        yinit_guess: Optional[jnp.ndarray] = None,
        max_iter: int = 100,
        ) -> jnp.ndarray:
    """
    Perform the iteration from the DEER framework.

    Arguments
    ---------
    inv_lin: Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]
        Inverse of the linear operator.
        Takes the G-matrix (nsamples, ny, ny), the boundary condition (nbound_samples, ny),
        the right hand side of the equation (nsamples, ny), and the sampling locations (nsamples, nd).
        Returns the results of the inverse linear operator (nsamples, ny).
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        The non-linear function.
        Function that takes the y (output, (ny,)), x (input, (nx,)), and parameters (any structure).
        Returns the output of the function.
    params: Any
        The parameters of the function ``func``.
    xinput: jnp.ndarray
        The external input signal of shape (nsamples, nx)
    rsample_pts: jnp.ndarray
        The location of the sample points (nsamples, nd)
    ybound: jnp.ndarray
        The boundary condition of the output signal, shape (nbound_samples, ny)
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal (nsamples, ny).
        If None, it will be initialized as 0s.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations (nsamples, ny).
    """
    # set the default initial guess
    if yinit_guess is None:
        yinit_guess = jnp.zeros((xinput.shape[0], ybound.shape[-1]), dtype=xinput.dtype)

    # make the function works on additional (nsamples,) axis
    jacfunc = jax.vmap(jax.jacfwd(func, argnums=0), in_axes=(0, 0, None))
    func2 = jax.vmap(func, in_axes=(0, 0, None))

    def iter_func(err: jnp.ndarray, yt: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gt = -jacfunc(yt, xinput, params)  # (nsamples, ny, ny)
        rhs = func2(yt, xinput, params) + jnp.einsum("...ij,...j->...i", gt, yt)  # (nsamples, ny)
        yt_next = inv_lin(gt, ybound, rhs, rsample_pts)  # (nsamples, ny)
        err = jnp.max(jnp.abs(yt_next - yt))  # checking convergence
        return err, yt_next

    # iter_inp: (err, yt)
    def scan_func(iter_inp: Tuple[jnp.ndarray, jnp.ndarray], _):
        return jax.lax.cond(iter_inp[0] > 1e-7, iter_func, lambda *iter_inp: iter_inp, *iter_inp), None

    err = jnp.array(1e10, dtype=xinput.dtype)  # initial error should be very high
    (err, yt), _ = jax.lax.scan(scan_func, (err, yinit_guess), None, length=max_iter)
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
