from typing import Callable, Tuple, Optional, Any, List
import jax
import jax.numpy as jnp
from deer.deer_iter import deer_iteration
from deer.maths import matmul_recursive

__all__ = ["seq1d"]

def seq1d(func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
          y0: jnp.ndarray, xinp: Any, params: Any,
          yinit_guess: Optional[jnp.ndarray] = None,
          max_iter: int = 10000,
          memory_efficient: bool = True,
          ) -> jnp.ndarray:
    r"""
    Solve the discrete sequential equation

    .. math::

        y_{i + 1} = f(y_i, x_i; \theta)

    where :math:`f` is a non-linear function, :math:`y_i` is the output signal at time :math:`i`,
    :math:`x_i` is the input signal at time :math:`i`, and :math:`\theta` are the parameters of the function.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray]
        Function to evaluate the next output signal :math:`y_{i+1}` from the current output signal :math:`y_i`.
        The arguments are: signal :math:`y` at the current time ``(ny,)``, input signal :math:`x` at the current time
        ``(*nx,)`` in a pytree, and parameters :math:`\theta` in a pytree.
        The return value is the next output signal :math:`y` at the next time ``(ny,)``.
    y0: jnp.ndarray
        Initial condition on :math:`y` ``(ny,)``.
    xinp: Any
        The external input signal in a pytree of shape ``(nsamples, *nx)``
    params: Any
        The parameters of the function ``func``.
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal ``(nsamples, ny)``.
        If ``None``, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation ``(nsamples, ny)``,
        excluding the initial states.
    """
    # set the default initial guess
    xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
    if yinit_guess is None:
        yinit_guess = jnp.zeros((xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype)  # (nsamples, ny)

    def func2(yshifts: List[jnp.ndarray], x: Any, params: Any) -> jnp.ndarray:
        # yshifts: (ny,)
        return func(yshifts[0], x, params)

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
