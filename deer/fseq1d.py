from typing import Callable, Tuple, Optional, Any, List
from abc import abstractmethod
from deer.utils import get_method_meta, check_method
import jax
import jax.numpy as jnp
from deer.deer_iter import deer_iteration
from deer.maths import matmul_recursive
from deer.utils import Result


__all__ = ["seq1d"]

def seq1d(func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
          y0: jnp.ndarray, xinp: Any, params: Any,
          method: Optional["Seq1DMethod"] = None,
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
    method: Optional[Seq1DMethod]
        The method to solve the 1D sequence. If None, then use the ``DEER()`` method.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the discrete difference equation ``(nsamples, ny)``,
        excluding the initial states.
    
    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from fseq1d import seq1d
    
    >>> def func(y, x, params):
    ...     return y ** 2 + x * params[0]
    
    >>> y0 = jnp.array([0.0])
    >>> xinp = jnp.linspace(0, 1, 10).reshape(-1, 1)
    >>> params = jnp.array([0.5])
    
    >>> y = seq1d(func, y0, xinp, params, method=seq1d.Sequential())
    >>> y
    Array([[0.        ],
           [0.05555556],
           [0.11419753],
           [0.17970774],
           [0.2545171 ],
           [0.34255673],
           [0.45067845],
           [0.59199995],
           [0.79490839],
           [1.13187934]], dtype=float64)
    """
    if method is None:
        method = DEER()
    check_method(method, seq1d)
    return method.compute(func, y0, xinp, params)

class Seq1DMethod(metaclass=get_method_meta(seq1d)):
    @abstractmethod
    def compute(self, func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any):
        pass

class Sequential(Seq1DMethod):
    """
    Compute the 1D sequence with traditional sequential method.
    """
    def compute(self, func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any):
        # compute y[i] = f(y[i - 1], x[i]; params)
        # xinp: pytree, each has `(nsamples, *nx)`
        # y0: (ny,) the initial states
        # returns: (nsamples, ny), excluding the initial states
        def scan_fn(carry, x):
            yim1 = carry
            y = func(yim1, x, params)
            return y, y
        _, y = jax.lax.scan(scan_fn, y0, xinp)
        return Result(y)

class DEER(Seq1DMethod):
    """
    Compute the 1D sequential method using DEER method.

    Arguments
    ---------
    yinit_guess: Optional[jnp.ndarray]
        The initial guess of the output signal ``(nsamples, ny)``.
        If None, it will be initialized as all ``y0``.
    max_iter: int
        The maximum number of DEER iterations to perform.
    """
    def __init__(self, yinit_guess: Optional[jnp.ndarray] = None, max_iter: int = 10000):
        self.yinit_guess = yinit_guess
        self.max_iter = max_iter

    def compute(self, func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any):
        # set the default initial guess
        xinp_flat = jax.tree_util.tree_flatten(xinp)[0][0]
        yinit_guess = self.yinit_guess
        if yinit_guess is None:
            yinit_guess = jnp.zeros((xinp_flat.shape[0], y0.shape[-1]), dtype=xinp_flat.dtype) + y0  # (nsamples, ny)

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
        result = deer_iteration(
            inv_lin=self.seq1d_inv_lin, p_num=1, func=func2, shifter_func=shifter_func, params=params, xinput=xinp,
            inv_lin_params=(y0,), shifter_func_params=(y0,),
            yinit_guess=yinit_guess, max_iter=self.max_iter, clip_ytnext=True)
        return result

    def seq1d_inv_lin(self, gmat: List[jnp.ndarray], rhs: jnp.ndarray,
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
