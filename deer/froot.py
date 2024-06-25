from typing import Callable, Any, Optional
from functools import partial
from abc import abstractmethod
import jax
import jax.numpy as jnp
from deer.utils import get_method_meta, check_method
from deer.utils import Result


def root(func: Callable[[jnp.ndarray, Any], jnp.ndarray], y0: jnp.ndarray, params: Any,
         method: Optional["RootMethod"] = None) -> Result:
    r"""
    Solve the root of the function,

    .. math::

        f(y; \theta) = 0

    Arguments
    ---------
    func: Callable[[jnp.ndarray, Any], jnp.ndarray]
        The function to find the root.
        The function that takes the current value of the root and the parameters.
    y0: jnp.ndarray
        The initial guess of the root.
    params: Any
        The parameters of the function.
    method: Optional[RootMethod]
        The method to solve the root. If None, then use the ``Newton()`` method.

    Returns
    -------
    res: Result
        The result of the root finding.
    """
    if method is None:
        method = Newton()
    check_method(method, root)
    return method.compute(func, y0, params)

class RootMethod(metaclass=get_method_meta(root)):
    @abstractmethod
    def compute(self, func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, params: Any):
        pass

class Newton(RootMethod):
    """
    Compute the root-finding method using Newton's method
    """
    def __init__(self, max_iter: int = 100, atol: float = 1e-6, rtol: float = 1e-3):
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol

    def compute(self, func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, params: Any):
        # y0: (ny,)
        # func: (ny,) -> (ny,)
        return newton_iter(func, y0, params, max_iter=self.max_iter, atol=self.atol, rtol=self.rtol)

@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5))
def newton_iter(func: Callable[[jnp.ndarray, Any], jnp.ndarray], y0: jnp.ndarray, params: Any,
                max_iter: int = 100, atol: float = 1e-6, rtol: float = 1e-3) -> Result:
    y, is_converged, jac = newton_iter_helper(
        func, y0, params, max_iter=max_iter, atol=atol, rtol=rtol)
    return Result(y, is_converged)

def newton_iter_helper(func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                       y0: jnp.ndarray,  # gradable as 0
                       params: Any,  # gradable
                       max_iter: int = 100,
                       atol: float = 1e-6,
                       rtol: float = 1e-3) -> Result:

    def iter_func(carry):
        y, err, tol, iiter, jac0 = carry
        jac = jax.jacfwd(func)(y, params)
        fy = func(y, params)
        ynext = y - jnp.linalg.solve(jac, fy)
        err = jnp.abs(ynext - y)
        tol = atol + rtol * jnp.abs(ynext)
        iiter += 1
        return ynext, err, tol, iiter, jac

    def cond_func(carry):
        y, err, tol, iiter, jac0 = carry
        return jnp.logical_and(jnp.any(err > tol), iiter < max_iter)

    err = jnp.full_like(y0, jnp.inf)
    tol = jnp.zeros_like(y0)
    iiter = jnp.array(0, dtype=jnp.int32)
    jac0 = jnp.zeros((y0.size, y0.size))
    y, err, tol, iiter, jac = jax.lax.while_loop(cond_func, iter_func, (y0, err, tol, iiter, jac0))
    is_converged = iiter < max_iter
    return y, is_converged, jac

@newton_iter.defjvp
def newton_iter_jvp(
        # collect non-gradable input first
        func: Callable[[jnp.ndarray, Any], jnp.ndarray],
        max_iter: int,
        atol: float,
        rtol: float,
        # meaningful arguments
        primals, tangents):
    y0, params = primals
    _, grad_params = tangents

    # compute the iterations
    yt, is_converged, jac = newton_iter_helper(
        func, y0, params, max_iter=max_iter, atol=atol, rtol=rtol)
    
    # compute grad of f
    func_partial_y = partial(func, yt)
    # grad_func: (ny,)
    _, grad_func = jax.jvp(func_partial_y, (params,), (grad_params,))
    
    grad_y = jnp.linalg.solve(jac, -grad_func)  # (ny,)
    result = Result(yt, is_converged)
    grad_result = Result(grad_y)
    return result, grad_result
