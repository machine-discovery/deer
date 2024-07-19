from typing import Callable, Any, Optional
from functools import partial
from abc import abstractmethod
import jax
import jax.numpy as jnp
from deer.utils import get_method_meta, check_method, while_loop_scan, Result


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
    @property
    @abstractmethod
    def num_iter_returned(self) -> int:
        # return the maximum number of iterations if the returned values include the intermediate values during
        # iterations or 0 if it just returns the last iteration value
        pass

    @abstractmethod
    def compute(self, func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, params: Any):
        pass

class Newton(RootMethod):
    """
    Compute the root-finding method using Newton's method.

    Arguments
    ---------
    max_iter: int
        The maximum number of iterations.
    atol: float
        The absolute tolerance for convergence.
    rtol: float
        The relative tolerance for convergence.
    return_full: bool
        If True, return the full iterations of the root-finding process. Returned shape will be ``(max_iter, *ny)``.
        If False, return only the last iteration. Returned shape will be the same as ``y0.shape: (*ny,)``.
    """
    def __init__(self, max_iter: int = 100, atol: float = 1e-6, rtol: float = 1e-3,
                 return_full: bool = False):
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.return_full = return_full

    @property
    def num_iter_returned(self) -> bool:
        return self.max_iter if self.return_full else 0

    def compute(self, func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, params: Any):
        # y0: (*ny,)
        # func: (*ny,) -> (*ny,)
        # returns: y: (*ny,) and is_converged: (*ny,) bool if not self.return_full
        # else returns yiter: (max_iter, *ny) and is_converged_iter: (max_iter, *ny)
        if self.return_full:
            return newton_iter_full(func, y0, params, max_iter=self.max_iter, atol=self.atol, rtol=self.rtol)
        else:
            return newton_iter(func, y0, params, max_iter=self.max_iter, atol=self.atol, rtol=self.rtol)

@partial(jax.custom_jvp, nondiff_argnums=(0, 3, 4, 5))
def newton_iter(func: Callable[[jnp.ndarray, Any], jnp.ndarray], y0: jnp.ndarray, params: Any,
                max_iter: int = 100, atol: float = 1e-6, rtol: float = 1e-3) -> Result:
    # y0: (*ny,), returns y: (*ny,) and is_converged: (*ny,) bool
    # the gradient is obtained by using the implicit function theorem
    (y, is_converged), _, _ = newton_iter_helper(
        func, y0, params, max_iter=max_iter, atol=atol, rtol=rtol)
    return Result(y, is_converged)

def newton_iter_full(func: Callable[[jnp.ndarray, Any], jnp.ndarray], y0: jnp.ndarray, params: Any,
                max_iter: int = 100, atol: float = 1e-6, rtol: float = 1e-3) -> Result:
    # y0: (*ny,), returns yiter: (max_iter, *ny) and is_converged_iter: (max_iter, *ny)
    # the gradient is obtained by propagating through the iterations
    _, (yiter, is_converged_iter), _ = newton_iter_helper(
        func, y0, params, max_iter=max_iter, atol=atol, rtol=rtol)
    return Result(yiter, is_converged_iter)

def newton_iter_helper(func: Callable[[jnp.ndarray, Any], jnp.ndarray],
                       y0: jnp.ndarray,  # gradable as 0
                       params: Any,  # gradable
                       max_iter: int = 100,
                       atol: float = 1e-6,
                       rtol: float = 1e-3):

    def iter_func(carry):
        y, err, tol, iiter, _ = carry
        jac = jax.jacfwd(func)(y, params)
        fy = func(y, params)
        jacinvfy = jnp.linalg.solve(jac, fy)
        # doing lstsq to handle singular matrix
        jacinvfy = jax.lax.cond(jnp.all(jnp.isfinite(jacinvfy)), lambda : jacinvfy, lambda : jnp.linalg.lstsq(jac, fy)[0])
        ynext = y - jacinvfy
        # ynext = y - jnp.linalg.lstsq(jac, fy)[0]

        # clip nans and infs
        clip = 1e8
        ynext = jnp.clip(ynext, min=-clip, max=clip)
        ynext = jnp.where(jnp.isnan(ynext), 0.0, ynext)

        err = jnp.abs(ynext - y)
        tol = atol + rtol * jnp.abs(ynext)
        # jax.debug.print("froot iiter: {iiter}, err: {err}, fy: {fy}", iiter=iiter, err=jnp.max(err),
        #                 fy=jnp.max(jnp.abs(func(ynext, params))))
        return ynext, err, tol, iiter + 1, jac

    def cond_func(carry):
        _, err, tol, _, _ = carry
        return jnp.any(err > tol)

    err = jnp.full_like(y0, jnp.inf)
    tol = jnp.zeros_like(y0)
    iiter = jnp.array(0, dtype=jnp.int32)
    jac0 = jnp.zeros((y0.size, y0.size))
    # yiter: (max_iter, *ny), erriter: (max_iter, *ny), toliter: (max_iter, *ny)
    (y, err, tol, iiter, jac), (yiter, erriter, toliter, _, _) = while_loop_scan(
        cond_func, iter_func, (y0, err, tol, iiter, jac0), max_iter=max_iter, unroll=1)
    is_converged = jnp.all(err <= tol)  # ()
    # (max_iter, *1)
    is_converged_iter = jnp.all(erriter <= toliter, axis=tuple(range(1, erriter.ndim)), keepdims=True)
    return (y, is_converged), (yiter, is_converged_iter), jac

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
    (yt, is_converged), _, jac = newton_iter_helper(
        func, y0, params, max_iter=max_iter, atol=atol, rtol=rtol)
    
    # compute grad of f
    func_partial_y = partial(func, yt)
    # grad_func: (ny,)
    _, grad_func = jax.jvp(func_partial_y, (params,), (grad_params,))

    grad_y = jnp.linalg.solve(jac, -grad_func)  # (ny,)
    result = Result(yt, is_converged)
    grad_result = Result(grad_y)
    return result, grad_result
