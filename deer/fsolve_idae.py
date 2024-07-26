from abc import abstractmethod
from typing import Any, Callable, List, Optional
import jax
import jax.numpy as jnp
import optimistix as optx
from deer.deer_iter import deer_iteration, deer_iteration_full
from deer.maths import matmul_recursive
from deer.utils import get_method_meta, check_method, Result
from deer.froot import root, RootMethod


__all__ = ["solve_idae"]

def solve_idae(func: Callable[[jnp.ndarray, jnp.ndarray, Any, Any], jnp.ndarray],
               y0: jnp.ndarray, xinp: Any, params: Any,
               tpts: jnp.ndarray,
               method: Optional["SolveIDAEMethod"] = None,
               ) -> Result:
    r"""
    Solve the implicit differential algebraic equations (IDAE) systems.

    .. math::

        f(\dot{y}, y, x; \theta) = 0

    where :math:`\dot{y}` is the time-derivative of the output signal :math:`y`, :math:`x` is the input signal
    at given sampling time :math:`t`, and :math:`\theta` are the parameters of the function.
    The tentative initial condition is given by :math:`y(0) = y_0`.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, jnp.ndarray, Any, Any], jnp.ndarray]
        Function to evaluate the residual of the IDAE system.
        The arguments are:
        (1) time-derivative of the output signal :math:`\dot{y}` ``(ny,)``,
        (2) output signal :math:`y` ``(ny,)``,
        (3) input signal :math:`x` ``(*nx,)`` in a pytree, and
        (4) parameters :math:`\theta` in a pytree.
        The return value is the residual of the IDAE system ``(ny,)``.
    y0: jnp.ndarray
        Tentative initial condition on :math:`y` ``(ny,)``. If the IDAE system has algebraic variables, then
        the initial values of the algebraic variables might be different to what is supplied.
    xinp: Any
        The external input signal of shape ``(nsamples, *nx)`` in a pytree.
    params: Any
        The parameters of the function ``func``.
    tpts: jnp.ndarray
        The time points to evaluate the solution ``(nsamples,)``. Must be monotonically increasing. Duplicate values
        will cause nans.
    method: Optional[SolveIDAEMethod]
        The method to solve the implicit DAE. If None, then use the ``BwdEulerDEER()`` method.

    Returns
    -------
    res: Result
        The ``Result`` object where ``.value`` is the solution of the IDAE system at the given time with
        shape ``(nsamples, ny)`` and ``.success`` is the boolean array indicating the convergence of the solver.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> def idae_func(dy, y, x, params):
    ...     return dy + y - x - params
    >>> y0 = jnp.array([1.0])
    >>> xinp = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    >>> params = jnp.array([0.5])
    >>> tpts = jnp.array([0.0, 1.0, 2.0, 3.0])
    >>> solve_idae(idae_func, y0, xinp, params, tpts).value
    Array([[1.    ],
           [1.25  ],
           [1.875 ],
           [2.6875]], dtype=float64)
    """
    if method is None:
        method = BwdEulerDEER()
    check_method(method, solve_idae)
    return method.compute(func, y0, xinp, params, tpts)

class SolveIDAEMethod(metaclass=get_method_meta(solve_idae)):
    @abstractmethod
    def compute(self, func: Callable[[jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray) -> Result:
        pass

class BwdEuler(SolveIDAEMethod):
    """
    Solve the implicit DAE method using backward Euler's method.

    Arguments
    ---------
    solver: Optional[RootMethod]
        The root finder solver. If None, then use the Newton's method.
    """
    def __init__(self, solver: Optional[RootMethod] = None):
        if solver is None:
            solver = root.Newton(max_iter=200, atol=1e-6, rtol=1e-3)
        self.solver = solver
        self.num_iter_returned = solver.num_iter_returned

    def compute(self, func: Callable[[jnp.ndarray, jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray) -> Result:
        # y0: (ny,) the initial states (it's not checked for correctness)
        # xinp: pytree, each has `(nsamples, *nx)`
        # tpts: (nsamples,) the time points
        # returns: (nsamples, ny), including the initial states
        return_full = self.num_iter_returned > 0
        def fn(yi, args):
            yim1, xi, dti, params = args
            return func((yi - yim1) / dti, yi, xi, params)

        def scan_fn(carry, x):
            _, success = carry

            def success_fn(carry, x):
                yprev, _ = carry
                xi, dti = x
                if return_full:
                    yprev = yprev[-1]
                sol = root(fn, yprev, (yprev, xi, dti, params), method=self.solver)
                yi = sol.value
                success = sol.success
                return yi, success

            def fail_fn(carry, x):
                yprev, _ = carry
                return yprev, jnp.full_like(yprev, False, dtype=jnp.bool)

            # success: (num_iter, ny) or (ny,)
            if return_full:
                cond = jnp.all(success[-1])
            else:
                cond = jnp.all(success)
            res = jax.lax.cond(cond, success_fn, fail_fn, carry, x)
            return res, res

        dti = tpts[1:] - tpts[:-1]  # (nsamples - 1,)
        xi = jax.tree_util.tree_map(lambda x: x[1:], xinp)  # (nsamples - 1, *nx)
        if return_full:
            y0 = jnp.tile(y0, (self.num_iter_returned, 1))  # (num_iter, ny)
        carry = (y0, jnp.full_like(y0, True, dtype=jnp.bool))
        # (nsamples - 1, ny) or (nsamples - 1, num_iter, ny)
        _, (y, success) = jax.lax.scan(scan_fn, carry, (xi, dti), unroll=1)
        y = jnp.concatenate((y0[None], y), axis=0)  # (nsamples, ny) or (nsamples, num_iter, ny)
        # (nsamples, ny) or (nsamples, num_iter, ny)
        success = jnp.concatenate((jnp.full_like(success[:1], True, dtype=jnp.bool), success), axis=0)

        # if the method returns multiple iterations, then move the axis to be consistent
        if return_full:
            y = jnp.moveaxis(y, 0, 1)  # (num_iter, nsamples, ny)
            success = jnp.moveaxis(success, 0, 1)  # (num_iter, nsamples, ny)
        return Result(y, success)

class BwdEulerDEER(SolveIDAEMethod):
    """
    Solve the implicit DAE method using DEER method for backward Euler's method.

    Arguments
    ---------
    yinit_guess: Optional[jnp.ndarray]
        The initial guess of the output signal ``(nsamples, ny)``.
        If None, it will be initialized as all ``y0``.
    max_iter: int
        The maximum number of DEER iterations to perform.
    atol: Optional[float]
        The absolute tolerance of the DEER iteration convergence.
    rtol: Optional[float]
        The relative tolerance of the DEER iteration convergence.
    return_full: bool
        If True, return the full result of the DEER iteration. Otherwise, return the
        final result only.
    """
    def __init__(self, yinit_guess: Optional[jnp.ndarray] = None, max_iter: int = 200, atol: Optional[float] = None,
                 rtol: Optional[float] = None, return_full: bool = False):
        self.yinit_guess = yinit_guess
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol
        self.return_full = return_full

    def compute(self, func: Callable[[jnp.ndarray, jnp.ndarray, Any, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray) -> Result:
        # y0: (ny,) the initial states (it's not checked for correctness)
        # xinp: pytree, each has `(nsamples, *nx)`
        # tpts: (nsamples,) the time points
        # returns: (nsamples, ny), including the initial states if not self.return_full
        # else returns (max_iter, nsamples, ny) for the full result

        # set the default initial guess
        yinit_guess = self.yinit_guess
        if yinit_guess is None:
            yinit_guess = jnp.zeros((tpts.shape[0], y0.shape[-1]), dtype=tpts.dtype) + y0

        def func2(yshifts: List[jnp.ndarray], x: Any, params: Any) -> jnp.ndarray:
            # yshifts: [2] + (ny,)
            # x is dt
            y, ym1 = yshifts
            dt, xinp = x
            return func((y - ym1) / dt, y, xinp, params)

        def linfunc(y: jnp.ndarray, lin_params: Any) -> List[jnp.ndarray]:
            # y: (nsamples, ny)
            # we're using backward euler's method, so we need to shift the values by one
            ym1 = jnp.concatenate((y[:1], y[:-1]), axis=0)  # (nsamples, ny)
            return [y, ym1]

        # dt[i] = t[i] - t[i - 1]
        dt_partial = tpts[1:] - tpts[:-1]  # (nsamples - 1,)
        dt = jnp.concatenate((dt_partial[:1], dt_partial), axis=0)  # (nsamples,)

        kwargs = {
            "inv_lin": self.solve_idae_inv_lin,
            "func": func2,
            "shifter_func": linfunc,
            "p_num": 2,
            "params": params,
            "xinput": (dt, xinp),
            "inv_lin_params": (y0,),
            "shifter_func_params": None,
            "yinit_guess": yinit_guess,
            "max_iter": self.max_iter,
            "clip_ytnext": True,
            "atol": self.atol,
            "rtol": self.rtol,
        }
        result = deer_iteration_full(**kwargs) if self.return_full else deer_iteration(**kwargs)
        return result

    def solve_idae_inv_lin(self, jacs: List[jnp.ndarray], z: jnp.ndarray,
                           inv_lin_params: Any) -> jnp.ndarray:
        # solving the equation: M0_i @ y_i + M1_i @ y_{i-1} = z_i
        # M: (nsamples, ny, ny)
        # G: (nsamples, ny, ny)
        # rhs: (nsamples, ny)
        # inv_lin_params: (y0,) where tpts: (nsamples,), y0: (ny,)
        M0, M1 = jacs
        y0, = inv_lin_params  # tpts: (nsamples,), y0: (ny,)

        # using index [1:] because we don't need to compute y_0 again (it's already available from y0)
        M01 = M0[1:]
        M0invM1 = -jax.vmap(jnp.linalg.solve)(M01, M1[1:])
        M0invz = jax.vmap(jnp.linalg.solve)(M01, z[1:])
        y = matmul_recursive(M0invM1, M0invz, y0)  # (nsamples, ny)
        return y
