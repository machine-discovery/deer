from abc import abstractmethod
from typing import Any, Callable, List, Optional, Tuple
import jax.numpy as jnp
from deer.deer_iter import deer_iteration
from deer.maths import matmul_recursive
from deer.utils import get_method_meta, check_method


__all__ = ["solve_ivp"]

def solve_ivp(func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
              y0: jnp.ndarray, xinp: jnp.ndarray, params: Any,
              tpts: jnp.ndarray,
              method: Optional["SolveIVPMethod"] = None,
              ) -> jnp.ndarray:
    r"""
    Solve the initial value problem.
    
    .. math::

        \frac{dy}{dt} = f(y, x; \theta)
    
    with given initial condition :math:`y(0) = y_0`,
    where :math:`y` is the output signal, :math:`x` is the input signal, and :math:`\theta` is the parameters
    of the function.
    This function will return the output signal :math:`y` at the time points :math:`t`.

    Arguments
    ---------
    func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        Function to evaluate the derivative of :math:`y` with respect to :math:`t`. The
        arguments are: output signal :math:`y` ``(ny,)``, input signal :math:`x` ``(nx,)``, and parameters
        :math:`\theta` in a pytree. The return value is the derivative of :math:`y` with respect to :math:`t`,
        i.e., :math:`\frac{dy}{dt}` ``(ny,)``.
    y0: jnp.ndarray
        Initial condition on :math:`y` ``(ny,)``.
    xinp: jnp.ndarray
        The external input signal of shape ``(nsamples, nx)``.
    params: Any
        The parameters of the function ``func``.
    tpts: jnp.ndarray
        The time points to evaluate the solution ``(nsamples,)``.
    method: Optional[SolveIVPMethod]
        The method to solve the initial value problem. If None, then use the ``DEER()`` method.

    Returns
    -------
    y: jnp.ndarray
        The output signal as the solution of the non-linear differential equations ``(nsamples, ny)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from fsolve_ivp import solve_ivp
    >>>
    >>> def simple_harmonic_oscillator(y, x, params):
    ...     k, m = params
    ...     dydt = jnp.array([y[1], -k/m*y[0]])
    ...     return dydt
    >>>
    >>> y0 = jnp.array([1.0, 0.0])
    >>> xinp = jnp.zeros((100, 0))  # no input signal
    >>> params = (1.0, 1.0)  # k, m
    >>> tpts = jnp.linspace(0, 10, 100)
    >>>
    >>> y = solve_ivp(simple_harmonic_oscillator, y0, xinp, params, tpts)
    >>> # The output y should be an array of shape (nsamples, ny)
    >>> y.shape
    (100, 2)
    >>> # Check the first and last values (should be close to [1.0, 0.0] and [cos(10), -sin(10)] respectively)
    >>> jnp.allclose(y[0], jnp.array([1.0, 0.0]))
    Array(True, dtype=bool)
    >>> jnp.allclose(y[-1], jnp.array([jnp.cos(10), -jnp.sin(10)]), atol=1e-2)
    Array(True, dtype=bool)
    """
    if method is None:
        method = DEER()
    check_method(method, solve_ivp)
    return method.compute(func, y0, xinp, params, tpts)

class SolveIVPMethod(metaclass=get_method_meta(solve_ivp)):
    @abstractmethod
    def compute(self, func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: jnp.ndarray, params: Any, tpts: jnp.ndarray):
        pass

class DEER(SolveIVPMethod):
    """
    Compute the solution of initial value problem with the DEER method.

    Arguments
    ---------
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal ``(nsamples, ny)``.
        If None, it will be initialized as 0s.
    max_iter: int
        The maximum number of iterations to perform.
    memory_efficient: bool
        If True, then use the memory efficient algorithm for the DEER iteration.
    """
    def __init__(self, yinit_guess: Optional[jnp.ndarray] = None, max_iter: int = 10000,
                 memory_efficient: bool = True):
        self.yinit_guess = yinit_guess
        self.max_iter = max_iter
        self.memory_efficient = memory_efficient

    def compute(self, func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: jnp.ndarray, params: Any, tpts: jnp.ndarray):
        # set the default initial guess
        yinit_guess = self.yinit_guess
        if yinit_guess is None:
            yinit_guess = jnp.zeros((tpts.shape[0], y0.shape[-1]), dtype=tpts.dtype) + y0

        def func2(ylist: List[jnp.ndarray], x: jnp.ndarray, params: Any) -> jnp.ndarray:
            return func(ylist[0], x, params)

        def shifter_func(y: jnp.ndarray, params: Any) -> List[jnp.ndarray]:
            # y: (nsamples, ny)
            return [y]

        # perform the deer iteration
        inv_lin_params = (tpts, y0)
        yt = deer_iteration(
            inv_lin=self.solve_ivp_inv_lin, p_num=1, func=func2, shifter_func=shifter_func, params=params, xinput=xinp,
            inv_lin_params=inv_lin_params, shifter_func_params=(), yinit_guess=yinit_guess, max_iter=self.max_iter,
            memory_efficient=self.memory_efficient)
        return yt

    def solve_ivp_inv_lin(self, gmat: List[jnp.ndarray], rhs: jnp.ndarray,
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
        yt = matmul_recursive(gtbar, htbar, y0)  # (nt, ny)
        return yt


class GeneralODE(SolveIVPMethod):
    """
    Compute the solution of initial value problem with the ODE methods.

    Arguments
    ---------
    step_size: float
        The step size for ODE solver. If None, it will use (tpts[i] - tpts[i - 1]).
    """
    def __init__(self, step_size: Optional[float] = None):
        self.step_size = step_size

    def compute(self, func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: jnp.ndarray, params: Any, tpts: jnp.ndarray):
        # Initialize the solution list with the initial condition
        y = [y0]

        # Iterate over time points to compute the solution at each step
        for i in range(1, len(tpts)):
            yi = y[-1]
            xi = xinp[i-1]
            xf = xinp[i]
            ti = tpts[i-1]
            tf = tpts[i]

            # Determine the step size
            dt = self.step_size if self.step_size is not None else (tf - ti)

            # Number of steps between tpts, at least 1
            num_steps = max(int((tf - ti) / dt), 1)
            dt = (tf - ti) / num_steps  # Recalculate dt to evenly divide the interval
            dx = (xf - xi) / (tf - ti) * dt

            for _ in range(num_steps):
                yi, xi = self.ode_step(func, yi, xi, dt, dx, params)
            y.append(yi)

        # Stack the list of solutions into a single jax array
        return jnp.stack(y)

    @abstractmethod
    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray], 
        yi: jnp.ndarray, 
        xi: jnp.ndarray, 
        dt: jnp.ndarray, 
        dx: jnp.ndarray, 
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solve a single step of the ODE using the specific method.

        Parameters
        ----------
        func : Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
            The function defining the differential equation dy/dt = f(y, x, params)
        yi : jnp.ndarray
            The state of the system at the beginning of the time step.
        xi : jnp.ndarray
            The input at the beginning of the time step.
        dt : jnp.ndarray
            The size of the time step.
        dx : jnp.ndarray
            The change in input over the time step.
        params : Any
            Additional parameters for the differential equation.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            The state of the system and input at the end of the time step.
        """
        pass

class ForwardEuler(GeneralODE):
    """
    Compute the solution of initial value problem with the Forward Euler method.
    
    Arguments
    ---------
    step_size: float
        The step size to use for the Euler method. If None, it will use the difference (tpts[1] - tpts[0]) divided by the number of steps.
    """
    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray], 
        yi: jnp.ndarray, 
        xi: jnp.ndarray, 
        dt: jnp.ndarray, 
        dx: jnp.ndarray, 
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        k = func(yi, xi, params)
        yi_new = yi + dt * k
        xi_new = xi + dx
        return yi_new, xi_new


class RK3(GeneralODE):
    """
    Compute the solution of initial value problem with the Runge-Kutta 3rd order method.

    Arguments
    ---------
    step_size: float
        The step size to use for the RK3 method. If None, it will use (tpts[i] - tpts[i - 1]).
    """

    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray], 
        yi: jnp.ndarray, 
        xi: jnp.ndarray, 
        dt: jnp.ndarray, 
        dx: jnp.ndarray, 
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        k1 = func(yi, xi, params)
        k2 = func(yi + dt * k1, xi + dx, params)
        k3 = func(yi + 0.25 * dt * (k1 + k2), xi + 0.5 * dx, params)

        yi = yi + (dt / 6.0) * (k1 + k2 + 4 * k3)
        xi = xi + dx
        return yi, xi


class RK4(GeneralODE):
    """
    Compute the solution of initial value problem with the Runge-Kutta 4th order method.

    Arguments
    ---------
    step_size: float
        The step size to use for the RK4 method. If None, it will use (tpts[i] - tpts[i - 1]).
    """
    
    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray], 
        yi: jnp.ndarray, 
        xi: jnp.ndarray, 
        dt: jnp.ndarray, 
        dx: jnp.ndarray, 
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        k1 = func(yi, xi, params)
        k2 = func(yi + 0.5 * dt * k1, xi + 0.5 * dx, params)
        k3 = func(yi + 0.5 * dt * k2, xi + 0.5 * dx, params)
        k4 = func(yi + dt * k3, xi + dx, params)

        yi = yi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        xi = xi + dx
        return yi, xi


class GeneralBackwardODE(GeneralODE):
    """
    Compute the solution of initial value problem with Backward ODE methods.

    Arguments
    ---------
    step_size: float
        The step size for ODE solver. If None, it will use (tpts[i] - tpts[i - 1]).
    tol: float
        The tolerance for the fixed-point iteration.
    max_iter: int
        The maximum number of iterations for the fixed-point iteration.
    """
    def __init__(self, step_size: Optional[float] = None, tol: float = 1e-6, max_iter: int = 100):
        super().__init__(step_size)
        self.tol = tol
        self.max_iter = max_iter


class BackwardEuler(GeneralBackwardODE):
    """
    Compute the solution of initial value problem with the Backward Euler method.
    """
    
    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        yi: jnp.ndarray,
        xi: jnp.ndarray,
        dt: jnp.ndarray,
        dx: jnp.ndarray,
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        y_new = yi
        for _ in range(self.max_iter):
            y_new_next = yi + dt * func(y_new, xi, params)
            if jnp.linalg.norm(y_new_next - y_new) < self.tol:
                y_new = y_new_next
                break
            y_new = y_new_next
        xi_new = xi + dx
        return y_new, xi_new


class TrapezoidalMethod(GeneralBackwardODE):
    """
    Compute the solution of initial value problem with the Trapezoidal method.
    """
    
    def ode_step(
        self,
        func: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
        yi: jnp.ndarray,
        xi: jnp.ndarray,
        dt: jnp.ndarray,
        dx: jnp.ndarray,
        params: Any
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Initial guess for fixed-point iteration (use forward Euler as an initial guess)
        y_new = yi + dt * func(yi, xi, params)
        
        # Fixed-point iteration to solve the trapezoidal equation
        for _ in range(self.max_iter):
            y_new_next = yi + (dt / 2.0) * (func(yi, xi, params) + func(y_new, xi + dx, params))
            if jnp.linalg.norm(y_new_next - y_new) < self.tol:
                y_new = y_new_next
                break
            y_new = y_new_next

        xi_new = xi + dx
        return y_new, xi_new