from typing import Callable, Optional, Any, List
from abc import abstractmethod
import jax
import jax.numpy as jnp
from deer.deer_iter import deer_iteration
from deer.maths import matmul_recursive
from deer.utils import get_method_meta, check_method, Result


def solve_sde(ffunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
              gfunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
              y0: jnp.ndarray, xinp: jnp.ndarray, params: Any,
              tpts: jnp.ndarray,
              method: Optional["SolveSDEMethod"] = None,
              *,
              key: jax.random.PRNGKey,
              ) -> Result:
    r"""
    Solve the stochastic differential equation (SDE) with a given initial value.

    .. math::

        dy = f(y, x; \theta) dt + G(y, x; \theta) dW
    
    Arguments
    ---------
    ffunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        Function to evaluate the drift term of the SDE. The arguments are: output signal :math:`y` ``(ny,)``,
        input signal :math:`x` ``(nx,)``, and parameters :math:`\theta` in a pytree. The return value is the drift
        term :math:`f(y, x; \theta)` in shape of ``(ny,)``.
    gfunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray]
        Function to evaluate the diffusion term of the SDE. The arguments are: output signal :math:`y` ``(ny,)``,
        input signal :math:`x` ``(nx,)``, and parameters :math:`\theta` in a pytree. The return value is the diffusion
        term :math:`G(y, x; \theta)` in shape of ``(ny,)`` or ``(ny, nbrown)``.
    y0: jnp.ndarray
        Initial condition on :math:`y` ``(ny,)``.
    xinp: jnp.ndarray
        The external input signal of shape ``(nsamples, nx)``.
    params: Any
        The parameters of the function ``func``, denoted as :math:`\theta` in the equations above.
    tpts: jnp.ndarray
        The time points to evaluate the solution ``(nsamples,)``.
    method: Optional[SolveSDEMethod]
        The method to solve the SDE. If None, then use the ``EulerMaruyamaDEER()`` method.

    Keyword arguments
    -----------------
    key: jax.random.PRNGKey
        The random key to generate the Brownian motion.

    Returns
    -------
    res: Result
        The ``Result`` object where ``.value`` is the solution of the SDE system at the given time with
        shape ``(nsamples, ny)`` and ``.success`` is the boolean array indicating the convergence of the solver.
    """
    if method is None:
        method = EulerMaruyamaDEER()
    check_method(method, solve_sde)

    # check the inputs
    if tpts.shape[0] != xinp.shape[0]:
        msg = ("The number of time points must be equal to the number of samples. "
               f"Got {tpts.shape[0]} and {xinp.shape[0]}.")
        raise ValueError(msg)

    # get the number of Brownian
    G = gfunc(y0, xinp[0], params)  # (ny,) or (ny, nbrown)
    if G.ndim not in {1, 2}:
        msg = f"The output of the diffusion term must be 1D or 2D. Got {G.ndim}"
        raise ValueError(msg)
    nbrown = G.shape[-1]

    return method.compute(ffunc, gfunc, y0, xinp, params, tpts, nbrown, key=key)

class SolveSDEMethod(metaclass=get_method_meta(solve_sde)):
    @abstractmethod
    def compute(self, ffunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                gfunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray,
                nbrown: Optional[int], *, key: jax.random.PRNGKey) -> Result:
        pass

class EulerMaruyama(SolveSDEMethod):
    """
    Compute the solution of initial value problem with the Euler-Maruyama method.                
    """
    def compute(self, ffunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                gfunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray,
                nbrown: Optional[int], *, key: jax.random.PRNGKey) -> Result:
        ntpts = tpts.shape[0]
        all_noise = jax.random.normal(key, shape=(ntpts - 1, nbrown))  # (ntpts - 1, nbrown)
        dt = tpts[1:] - tpts[:-1]  # (ntpts - 1,)
        dt = dt[..., None]  # (ntpts - 1, 1)
        brownian = all_noise * jnp.sqrt(dt)  # (ntpts, nbrown)

        def scan_fn(carry, inputs):
            dt, brownian_noise, xinp = inputs
            y, params = carry
            ft = ffunc(y, xinp, params)  # (ny,)
            gt = gfunc(y, xinp, params)  # (ny,) or (ny, nbrown)
            noise = _apply_gt(gt, brownian_noise)
            y = y + ft * dt + noise
            return (y, params), y

        # yall: (ntpts - 1, ny)
        _, yall = jax.lax.scan(scan_fn, (y0, params), (dt, brownian, xinp[:-1]))
        yall = jnp.concatenate((y0[None], yall), axis=0)  # (ntpts, ny)
        return Result(yall)

class EulerMaruyamaDEER(SolveSDEMethod):
    """
    Compute the solution of initial value problem with the Euler-Maruyama DEER method.

    Arguments
    ---------
    yinit_guess: jnp.ndarray or None
        The initial guess of the output signal ``(nsamples, ny)``.
        If None, it will be initialized as all ``y0``.
    max_iter: int
        The maximum number of iterations to perform.
    atol: Optional[float]
        The absolute tolerance for the convergence of the solver.
    rtol: Optional[float]
        The relative tolerance for the convergence of the solver.
    """
    def __init__(self, yinit_guess: Optional[jnp.ndarray] = None, max_iter: int = 100,
                 atol: Optional[float] = None, rtol: Optional[float] = None):
        self.yinit_guess = yinit_guess
        self.max_iter = max_iter
        self.atol = atol
        self.rtol = rtol

    def compute(self, ffunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                gfunc: Callable[[jnp.ndarray, jnp.ndarray, Any], jnp.ndarray],
                y0: jnp.ndarray, xinp: Any, params: Any, tpts: jnp.ndarray,
                nbrown: Optional[int], *, key: jax.random.PRNGKey) -> Result:
        ntpts = tpts.shape[0]
        all_noise = jax.random.normal(key, shape=(ntpts - 1, nbrown))  # (ntpts - 1, nbrown)
        dt = tpts[1:] - tpts[:-1]  # (ntpts - 1,)
        dt = dt[..., None]  # (ntpts - 1, 1)
        brownian = all_noise * jnp.sqrt(dt)  # (ntpts - 1, nbrown)

        def nonlin_func(ys, inputs, params):
            yprev, = ys  # (ny,)
            dt, brownian_noise, xinp = inputs  # xinp: (nx,), brownian_noise: (nbrown,)
            ft = ffunc(yprev, xinp, params)
            gt = gfunc(yprev, xinp, params)  # (ny,) or (ny, nbrown)
            return yprev + ft * dt + _apply_gt(gt, brownian_noise)  # (ny,)

        def shifter_func(y, shifter_params: Any):
            # y: (nsamples, ny)
            # shifter_params = (y0,)
            y0, = shifter_params  # (ny,)
            yprev = jnp.concatenate((y0[None], y[:-1]), axis=0)  # (nsamples, ny)
            return (yprev,)

        dt_expand = jnp.concatenate((dt[:1], dt), axis=0)  # (ntpts,)
        brownian_expand = jnp.concatenate((brownian[:1], brownian), axis=0)  # (ntpts, nbrown)
        # the last xinp is not needed
        xinp_expand = jnp.concatenate((xinp[:1], xinp[:-1]), axis=0)  # (ntpts, nx)
        result = deer_iteration(
            inv_lin=self.solve_inv_lin,
            func=nonlin_func,
            shifter_func=shifter_func,
            p_num=1,
            params=params,
            xinput=(dt_expand, brownian_expand, xinp_expand),
            inv_lin_params=(y0,),
            shifter_func_params=(y0,),
            yinit_guess=jnp.zeros((ntpts, y0.shape[-1]), dtype=tpts.dtype) + y0 \
                if self.yinit_guess is None else self.yinit_guess,
            max_iter=self.max_iter,
            clip_ytnext=True,
            atol=self.atol,
            rtol=self.rtol,
        )
        return result

    def solve_inv_lin(self, jacs: List[jnp.ndarray], z: jnp.ndarray, inv_lin_params: Any) -> jnp.ndarray:
        # jacs: [1] + (nsamples, ny, ny)
        # z: (nsamples, ny)
        # y0: (ny,)
        M, = jacs
        y0, = inv_lin_params
        y = matmul_recursive(-M[1:], z[1:], y0)  # (nsamples, ny)
        return y

def _apply_gt(gt: jnp.ndarray, brownian_noise: jnp.ndarray) -> jnp.ndarray:
    # gt: (ny,) or (ny, nbrown)
    # brownian_noise: (nbrown,)
    return jnp.dot(gt, brownian_noise) if gt.ndim == 2 else gt * brownian_noise  # (ny,)
