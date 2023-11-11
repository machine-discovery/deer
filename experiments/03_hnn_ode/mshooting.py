import time
from typing import Callable
import jax.numpy as jnp
from flax import linen as nn
from debug import shape


def odeint_mshooting(
    f: Callable,
    x: jnp.array,
    t_span: jnp.array,
    params,
    B0=None,
    fine_steps=2,
    maxiter=4,
):
    print(f"args of odeint_mshooting: {shape((f, x, t_span, params, B0, fine_steps, maxiter), do_print=False)}")

    solver = MSZero()
    # first-guess B0 of shooting parameters
    B0 = fixed_odeint(f, x, t_span, solver.coarse_method, params)
    print(f"shape of B0: {B0.shape}")
    # determine which odeint to apply to MS solver. This is where time-variance can be introduced
    B = solver.root_solve(f, x, t_span, B0, fine_steps, maxiter, params)
    return B


def fixed_odeint(f, x, t_span, solver, params):
    """Solves IVPs with same `t_span`, using fixed-step methods"""
    start_time = time.time()
    t, dt = t_span[0], t_span[1] - t_span[0]
    sol = [x]
    steps = 1
    while steps <= len(t_span) - 1:
        if steps % 100 == 0:
            print(f"step {steps} of {len(t_span)}, shape of x: {x.shape} and t: {t.shape}")
        x = solver.step(f, x, t, dt, params=params)
        sol.append(x)
        t = t + dt
        if steps < len(t_span) - 1:
            dt = t_span[steps + 1] - t
        steps += 1

    time_elapsed = time.time() - start_time
    print(f"fixed_odeint took {time_elapsed} seconds, shape of x: {x.shape} and t: {t.shape}, shape of t_span: {t_span.shape}")
    return jnp.stack(sol)


class MSZero(nn.Module):
    def __init__(self):
        """Multiple shooting solver using Parareal updates (zero-order approximation of the Jacobian)

        Args:
            coarse_method (str, optional): . Defaults to 'euler'.
            fine_method (str, optional): . Defaults to 'rk4'.
        """
        self.coarse_method = Euler()
        self.fine_method = RungeKutta4()

    def root_solve(self, f, x, t_span, B, fine_steps, maxiter, params):
        dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
        sub_t_span = jnp.linspace(0, dt, fine_steps)
        i = 0
        while i <= maxiter:
            i += 1
            B_coarse = fixed_odeint(f, B[i-1:], sub_t_span, solver=self.coarse_method, params=params)[-1]
            B_fine = fixed_odeint(f, B[i-1:], sub_t_span, solver=self.fine_method, params=params)[-1]
            B_out = jnp.zeros_like(B)
            B_out.at[:i].set(B[:i])
            B_in = B[i-1]
            for m in range(i, n_subinterv):
                B_in = fixed_odeint(f, B_in, sub_t_span, solver=self.coarse_method, params=params)[-1]
                B_in = B_in - B_coarse[m-i] + B_fine[m-i]
                B_out.at[m].set(B_in)
            B = B_out
        return B


class Euler():
    def __init__(self, dtype=jnp.float32):
        """Explicit Euler ODE stepper, order 1"""
        self.dtype = dtype

    def step(self, f, x, t, dt, k1=None, params=None):
        if k1 is None:
            k1 = f(x, t, params)
        x_sol = x + dt * k1
        return x_sol


class RungeKutta4():
    def __init__(self, dtype=jnp.float32):
        """Explicit Midpoint ODE stepper, order 4"""
        super().__init__()
        self.dtype = dtype
        c = jnp.array([0., 1 / 2, 1 / 2, 1], dtype=dtype)
        a = [
            jnp.array([1 / 2], dtype=dtype),
            jnp.array([0., 1 / 2], dtype=dtype),
            jnp.array([0., 0., 1], dtype=dtype)]
        bsol = jnp.array([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=dtype)
        berr = jnp.array([0.])
        self.tableau = (c, a, bsol, berr)

    def step(self, f, x, t, dt, k1=None, params=None):
        c, a, bsol, _ = self.tableau
        if k1 is None:
            k1 = f(x, t, params)
        k2 = f(x + dt * (a[0] * k1), t + c[0] * dt, params)
        k3 = f(x + dt * (a[1][0] * k1 + a[1][1] * k2), t + c[1] * dt, params)
        k4 = f(x + dt * (a[2][0] * k1 + a[2][1] * k2 + a[2][2] * k3), t + c[2] * dt, params)
        x_sol = x + dt * (bsol[0] * k1 + bsol[1] * k2 + bsol[2] * k3 + bsol[3] * k4)
        return x_sol
