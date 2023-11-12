import jax.numpy as jnp
from jax import random
from jax import lax, jit, vmap
from functools import partial


def odeint_mshooting(f, x: jnp.array, t_span: jnp.array, params, fine_steps=4, maxiter=4):
    B0 = fixed_odeint(f, x, t_span, euler_step, params)
    B = root_solve(f, t_span, B0, fine_steps, maxiter, params)
    return B


@partial(jit, static_argnames=("f", "solver"))
def fixed_odeint(f, x_init, t_span, solver, params):
    """Solves IVPs with same `t_span`, using fixed-step methods"""

    def step_fn(carry, t):
        x, t_prev = carry
        dt = t - t_prev
        t_prev = t
        x = solver(f, x, t_prev, dt, params=params)
        return (x, t_prev), x

    _, sol = lax.scan(step_fn, (x_init, t_span[0]), t_span[1:])
    return jnp.concatenate([x_init[None, :], sol])


@partial(jit, static_argnames=("f", "fine_steps", "maxiter"))
def root_solve(f, t_span, B, fine_steps, maxiter, params):
    vmap_f = vmap(f, in_axes=(0, None, None))
    def step_fn_inner(carry, m):
        B_in, B_coarse, B_fine = carry
        B_in = fixed_odeint(f, B_in, sub_t_span, solver=euler_step, params=params)[-1]
        B_in = B_in - B_coarse[m] + B_fine[m]
        return (B_in, B_coarse, B_fine), B_in

    dt, n_subinterv = t_span[1] - t_span[0], len(t_span)
    sub_t_span = jnp.linspace(0, dt, fine_steps)
    for i in range(maxiter+1):
        B_coarse = fixed_odeint(vmap_f, B, sub_t_span, solver=euler_step, params=params)[-1]
        B_fine = fixed_odeint(vmap_f, B, sub_t_span, solver=rk4_step, params=params)[-1]
        _, B_tail = lax.scan(step_fn_inner, (B[i], B_coarse, B_fine), jnp.arange(i, n_subinterv - 1))
        B = B.at[i + 1:].set(B_tail)
    return B


@partial(jit, static_argnames=("f",))
def euler_step(f, x, t, dt, params=None):
    k1 = f(x, t, params)
    x_sol = x + dt * k1
    return x_sol


@partial(jit, static_argnames=("f",))
def rk4_step(f, x, t, dt, params=None):
    k1 = f(x, t, params)
    k2 = f(x + dt * (1/2 * k1), t,            params)
    k3 = f(x + dt * (1/2 * k2), t + 1/2 * dt, params)
    k4 = f(x + dt * k3,         t + 1/2 * dt, params)
    x_sol = x + dt * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)
    return x_sol


if __name__ == "__main__":
    @jit
    def lorenz(x, t, params=None):
        x1, x2, x3 = jnp.split(x, 3, axis=-1)
        dx1 = 10 * (x2 - x1)
        dx2 = x1 * (28 - x3) - x2
        dx3 = x1 * x2 - 8/3 * x3
        return jnp.concatenate([dx1, dx2, dx3], -1)
    key = random.PRNGKey(0)
    x0 = 15 + random.normal(key, (8, 3))
    t_span = jnp.linspace(0, 3, 3000)
    b1 = odeint_mshooting(lorenz, x0, t_span, params=None)
    # b2 = odeint_mshooting_2(lorenz, x0, t_span, params=None)
    # print(jnp.allclose(b1, b2))
    # print(jnp.sum(jnp.abs(b1 - b2)))
