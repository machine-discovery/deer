import jax
import jax.test_util
import jax.numpy as jnp
from deer.utils import while_loop_scan


jax.config.update("jax_enable_x64", True)

def test_while_loop_scan():
    s = jnp.array(1.0)
    i = 0
    carry = (s, i)
    cond_func = lambda carry: carry[1] < 10
    iter_func = lambda carry: (carry[0] * 2, carry[1] + 1)
    carry, stacked_carry = while_loop_scan(cond_func, iter_func, carry, 100)
    assert carry == (1024, 10)
    assert jnp.allclose(stacked_carry[0], 2 ** jnp.clip(jnp.arange(100) + 1, max=10))
    assert jnp.allclose(stacked_carry[1], jnp.clip(jnp.arange(100) + 1, max=10))

    # check the gradable
    def get_loss(s0: jnp.ndarray):
        carry = (s0, 0)
        carry, stacked_carry = while_loop_scan(cond_func, iter_func, carry, 100)
        return stacked_carry[0].sum()

    jax.test_util.check_grads(
        get_loss, (s,), order=1, modes=['rev', 'fwd'],
        # atol, rtol, eps following torch.autograd.gradcheck
        atol=1e-5, rtol=1e-3, eps=1e-6)
