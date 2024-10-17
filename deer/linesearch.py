from abc import abstractmethod
from typing import Callable, Any
import jax.numpy as jnp
import jax


class LineSearch:
    @abstractmethod
    def forward(self, y: jnp.ndarray, ynext: jnp.ndarray, func: Callable, params: Any) -> jnp.ndarray:
        pass

class ArmijoLineSearch(LineSearch):
    def __init__(self, c: float = 0.5, alpha: float = 0.1, max_iter: int = 4):
        self.c = c  # multiplier on the step size
        # 1e-4 is the recommendation in https://iate.oac.uncor.edu/~mario/materia/nr/numrec/f9-7.pdf
        self.alpha = alpha  # multiplier on the gradient threshold
        self.max_iter = max_iter

    def forward(self, y: jnp.ndarray, ynext: jnp.ndarray, func: Callable, params: Any) -> jnp.ndarray:
        def dot(x, y):
            return jnp.einsum("...,...->", x, y)

        def get_loss(y, params):
            fy = func(y, params)
            return dot(fy, fy)

        dy = ynext - y
        fynext = func(ynext, params)
        func_vg = jax.value_and_grad(get_loss)
        dotfyfy, grad_loss = func_vg(y, params)
        lmbda = jnp.array(1.0, dtype=y.dtype)  # step size
        c1 = self.c
        alpha = self.alpha
        max_iter = self.max_iter
        dot_gradloss_dy = dot(grad_loss, dy)
        iiter = jnp.array(0, dtype=jnp.float32)

        def while_func2(carry):
            lmbda, y, ynext, dy, params, iiter, fynext, dotfyfy, dot_gradloss_dy = carry
            # lmbda = jnp.clip(-dot_gradloss_dy / 2 / (dot(fynext, fynext) * 0.5 - dotfyfy * 0.5 - dot_gradloss_dy),
            #                  max=lmbda * c1, min=lmbda * c2)
            lmbda = lmbda * c1
            ynext = y + lmbda * dy
            fynext = func(ynext, params)
            iiter = iiter + 1
            return lmbda, y, ynext, dy, params, iiter, fynext, dotfyfy, dot_gradloss_dy

        def cond_func2(carry):
            lmbda, y, ynext, dy, params, iiter, fynext, dotfyfy, dot_gradloss_dy = carry
            cond1 = dot(fynext, fynext) > dotfyfy + alpha * lmbda * dot_gradloss_dy
            cond2 = iiter < max_iter
            return jnp.logical_and(cond1, cond2)

        ppwhile0 = (lmbda, y, ynext, dy, params, iiter, fynext, dotfyfy, dot_gradloss_dy)
        ppwhile = jax.lax.while_loop(cond_func2, while_func2, ppwhile0)
        (lmbda, y, ynext, dy, params, iiter, fynext, dotfyfy, dot_gradloss_dy) = ppwhile
        return ynext
