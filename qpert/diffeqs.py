from typing import Dict, Any
from abc import abstractmethod
import torch
import functorch


class BaseDiffEq:
    """
    The base class of the framework in solving differential equations of the form
        L[y] = f(y, x)
    using the fixed-point iterations
    """
    def __init__(self, *,
                 dtype: torch.dtype = torch.float64, device: torch.device = torch.device('cpu')):
        self._iter_opts = {
            "max_iter": 100,
            "atol": 1e-8,
            "verbose": 1,
        }
        self._dtype = dtype
        self._device = device

    def set_iter_options(self, **kwargs):
        self._iter_opts.update(kwargs)

    def get_iter_options(self) -> Dict[str, Any]:
        return self._iter_opts

    def dtype(self) -> torch.dtype:
        return self._dtype

    def device(self) -> torch.device:
        return self._device
    
    @abstractmethod
    def func(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # the function on the right hand side of equation 
        #   L[y] = f(y, x)
        # where y is the signal and x is the coordinate
        # y: (..., ny), x: (..., nx)
        # returns: (..., ny)
        pass

    def jac(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # func(y, t) -> fyt: [(..., ny), (..., nx)] -> (..., ny)
        # y: (..., ny)
        # x: (..., nx)
        # returns: (..., ny, ny)
        nbatch_dims = y.ndim - 1
        jacfunc = functorch.jacrev(self.func, argnums=0)  # [(ny), (nx)] -> (ny, ny)
        for _ in range(nbatch_dims):
            jacfunc = functorch.vmap(jacfunc)

        # compute the jacobian
        jac_fyt = jacfunc(y, x)
        return jac_fyt

    @abstractmethod
    def solve_lin(self, rhs: torch.Tensor, x: torch.Tensor, gy: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # solving the linear equation
        #   L[y] + gy @ y = rhs(x)
        # rhs: (..., ny)
        # x: (..., nx)
        # rhs: (..., ny)
        pass

    def solve_dual_lin(self, rhs: torch.Tensor, x: torch.Tensor, gy: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # the dual operator of solve_lin, that fulfills <a, L[b]> = <L*[a], b>
        # rhs: (..., ny)
        # x: (..., nx)
        # rhs: (..., ny)
        f = torch.ones_like(rhs, requires_grad=True)  # (..., ny)
        with torch.enable_grad():
            h = self.solve_lin(f, x, gy, *args, **kwargs)
        res, = torch.autograd.grad(h, f, grad_outputs=rhs, create_graph=torch.is_grad_enabled())
        return res

    def solve(self, y0: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # y0 is the initial guess, x is the coordinate
        # **kwargs are for the boundary conditions or initial conditions
        # y: (..., ny), x: (..., nx)
        y = y0

        converge = False
        iter_opts = self.get_iter_options()
        for i in range(iter_opts["max_iter"]):
            # compute the right hand side
            # fy: (..., ny), jac_fy: (..., ny, ny)
            fy = self.func(y, x)
            jac_fy = self.jac(y, x)
            gy = -jac_fy
            rhs = fy - (jac_fy @ y[..., None])[..., 0]  # (..., ny)

            # solve the linear equation
            ynew = self.solve_lin(rhs, x, gy, *args, **kwargs)

            # check the convergence
            diff = torch.mean(torch.abs(ynew - y))
            if iter_opts["verbose"]:
                print(f"Iter {i + 1}:", diff)
            y = ynew
            if diff < iter_opts["atol"]:
                converge = True
                break

        if not converge:
            print(f"The iteration does not converge in {iter_opts['max_iter']} iterations.")
        return y

    def get_fgrad(self, grad_y: torch.Tensor, ysol: torch.Tensor, x: torch.Tensor,
                  *args, **kwargs) -> torch.Tensor:
        # get the dL/df where f=f(y, x) so it can be used in propagating the gradients
        # to the parameters of f
        # ysol: (..., ny)
        # grad_y: (..., ny)
        # x: (..., nx)

        # jac_fy: (..., ny, ny)
        jac_fy = self.jac(ysol, x)
        fgrad = self.solve_dual_lin(grad_y, x, -jac_fy, *args, **kwargs)  # (..., ny)
        return fgrad
