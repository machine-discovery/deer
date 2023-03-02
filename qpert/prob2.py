from abc import abstractmethod
import torch
import numpy as np
import beeblo as bb
from scipy.optimize import root
from xitorch.optimize import equilibrium
from qpert.utils import Value


class NonlinProblem(torch.nn.Module):
    @abstractmethod
    def l0_inv_rhs(self) -> Value:
        # L0^{-1}[rhs]
        pass

    @abstractmethod
    def nonlin(self, y: Value) -> Value:
        pass

    def root_eq(self, y: Value) -> Value:
        return self.equil_eq(y) - y

    def equil_eq(self, y: Value) -> Value:
        return self.l0_inv_rhs() - self.nonlin(y)

class QuadEq(NonlinProblem):
    # solving a * y ^ 2 + b * y + c = 0
    # L0^{-1}[f] = f / b
    # L0^{-1}L[y] = y / b
    # sigma(y) = a * y * y
    # jacob(y) y1 = 2 * a * y * y1
    # hermit(y) y1 y2^T = 2 * a * y1 * y2
    def __init__(self, a: float, b: float, c: float) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def l0_inv_rhs(self) -> Value:
        return -self.c / self.b

    def nonlin(self, y: Value) -> Value:
        return self.a * y * y / self.b

class BurgersInvicid(NonlinProblem):
    # Burgers' invicid problem, assuming uniform sampling
    # solving ut + u * ux = 0, or ut + 1/2 * d(u^2)/dx = 0
    def __init__(self, u0: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> None:
        # u0, x: (nx,)
        # t: (nt,)
        # y: (nt, nx)
        super().__init__()
        self.u0 = u0  # (nx,)
        self.x = x
        self.t = t
        self.dx = x[1] - x[0]
        self.dt = t[1] - t[0]

    def l0_inv_rhs(self) -> torch.Tensor:
        return self.u0.expand(self.t.shape[0], -1)  # (nt, nx)

    def nonlin(self, y: torch.Tensor) -> torch.Tensor:
        # y: (nt, nx)
        # L1[y] = dy/dx
        # L0inv[y] = int_0^t(y * dt) + y(t=0)
        # assuming y(t=0) = 0 because this is operated only on y_i where i > 0
        # note that y_0(t=0) = u0

        y = 0.5 * y * y

        # compute dydx
        dydx_mid = (y[:, 2:] - y[:, :-2]) / (2 * self.dx)  # (nt, nx - 2)
        dydx_edge = (y[:, 1:] - y[:, :-1]) / self.dx  # (nt, nx - 1)
        dydx = torch.cat((dydx_edge[:, :1], dydx_mid, dydx_edge[:, -1:]), dim=-1)  # (nt, nx)

        # compute the cumulative sum in t direction
        cs_dydx = torch.cumsum(dydx, dim=0)  # (nt, nx)
        dydx_int_trapz = (cs_dydx - 0.5 * (dydx[:1, :] + dydx)) * self.dt
        return dydx_int_trapz

class SimplePendulum(NonlinProblem):
    # simulating the pendulum equations:
    # \ddot{theta} + (-fy * sin(theta) - fx * cos(theta)) = 0
    # but we're rewriting it to
    # (ddot{theta} + 2.5 * dot{theta} + theta) + (-fy * sin(theta) - fx * cos(theta) - 2.5 * dot{theta} - theta) = 0

    def __init__(self, t: torch.Tensor, f: torch.Tensor) -> None:
        super().__init__()
        # t: (nt,)
        # f: (2, nt)
        self._t = t
        self._f = f
    
    def l0_inv_rhs(self) -> Value:
        return self._f * 0

    def nonlin(self, y: torch.Tensor) -> torch.Tensor:
        # y: (2, nt)
        fx = self._f[0]  # (nt,)
        fy = self._f[1]
        theta = y[0]
        dtheta = y[1]
        b1 = -fy * torch.sin(theta) - fx * torch.cos(theta) - 2.5 * dtheta - theta  # (nt,)
        b = torch.stack((torch.zeros_like(b1), b1), dim=0)  # (2, nt)
        A = torch.tensor([[0.0, -1.0], [1.0, 2.5]], dtype=b1.dtype, device=b1.device)  # (2, 2)
        eival, u = torch.linalg.eig(A)  # eival: (2,), U: (2, 2)
        eival = eival.real
        u = u.real
        u1 = torch.linalg.inv(u)  # (2, 2)
        u1b = u1 @ b  # (2, nt)

        # apply the convolution
        aconv = torch.tensor([0.5, 2.0])[..., None]  # (2, 1)
        bconv = torch.tensor([1.0, 1.0])[..., None] / aconv  # (2, 1)
        q0 = bb.nn.E1Conv1D.calc_conv_lin(u1b[0], self._t, aconv[:1, :], bconv[:1, :])  # (nt,)
        q1 = bb.nn.E1Conv1D.calc_conv_lin(u1b[1], self._t, aconv[1:, :], bconv[1:, :])  # (nt,)
        q = torch.stack((q0, q1), dim=0)  # (2, nt)
        s = u @ q  # (2, nt)

        # res = torch.stack((torch.zeros_like(res1), res1), dim=0)  # (2, nt)
        return s

class _Counter:
    def __init__(self) -> None:
        self.a = 0

    def update(self) -> None:
        self.a += 1
        return self.a

def solve(prob: NonlinProblem) -> Value:
    x0 = prob.l0_inv_rhs()
    shape = x0.shape
    c = _Counter()

    def callback(y, f):
        print(c.update(), y.max(), np.abs(f).max())

    def rootfcn(xnp):
        x = torch.as_tensor(xnp)
        x = x.reshape(x0.shape)
        fx = prob.root_eq(x)
        fxnp = fx.detach().cpu().numpy().ravel()
        return fxnp

    res = equilibrium(fcn=prob.equil_eq, y0=x0, method="anderson_acc", x_rtol=1e-3, feat_ndims=2, verbose=True,
                      beta=0.1, msize=3, maxiter=1000)

    # res = root(rootfcn, x0=x0.ravel(), method="broyden1", callback=callback, tol=1e-3, options={"maxiter": 1000})
    # if not res.success:
    #     print(res)
    # res = torch.as_tensor(res.x)

    return res.reshape(shape)

if __name__ == "__main__":
    # prob = QuadEq(a=4.0, b=1.0, c=-1.0)
    # sol = solve(prob)
    # print(sol)
    # print(prob.root_eq(sol))

    # # BurgersInvicid
    # x = torch.linspace(-1, 1, 1000) * 40
    # t = torch.linspace(0, 4, 400)
    # u0 = torch.exp(-x * x / (2 * 4.5 ** 2))
    # prob = BurgersInvicid(u0, x, t)
    # sol = solve(prob).detach().cpu().numpy()
    # print(sol)
    # print(sol.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(x, sol[0])
    # plt.plot(x, sol[sol.shape[0] // 4])
    # plt.plot(x, sol[sol.shape[0] // 2])
    # plt.plot(x, sol[sol.shape[0] * 3 // 4])
    # plt.plot(x, sol[-1])
    # plt.savefig("fig.png")
    # plt.close()

    # Pendulum
    t = torch.linspace(0, 20, 10000)  # (nt,)
    f = torch.stack((t * 0 + 1, t * 0), dim=0)  # (2, nt)
    prob = SimplePendulum(t, f)
    sol = solve(prob).detach().cpu().numpy()
    print(sol)
    import matplotlib.pyplot as plt
    plt.plot(t, sol[0])
    plt.savefig("fig.png")
    plt.close()
