from typing import List
from abc import abstractmethod
import time
import numpy as np
import torch
from qpert.utils import Value
from qpert.terms import solve_equations
from qpert.ops import SameSigmaEval

# List of problems

class QuadProblem:
    @abstractmethod
    def l0_inv_rhs(self) -> Value:
        # L0^{-1}[rhs]
        pass

    @abstractmethod
    def l0_inv_l1(self, y: Value) -> Value:
        # L0^{-1}L[y]
        pass

    @abstractmethod
    def nonlin(self, y: Value) -> Value:
        pass

    @abstractmethod
    def jacob_nonlin(self, y: Value, y1: Value) -> Value:
        pass

    @abstractmethod
    def hermit_nonlin(self, y: Value, y1: Value, y2: Value) -> Value:
        pass

class QuadEq(QuadProblem):
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

    def l0_inv_l1(self, y: Value) -> Value:
        return y / self.b

    def nonlin(self, y: Value) -> Value:
        return self.a * y * y

    def jacob_nonlin(self, y: Value, y1: Value) -> Value:
        return 2 * self.a * y * y1

    def hermit_nonlin(self, y: Value, y1: Value, y2: Value) -> Value:
        return 2 * self.a * y1 * y2

class BurgersInvicid(QuadProblem):
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

    def l0_inv_l1(self, y: torch.Tensor) -> torch.Tensor:
        # y: (nt, nx)
        # L1[y] = dy/dx
        # L0inv[y] = int_0^t(y * dt) + y(t=0)
        # assuming y(t=0) = 0 because this is operated only on y_i where i > 0
        # note that y_0(t=0) = u0

        # compute dydx
        dydx_mid = (y[..., 2:] - y[..., :-2]) / (2 * self.dx)  # (nt, nx - 2)
        dydx_edge = (y[..., 1:] - y[..., :-1]) / self.dx  # (nt, nx - 1)
        dydx = torch.cat((dydx_edge[..., :1], dydx_mid, dydx_edge[..., -1:]), dim=-1)  # (nt, nx)

        # compute the cumulative sum in t direction
        cs_dydx = torch.cumsum(dydx, dim=0)  # (nt, nx)
        dydx_int_trapz = (cs_dydx - 0.5 * (dydx[:1, :] + dydx)) * self.dt
        return dydx_int_trapz

    def nonlin(self, y: torch.Tensor) -> torch.Tensor:
        return 0.5 * y * y

    def jacob_nonlin(self, y: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
        # return y
        return y * y1

    def hermit_nonlin(self, y: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        # return y1 * y2 * 0 + 1
        return y1 * y2

def eval_equation_quad_prob(
        max_order: int, quad_prob: QuadProblem,
        *, mode: int = 1, epsilon: float = 0.5, epsilon_mode: int = 1) -> List[Value]:
    # solving the equation of L0[y] + L1[sigma(y)] = f

    t0 = time.time()
    prims_eqs = solve_equations(max_order, mode=mode)  # get all the equations
    t1 = time.time()
    print("Analytic work: ", t1 - t0)
    evaluator = SameSigmaEval(
        quad_prob.l0_inv_l1, quad_prob.nonlin, quad_prob.jacob_nonlin, quad_prob.hermit_nonlin,
        epsilon=epsilon, epsilon_mode=epsilon_mode)

    # get the first solution
    ysup = [quad_prob.l0_inv_rhs()]  # y^{(key)}
    ysub = [ysup[0]]  # y_1
    for i in range(len(prims_eqs)):
        order = i + 1
        _, multiterm = prims_eqs[i]
        y_o = evaluator.eval_multiterm(multiterm, ysup, ysub)
        # y_o = sum([evaluator.eval(term, ysup, ysub) for term in multiterm.terms])
        assert len(ysub) == order
        assert len(ysup) == order
        ysub.append(y_o)
        ysup.append(ysup[order - 1] + epsilon ** order * y_o)
    t2 = time.time()
    print("Computation  : ", t2 - t1)

    return ysup

if __name__ == "__main__":
    dtype = torch.float64
    x = torch.linspace(-1, 1, 1000, dtype=dtype) * 40
    t = torch.linspace(0, 5, 200, dtype=dtype)
    # u0 = 1 + 2 / np.pi * torch.arctan(-x)
    u0 = torch.exp(-x * x / (2 * 4.5 ** 2))
    prob = BurgersInvicid(u0, x, t)

    # a = 1.0
    # c = -1.0
    # prob = QuadEq(a=a, b=1.0, c=c)
    vals = eval_equation_quad_prob(
        max_order=40,
        quad_prob=prob,
        mode=2,
        epsilon=0.9,
        epsilon_mode=2,
    )
    print(vals[-1])  # (nt, nx)
    v = vals[-1].cpu().detach().numpy()
    import matplotlib.pyplot as plt
    plt.plot(x, v[0])
    plt.plot(x, v[v.shape[0] // 2])
    plt.plot(x, v[-1])
    plt.savefig("fig.png")
    plt.close()

    # print(vals)
    # sol = (-1 + (1 - 4 * a * c) ** 0.5) / (2 * a)
    # print(sol)
    # print(sol ** 2 * a + sol)
    # print(vals[-1] ** 2 * a + vals[-1])
