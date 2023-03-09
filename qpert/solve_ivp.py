from typing import Callable, Sequence, Optional
import torch
import functorch
from qpert.diffeqs import BaseDiffEq


__all__ = ["solve_ivp"]

class IVP_Solve_DiffEq(BaseDiffEq):
    """
    A basic ODE with form dy/dx = f(y, x)
    """
    def __init__(self, func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 params: Sequence[torch.Tensor], *,
                 dtype: torch.dtype = torch.float64, device: torch.device = torch.device('cpu')):
        super().__init__(dtype=dtype, device=device)
        self._func = func
        self._params = params

    def func(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self._func(y, x, *self._params)

    def solve_lin(self, rhs: torch.Tensor, x: torch.Tensor, gy: torch.Tensor, yinit: torch.Tensor) -> torch.Tensor:
        # solving dy/dx + G @ y = rhs(x)
        # rhs: (..., nt, ny)
        # x: (..., nt, 1)
        # gy: (..., nt, ny, ny)
        # yinit: (..., 1, ny)

        # define functions that will be frequently used
        T = lambda x: x.transpose(-2, -1)
        bmm = lambda x, y: (x @ y[..., None])[..., 0]
        solve = lambda x, y: torch.linalg.solve(x, y[..., None])[..., 0]

        ny = rhs.shape[-1]
        if ny > 1:
            # add random noise to increase the chance of diagonalizability
            gy2 = gy  # + torch.randn_like(gy) * 1e-16
            # eival_g: (..., nt, ny), eivec_g: (..., nt, ny, ny)
            eival_g, eivec_g = torch.linalg.eig(gy2)

            # compute the right hand side (the argument of the inverse linear operator)
            # rhs: (..., nt, ny)
            wrhs = solve(eivec_g, torch.complex(rhs, torch.zeros_like(rhs)))  # (..., nt, ny)

            # compute the initial values
            uinit = solve(eivec_g, torch.complex(yinit, torch.zeros_like(yinit)))  # (..., nt, ny)

            # compute the convolution
            ut = T(conv_gt(T(wrhs), T(eival_g), T(uinit), T(x)))  # (..., nt, ny)
            res = bmm(eivec_g, ut)  # (..., nt, ny)
            res = res.real
        else:
            # rhs.T: (..., ny, nt), gt[..., 0].T: (..., ny, nt), y0.T: (..., ny, 1), tpts.T: (..., 1, nt)
            res = T(conv_gt(T(rhs), T(gy[..., 0]), T(yinit), T(x)))
        return res

class IVP_Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func: Callable[..., torch.Tensor], yinit: torch.Tensor, tpts: torch.Tensor,
                y0guess: Optional[torch.Tensor], *params: torch.Tensor):
        # yinit: (..., 1, ny)
        # tpts: (..., ntsamples, 1)
        diffeq = IVP_Solve_DiffEq(func, params, dtype=yinit.dtype, device=yinit.device)
        yshape = (*yinit.shape[:-2], tpts.shape[-2], yinit.shape[-1])
        if y0guess is None:
            y0 = torch.zeros(yshape, dtype=yinit.dtype, device=yinit.device)
        else:
            assert y0guess.shape == yshape
            y0 = y0guess
        ysol = diffeq.solve(y0, tpts, yinit=yinit)
        ctx.save_for_backward(yinit, ysol, tpts, *params)
        ctx._diffeq = diffeq
        ctx._func = func
        return ysol

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_ysol: torch.Tensor):
        # get the saved variables from forward
        diffeq = ctx._diffeq
        func = ctx._func
        yinit, ysol, tpts = ctx.saved_tensors[:3]
        params = ctx.saved_tensors[3:]

        # obtain the grad_f
        grad_f = diffeq.get_fgrad(grad_ysol, ysol, tpts, yinit=yinit)  # (..., ntsamples, ny)

        # propagate backward
        with torch.enable_grad():
            f = func(ysol.detach(), tpts, *params)
        grad_params = torch.autograd.grad(f, params, grad_outputs=grad_f)
        # divide by dt[0]/2, where the factor 2 comes from the fact that we're using trapezoidal method
        # to integrate
        grad_yinit = grad_f[..., :1, :] * (2 / (tpts[..., 1:2, :] - tpts[..., :1, :]))

        return None, grad_yinit, None, None, *grad_params

def solve_ivp(func: Callable[..., torch.Tensor], tpts: torch.Tensor, yinit: torch.Tensor,
              params: Sequence[torch.Tensor] = [], y0guess: Optional[torch.Tensor] = None) -> torch.Tensor:
    # solve the initial value problem
    # tpts: (..., ntpts)
    # yinit: (..., ny)
    # returns: (..., ntpts, ny)
    yinit = yinit.unsqueeze(-2)  # (..., 1, ny)
    tpts = tpts.unsqueeze(-1)  # (..., ntpts, 1)
    params = params[:]  # shallow copy

    # preprocess the function
    if isinstance(func, torch.nn.Module):
        assert len(params) == 0
        func_temp, params = functorch.make_functional(func)
        func2 = lambda y, t, *params: func_temp(params, y, t)
    else:
        func2 = func

    # apply the IVP solve
    ysol = IVP_Solve.apply(func2, yinit, tpts, y0guess, *params)
    return ysol

def conv_gt(rhs: torch.Tensor, gt: torch.Tensor, y0: torch.Tensor, tpts: torch.Tensor) -> torch.Tensor:
    # solve dy/dt + g(t) y = rhs(t) with y(0) = y0
    # rhs: (..., nt)
    # gt: (..., nt)
    # y0: (..., 1)
    # tpts: (..., nt)
    # return: (..., nt)

    # applying conv_gt(rhs) + conv_gt(y0 * delta(0))
    dt = tpts[..., 1:] - tpts[..., :-1]  # (..., nt - 1)
    half_dt = dt * 0.5

    # integrate gt with trapezoidal method
    trapz_area = (gt[..., :-1] + gt[..., 1:]) * half_dt  # (..., nt - 1)
    zero_pad = torch.zeros((*gt.shape[:-1], 1), dtype=gt.dtype, device=gt.device)  # (..., 1)
    gt_int = torch.cumsum(trapz_area, dim=-1)  # (..., nt - 1)
    gt_int = torch.cat((zero_pad, gt_int), dim=-1)  # (..., nt)

    # compute log[integral_0^t rhs(tau) * exp(gt_int(tau)) dtau] with trapezoidal method
    if not torch.is_complex(rhs):
        rhs = torch.complex(rhs, torch.zeros_like(rhs))
    exp_content = torch.log(rhs) + gt_int  # (..., nt)
    # TODO: change this to logaddexp
    exp_content2 = torch.stack((exp_content[..., :-1], exp_content[..., 1:]), dim=-1) + torch.log(half_dt)[..., None]  # (..., nt - 1, 2)
    trapz_area_gj = torch.logcumsumexp(exp_content2, dim=-1)[..., -1]  # (..., nt - 1)
    log_area_int = torch.logcumsumexp(trapz_area_gj, dim=-1)  # (..., nt - 1)
    log_conv = log_area_int - gt_int[..., 1:]  # (..., nt - 1)
    conv_res = torch.exp(log_conv).real
    # print(exp_content.shape, gt_int.shape, exp_content2.shape, trapz_area_gj.shape)
    conv_res = torch.cat((zero_pad, conv_res), dim=-1)  # (..., nt)

    # add the initial condition
    conv_res = conv_res + y0 * torch.exp(-gt_int)

    # results: exp(-gt_int(t)) * integral(t)
    # res = torch.exp(log_integral - gt_int)
    return conv_res
