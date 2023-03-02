from typing import Callable
import torch
import beeblo as bb
import functorch


def func_and_jac(func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], yt: torch.Tensor, tpts: torch.Tensor):
    # func(y, t) -> dy/dt: [(..., ny), (..., 1)] -> (..., ny)
    # yt: (..., nt, ny)
    # tpts: (..., nt, 1)
    # returns: (..., nt, ny) and (..., nt, ny, ny)
    nbatch_dims = yt.ndim - 1
    jacfunc = functorch.jacrev(func, argnums=0)  # [(ny), (1)] -> (ny, ny)
    for _ in range(nbatch_dims):
        jacfunc = functorch.vmap(jacfunc)

    # compute the function and its jacobian
    fyt = func(yt, tpts)
    jac_fyt = jacfunc(yt, tpts)
    return fyt, jac_fyt

def conv_gt(rhs: torch.Tensor, gt: torch.Tensor, y0: torch.Tensor, tpts: torch.Tensor) -> torch.Tensor:
    # rhs: (..., nt, ny=1)
    # gt: (..., nt, ny=1)
    # y0: (..., 1, ny)
    # tpts: (..., nt, 1)
    # return: (..., nt, ny)
    # applying conv_gt(rhs) + conv_gt(y0 * delta(0))
    assert rhs.shape[-1] == gt.shape[-1] == y0.shape[-1] == 1
    dt = tpts[..., 1:, :] - tpts[..., :-1, :]  # (..., nt - 1, 1)
    half_dt = dt * 0.5

    # integrate gt with trapezoidal method
    trapz_area = (gt[..., :-1, :] + gt[..., 1:, :]) * half_dt  # (..., nt - 1, ny)
    zero_pad = torch.zeros((*gt.shape[:-2], 1, gt.shape[-1]), dtype=gt.dtype, device=gt.device)  # (..., 1, 1)
    gt_int = torch.cumsum(trapz_area, dim=-2)
    gt_int = torch.cat((zero_pad, gt_int), dim=-2)  # (..., nt, ny)
    
    # compute log[integral_0^t rhs(tau) * exp(gt_int(tau)) dtau] with trapezoidal method
    exp_content = torch.log(torch.complex(rhs, torch.zeros_like(rhs))) + gt_int  # (..., nt, ny)
    exp_content2 = torch.stack((exp_content[..., :-1, :], exp_content[..., 1:, :]), dim=-1) + torch.log(half_dt)[..., None]  # (..., nt - 1, ny, 2)
    trapz_area_gj = torch.logcumsumexp(exp_content2, dim=-1)[..., -1]  # (..., nt - 1, ny)
    log_area_int = torch.logcumsumexp(trapz_area_gj, dim=-2)  # (..., nt - 1, ny)
    log_conv = log_area_int - gt_int[..., 1:, :]  # (..., nt - 1, ny)
    conv_res = torch.exp(log_conv).real
    # print(exp_content.shape, gt_int.shape, exp_content2.shape, trapz_area_gj.shape)
    conv_res = torch.cat((zero_pad, conv_res), dim=-2)  # (..., nt, ny)

    # results: exp(-gt_int(t)) * integral(t)
    # res = torch.exp(log_integral - gt_int)
    return conv_res

def solve_ivp(func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], y0: torch.Tensor, tpts: torch.Tensor):
    # func(y, t) -> dy/dt: [(..., ny), (..., 1)] -> (..., ny)
    # y0: (..., ny)
    # tpts: (..., nt)
    # returns: (..., nt, ny)
    ny = y0.shape[-1]
    assert ny == 1, "Only y with 1 channel is supported now"
    y0 = y0.unsqueeze(-2)  # (..., 1, ny)
    tpts = tpts.unsqueeze(-1)  # (..., nt, 1)

    # first guess: all zeros
    # yt0: (..., nt, ny)
    yt = torch.zeros((*y0.shape[:-2], tpts.shape[-2], y0.shape[-1]), dtype=y0.dtype, device=y0.device)
    converge = False
    for i in range(100):
        # fyt0: (..., nt, ny), jac_fyt0: (..., nt, ny, ny)
        fyt, jac_fyt = func_and_jac(func, yt, tpts)
        if ny == 1:
            gt = -jac_fyt[..., 0]
            gty = gt * yt  # (..., nt, ny)
        else:
            assert False
        
        rhs = fyt + gty  # (..., nt, ny)
        yt_new = conv_gt(rhs, gt, y0, tpts)

        diff = torch.mean(torch.abs(yt_new - yt))
        print(diff)
        yt = yt_new
        if diff < 1e-6:
            converge = True
            break

    if not converge:
        print("Does not converge")
    return yt

if __name__ == "__main__":
    dtype = torch.float64
    device = torch.device('cuda')
    module = bb.nn.MLP(1, 1).to(dtype).to(device)
    def func(y, t):
        # (..., ny), (..., 1) -> (..., ny)
        return -module(y) * 60 * y - 10 * y ** 3 + torch.sin(600 * t)

    def fun(t, y):
        y = torch.as_tensor(y)
        t = torch.as_tensor(t)
        fy = func(y, t)
        return fy.detach().numpy()

    npts = 100000
    tpts = torch.linspace(0, 10, npts, dtype=dtype, device=device)  # (ntpts,)
    y0 = torch.zeros(1, dtype=dtype, device=device)  # (ny=1,)
    import time
    t0 = time.time()
    with torch.no_grad():
        yt = solve_ivp(func, y0, tpts).detach()  # (ntpts, ny)
    t1 = time.time()
    print(t1 - t0)

    from scipy.integrate import solve_ivp as solve_ivp2
    module = module.to(torch.device('cpu'))
    t0 = time.time()
    res = solve_ivp2(fun, t_span=(tpts[0].cpu(), tpts[-1].cpu()), y0=y0.cpu(), t_eval=tpts.cpu(), atol=1e-6, rtol=1e-7)
    t1 = time.time()
    print(t1 - t0)

    import matplotlib.pyplot as plt
    plt.plot(tpts.cpu(), yt.cpu()[..., 0])
    plt.plot(res.t, res.y[0])
    plt.savefig("fig.png")


"""
dy/dt = f(y, t)
dydt + g(t) * y = f(y, t) + g(t) * y
y = conv_gt(f(y, t) + g(t) * y) + conv_gt(y0 * delta(0))
g(t) = -df / dy
"""
