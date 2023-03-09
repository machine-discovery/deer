import time
import itertools
import torch
import pytest
from scipy.integrate import solve_ivp as sp_solve_ivp
from qpert.solve_ivp import solve_ivp
from qpert.tests.utils import get_avail_devices


@pytest.mark.parametrize('device, test_type', itertools.product(get_avail_devices(), ['forward', 'grad']))
def test_solve_ivp_1var(device, test_type):
    torch.manual_seed(123)
    dtype = torch.float64

    def func(y, t, a):
        return -a * y ** 3 + torch.sin(60 * t)

    npts = 10000
    tpts0 = torch.linspace(0, 10, npts, dtype=dtype, device=device)  # (ntpts,)
    tpts1 = torch.rand(npts // 10, dtype=dtype, device=device) * (tpts0[1] - tpts0[0])
    tpts = torch.sort(torch.cat((tpts0, tpts1), dim=-1))[0]
    y0 = torch.randn(1, dtype=dtype, device=device, requires_grad=True)  # (ny,)
    a = torch.tensor(10., dtype=dtype, device=device, requires_grad=True)
    params = (a,)
    solve_ivp_test_helper(func, tpts, y0, params=params, test_type=test_type)

@pytest.mark.parametrize('device, test_type', itertools.product(get_avail_devices(), ['forward', 'grad']))
def test_solve_ivp_multi_var(device, test_type):
    torch.manual_seed(123)
    dtype = torch.float64

    def func(y, t, a):
        return (y[None] @ a.transpose(-2, -1))[0] + torch.sin(60 * t)

    npts = 100000
    tpts0 = torch.linspace(0, 10, npts, dtype=dtype, device=device)  # (ntpts,)
    tpts1 = torch.rand(npts // 10, dtype=dtype, device=device) * (tpts0[1] - tpts0[0])
    tpts = torch.sort(torch.cat((tpts0, tpts1), dim=-1))[0]
    y0 = torch.randn(2, dtype=dtype, device=device, requires_grad=True)  # (ny,)
    a = torch.tensor([[0, 1.0], [-100., -101.0]], dtype=dtype, device=device, requires_grad=True)
    params = (a,)
    solve_ivp_test_helper(func, tpts, y0, params=params, test_type=test_type)

def solve_ivp_test_helper(func, tpts, y0, params, test_type,
                          sp_atol=1e-8, sp_rtol=1e-7):
    # test our solve_ivp vs scipy's solve_ivp
    dtype = y0.dtype
    device = y0.device

    if test_type == 'forward':
        t0 = time.time()
        ysol = solve_ivp(func, tpts, y0, params=params)
        t1 = time.time()
        print("   This solve_ivp time (s):", t1 - t0)

        sp_func = make_scipy_solve_ivp_func(func, params)
        t0 = time.time()
        ysol2 = sp_solve_ivp(sp_func, t_span=(tpts[0].cpu(), tpts[-1].cpu()), y0=y0.cpu().detach(),
                            t_eval=tpts.cpu().detach(), atol=sp_atol, rtol=sp_rtol).y.T
        ysol2 = torch.as_tensor(ysol2, device=device, dtype=dtype)
        t1 = time.time()
        print("Scipy's solve_ivp time (s):", t1 - t0)

        print("Max abs dev:", (ysol - ysol2).abs().max())
        assert torch.allclose(ysol, ysol2, rtol=1e-4, atol=1e-5)

    elif test_type == 'grad':
        # grad check
        def solve_ivp_grad_wrapped(func, tpts, y0, *params):
            ysol = solve_ivp(func, tpts, y0, params=params)
            return (ysol ** 2).sum()
        # res = solve_ivp_grad_wrapped(y0, *params)
        # res.backward()
        # print(y0.grad)
        # print(params[0].grad)
        # assert False
        assert torch.autograd.gradcheck(solve_ivp_grad_wrapped, (func, tpts, y0, *params))

    else:
        assert False, f"Invalid test_type: {test_type}"

def make_scipy_solve_ivp_func(func, params):
    def scipy_func(t_np, y_np):
        t = torch.as_tensor(t_np)
        y = torch.as_tensor(y_np)
        fy = func(y, t, *params)
        return fy.detach().numpy()
    return scipy_func





def atest_solve_ivp_multi_var1(device, test_type):
    torch.manual_seed(123)
    dtype = torch.float64

    def func(y, t, a):
        return (y[None] @ a.transpose(-2, -1))[0] + torch.sin(60 * t)

    npts = 10000
    tpts0 = torch.linspace(0, 10, npts, dtype=dtype, device=device)  # (ntpts,)
    tpts1 = torch.rand(npts // 10, dtype=dtype, device=device) * (tpts0[1] - tpts0[0])
    tpts = torch.sort(torch.cat((tpts0, tpts1), dim=-1))[0]
    y0 = torch.randn(2, dtype=dtype, device=device, requires_grad=True)  # (ny,)
    a = torch.tensor([[0, 1.0], [-10., -11.0]], dtype=dtype, device=device, requires_grad=True)
    params = (a,)
    solve_ivp_test_helper(func, tpts, y0, params=params, test_type=test_type)

if __name__ == "__main__":
    with torch.autograd.detect_anomaly():
        atest_solve_ivp_multi_var1(torch.device('cpu'), 'grad')
