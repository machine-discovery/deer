from typing import List, Tuple, Dict
import sympy

def get_eps() -> sympy.Symbol:
    return sympy.symbols("eps", real=True, positive=True)

def get_sigma() -> sympy.Function:
    return sympy.Function("sigma")

def get_a_coeff(i: int) -> sympy.Symbol:
    return sympy.symbols(f"a{i}", real=True)

def generate_function(order: int, mode: int = 1) -> Tuple[sympy.Expr, sympy.Symbol]:
    # generate the first equation of the given order
    funcs = {i: get_sigma() for i in range(1, order + 1)}
    coeffs = {i: get_a_coeff(i) for i in range(1, order + 1)}
    ys = {i: sympy.symbols(f"y{i}") for i in range(order + 1)}
    yys = {i: sympy.symbols(f"yy{i}") for i in range(order + 1)}
    eps = get_eps()
    expr = 0
    if mode == 1:
        for i in range(1, order + 1):
            expr += (ys[i] + coeffs[i] * funcs[i](yys[order - 1])) * eps ** (i - 1)
    elif mode == 2:
        for i in range(1, order + 1):
            expr += (ys[i] + coeffs[i] * funcs[i](yys[order - i])) * eps ** (i - 1)
    else:
        raise ValueError
    return expr, ys[order], yys

def solve_equations(order: int, mode: int = 1) -> List[Tuple[sympy.Symbol, sympy.Expr]]:
    # solve n-equations up to the given order, solving for yi, except at 0
    substit_lst: List[Tuple[sympy.Symbol, sympy.Expr]] = []
    for iord in range(1, order + 1):
        expr, yord, yys = generate_function(iord, mode=mode)

        # substitute the variables
        expr2 = expr.subs(substit_lst)

        # solve for the given order
        expr3 = sympy.solve(expr2, yord)
        assert len(expr3) == 1
        seps = [get_sigma()(yy) for yy in yys.values()]
        substit_lst.append((yord, sympy.separatevars(expr3[0], symbols=seps)))

    return substit_lst

def get_yyj_expr(order: int, mode: int = 1) -> List[sympy.Expr]:
    yi_eqs = solve_equations(order, mode=mode)

def get_coeffs(exprs: List[Tuple[sympy.Symbol, sympy.Expr]]) -> Dict[int, Dict[int, sympy.Expr]]:
    # get the coefficients of eps**i yi, collected by sigma(yyj)
    # the output of this is a nested dictionary, res[i][j]
    # where it expresses the equation:
    # (eps ** imax * y_imax) = sum_i (sum_j (res[i][j] * sigma(yyj)))
    order = len(exprs)
    res = {}
    for ord in range(1, order + 1):
        res[ord] = {}
        expr = exprs[ord - 1][1]
        for j in range(ord):
            subs_lst = []
            for k in range(ord):
                fcn = get_sigma()(sympy.symbols(f"yy{k}"))
                subs_lst.append((fcn, 1 if k == j else 0))
            coeff = expr.subs(subs_lst)
            coeff = coeff * get_eps() ** ord
            res[ord][j] = coeff
    return res

def get_coeffs_reduced(order: int, mode: int = 1) -> Dict[int, sympy.Expr]:
    # yy_order = res[j] * sigma(yyj)
    # where yyj = sum_{k=0}^j (eps ** k * yk)
    coeffs_dict2 = get_coeffs(solve_equations(order, mode=mode))
    res = {}
    for i, coeffs_dct in coeffs_dict2.items():
        for j, expr in coeffs_dct.items():
            if j not in res:
                res[j] = 0
            res[j] += expr
    
    for j in res:
        res[j] = sympy.simplify(res[j])
    return res


if __name__ == "__main__":
    # eqs = get_coeffs(solve_equations(order=4))
    eqs = get_coeffs_reduced(order=6, mode=2)
    print(eqs)
