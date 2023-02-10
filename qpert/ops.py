from typing import Callable, Dict, List
from abc import abstractmethod
from qpert.terms import Term, SigmaTerm, JacobSigmaTerm, HermitSigmaTerm, solve_equations
from qpert.utils import Value


class Evaluator:
    def eval(self, term: Term, ysup: Dict[int, Value], ysub: Dict[int, Value]) -> Value:
        if isinstance(term, SigmaTerm):
            val0 = ysup[term.operand_idx]
            return self.eval_sigma(term.idx, val0) * term.coeff
        elif isinstance(term, JacobSigmaTerm):
            val0 = ysup[term.operand_idx]
            val1 = ysub[term.aux_idx]
            return self.eval_jacob_sigma(term.idx, val0, val1) * term.coeff
        elif isinstance(term, HermitSigmaTerm):
            val0 = ysup[term.operand_idx]
            val1 = ysub[term.aux_idxs[0]]
            val2 = ysub[term.aux_idxs[1]]
            return self.eval_hermit_sigma(term.idx, val0, val1, val2) * term.coeff
        else:
            raise TypeError(f"{type(term)}")

    @abstractmethod
    def eval_sigma(self, term_idx: int, val0: Value) -> Value:
        # sigma(val0)
        pass

    @abstractmethod
    def eval_jacob_sigma(self, term_idx: int, val0: Value, val1: Value) -> Value:
        # sigma'(val0) val1
        pass

    @abstractmethod
    def eval_hermit_sigma(self, term_idx: int, val0: Value, val1: Value, val2: Value) -> Value:
        # sigma''(val0) val1 val2^T
        pass

class SameSigmaEval(Evaluator):
    # evaluator where all the sigmas are the same and the hermitian is constant
    def __init__(self, l0_inv_l1: Callable[[Value], Value],
            nonlin: Callable[[Value], Value],  # sigma(y)
            jacob_nonlin: Callable[[Value, Value], Value],  # sigma'(y0) y1
            hermit_nonlin: Callable[[Value, Value, Value], Value],  # sigma''(y0) y1 y2^T
            ) -> None:
        self.l0_inv_l1 = l0_inv_l1
        self.nonlin = nonlin
        self.jacob_nonlin = jacob_nonlin
        self.hermit_nonlin = hermit_nonlin

    def eval_sigma(self, term_idx: int, val0: Value) -> Value:
        # sigma(val0)
        y0 = self.nonlin(val0)
        y = self.l0_inv_l1(y0)
        return y

    def eval_jacob_sigma(self, term_idx: int, val0: Value, val1: Value) -> Value:
        # sigma'(val0) val1
        y0 = self.jacob_nonlin(val0, val1)
        y = self.l0_inv_l1(y0)
        return y

    def eval_hermit_sigma(self, term_idx: int, val0: Value, val1: Value, val2: Value) -> Value:
        # sigma'' val1 val2^T
        y0 = self.hermit_nonlin(val0, val1, val2)
        y = self.l0_inv_l1(y0)
        return y

def eval_equation_one_nonlin(
        max_order: int, rhs_val: Value, l0_inv: Callable[[Value], Value], l0_inv_l1: Callable[[Value], Value],
        nonlin: Callable[[Value], Value],  # sigma(y)
        jacob_nonlin: Callable[[Value, Value], Value],  # sigma'(y0) y1
        hermit_nonlin: Callable[[Value, Value, Value], Value],  # sigma'' y1 y2^T
        *, mode: int = 1, epsilon: float = 0.5) -> List[Value]:
    # solving the equation of L0[y] + L1[sigma(y)] = f

    prims_eqs = solve_equations(max_order, mode=mode)  # get all the equations
    mult = (1 - epsilon) / epsilon
    l0_inv_l1_2 = lambda x: l0_inv_l1(x) * mult
    evaluator = SameSigmaEval(l0_inv_l1_2, nonlin, jacob_nonlin, hermit_nonlin)

    # get the first solution
    ysup = [l0_inv(rhs_val)]  # y^{(key)}
    ysub = [ysup[0]]  # y_1
    for i in range(len(prims_eqs)):
        order = i + 1
        _, multiterm = prims_eqs[i]
        y_o = sum([evaluator.eval(term, ysup, ysub) for term in multiterm.terms])
        assert len(ysub) == order
        assert len(ysup) == order
        ysub.append(y_o)
        ysup.append(ysup[order - 1] + epsilon ** order * y_o)

    return ysup

if __name__ == "__main__":
    a = 3.0
    rhs = 1.0
    # solve: a * y ^ 2 + y = rhs
    vals = eval_equation_one_nonlin(
        max_order=40,
        rhs_val=rhs,
        l0_inv=lambda x: x,
        l0_inv_l1=lambda x: x,
        nonlin=lambda x: a * x * x,
        jacob_nonlin=lambda x, y: 2 * a * x * y,
        hermit_nonlin=lambda x, y, z: 2 * a * y * z,
        epsilon=0.9,
        # mode=2
    )
    print(vals)
    sol = (-1 + (1 + 4 * a * rhs) ** 0.5) / (2 * a)
    print(sol)
    print(sol ** 2 * a + sol)
    print(vals[-1] ** 2 * a + vals[-1])
