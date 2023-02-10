from typing import Callable
from abc import abstractmethod
from qpert.terms import Term, SigmaTerm, JacobSigmaTerm, HermitSigmaTerm, solve_equations
from qpert.utils import Value


class Evaluator:
    def eval(self, term: Term, *val: Value) -> Value:
        if isinstance(term, SigmaTerm):
            assert len(val) == 1
            return self.eval_sigma(term, val[0])
        elif isinstance(term, JacobSigmaTerm):
            assert len(val) == 2
            return self.eval_jacob_sigma(term, val[0], val[1])
        elif isinstance(term, HermitSigmaTerm):
            assert len(val) == 3
            return self.eval_hermit_sigma(term, val[0], val[1], val[2])
        else:
            raise TypeError(f"{type(term)}")

    @abstractmethod
    def eval_sigma(self, term: SigmaTerm, val0: Value) -> Value:
        # sigma(val0)
        pass

    @abstractmethod
    def eval_jacob_sigma(self, term: JacobSigmaTerm, val0: Value, val1: Value) -> Value:
        # sigma'(val0) val1
        pass

    @abstractmethod
    def eval_hermit_sigma(self, term: JacobSigmaTerm, val0: Value, val1: Value, val2: Value) -> Value:
        # sigma''(val0) val1 val2^T
        pass

class SameSigmaEval(Evaluator):
    # evaluator where all the sigmas are the same and the hermitian is constant
    def __init__(self, l0_inv_l1: Callable[[Value], Value],
            nonlin: Callable[[Value], Value],  # sigma(y)
            jacob_nonlin: Callable[[Value, Value], Value],  # sigma'(y0) y1
            hermit_nonlin: Callable[[Value, Value], Value],  # sigma'' y1 y2^T
            ) -> None:
        self.l0_inv_l1 = l0_inv_l1
        self.nonlin = nonlin
        self.jacob_nonlin = jacob_nonlin
        self.hermit_nonlin = hermit_nonlin

    def eval_sigma(self, term: SigmaTerm, val0: Value) -> Value:
        # sigma(val0)
        y0 = self.nonlin(val0)
        y = self.l0_inv_l1(y0)
        return y

    def eval_jacob_sigma(self, term: JacobSigmaTerm, val0: Value, val1: Value) -> Value:
        # sigma'(val0) val1
        y0 = self.jacob_nonlin(val0, val1)
        y = self.l0_inv_l1(y0)
        return y

    def eval_hermit_sigma(self, term: HermitSigmaTerm, val0: Value, val1: Value, val2: Value) -> Value:
        # sigma'' val1 val2^T
        y0 = self.hermit_nonlin(val1, val2)
        y = self.l0_inv_l1(y0)
        return y

def eval_equation_one_nonlin(
        max_order: int, rhs_val: Value, l0_inv: Callable[[Value], Value], l0_inv_l1: Callable[[Value], Value],
        nonlin: Callable[[Value], Value],  # sigma(y)
        jacob_nonlin: Callable[[Value, Value], Value],  # sigma'(y0) y1
        hermit_nonlin: Callable[[Value, Value], Value],  # sigma'' y1 y2^T
        *, mode: int = 1):
    # solving the equation of L0[y] + L1[sigma(y)] = f
    prims_eqs = solve_equations(max_order, mode=mode)
    evaluator = SameSigmaEval(l0_inv_l1, nonlin, jacob_nonlin, hermit_nonlin)
    pass
