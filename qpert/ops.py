from typing import Callable, Dict, List
from abc import abstractmethod
from qpert.terms import Term, SigmaTerm, JacobSigmaTerm, HermitSigmaTerm, MultiTerm
from qpert.utils import Value


class Evaluator:
    def eval_multiterm(self, multiterm: MultiTerm, ysup: Dict[int, Value], ysub: Dict[int, Value]) -> Value:
        return sum([self.eval(term, ysup, ysub) for term in multiterm.terms])

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
    # and the L0_inv_l1 are all the same
    def __init__(self, l0_inv_l1: Callable[[Value], Value],
            nonlin: Callable[[Value], Value],  # sigma(y)
            jacob_nonlin: Callable[[Value, Value], Value],  # sigma'(y0) y1
            hermit_nonlin: Callable[[Value, Value, Value], Value],  # sigma''(y0) y1 y2^T
            *,
            epsilon: float = 0.5,
            epsilon_mode: int = 1,
            ) -> None:
        self.l0_inv_l1 = l0_inv_l1
        self.nonlin = nonlin
        self.jacob_nonlin = jacob_nonlin
        self.hermit_nonlin = hermit_nonlin
        self.epsilon_mode = epsilon_mode

        if self.epsilon_mode == 1:
            self.mult = (1 - epsilon) / epsilon
        elif self.epsilon_mode == 2:
            self.mult = (1 - epsilon) ** 2 / epsilon
        else:
            raise ValueError(f"Invalid epsilon mode: {self.epsilon_mode}")

    def eval_multiterm(self, multiterm: MultiTerm, ysup: Dict[int, Value], ysub: Dict[int, Value]) -> Value:
        sum_all_sigmas = super().eval_multiterm(multiterm, ysup, ysub)
        y = sum_all_sigmas * self.mult
        y = self.l0_inv_l1(y)
        return y

    def eval_sigma(self, term_idx: int, val0: Value) -> Value:
        # sigma(val0)
        y = self.nonlin(val0)
        if self.epsilon_mode == 2:
            y = y * term_idx
        return y

    def eval_jacob_sigma(self, term_idx: int, val0: Value, val1: Value) -> Value:
        # sigma'(val0) val1
        y = self.jacob_nonlin(val0, val1)
        if self.epsilon_mode == 2:
            y = y * term_idx
        return y

    def eval_hermit_sigma(self, term_idx: int, val0: Value, val1: Value, val2: Value) -> Value:
        # sigma'' val1 val2^T
        y = self.hermit_nonlin(val0, val1, val2)
        if self.epsilon_mode == 2:
            y = y * term_idx
        return y
