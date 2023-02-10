from __future__ import annotations
from typing import Tuple, List, Union, Optional, Dict
from abc import abstractmethod
from collections import Counter


DEBUG = False

################ terms and collections of terms ################

class Term:
    @abstractmethod
    def can_operate(self, other: Term) -> bool:
        pass

    @abstractmethod
    def operate(self, other: Term, max_int: int = 999999) -> List[Tuple[Term, int]]:
        pass

    def negate(self) -> Term:
        pass

class Primitive(Term):
    def __init__(self, name: str):
        self.name = name
        self.coeff = 1.0
    
    def equal(self, other: Primitive):
        return self.name == other.name

    def can_operate(self, other: Term) -> bool:
        return False
    
    def operate(self, other: Term, max_int: int = 999999) -> List[Tuple[Term, int]]:
        assert False

    def negate(self) -> Term:
        assert False

    def __repr__(self) -> str:
        return self.name

class SigmaTerm(Term):
    # sigma_{idx}(y^{(operand_idx)})
    def __init__(self, coeff: float, idx: int, operand_idx: int) -> None:
        self.coeff = coeff
        self.idx = idx  # sigma index
        self.operand_idx = operand_idx  # index of y^{(i)} inside the bracket of sigma (the i)

    def can_operate(self, other: Term) -> bool:
        if not isinstance(other, SigmaTerm):
            return False
        if not (self.idx == other.idx):
            return False
        if self.operand_idx == other.operand_idx:
            return True
        return coeff_negat(self.coeff, other.coeff)

    def operate(self, other: SigmaTerm, max_int: int = 999999) -> List[Tuple[Term, int]]:
        # other is SigmaTerm with the same index
        if self.operand_idx == other.operand_idx:
            if coeff_negat(self.coeff, other.coeff):
                res = []
            else:
                res = [(SigmaTerm(self.coeff + other.coeff, idx=self.idx, operand_idx=self.operand_idx), 0)]
        elif self.operand_idx < other.operand_idx:
            res = other.operate(self, max_int=max_int)
        else:
            # self has higher operand_idx than other
            i = self.operand_idx
            j = other.operand_idx
            res = []
            if j + 1 > max_int:
                return res
            # the first order terms
            for k in range(j + 1, min(i + 1, max_int + 1)):
                term = JacobSigmaTerm(self.coeff, self.idx, operand_idx=i, aux_idx=k)
                res.append((term, k))
            # the second order terms
            for k in range(2 * j + 1, min(2 * i, max_int + 1)):
                pairs = []
                for l1 in range(j + 1, i + 1):
                    l2 = k - l1
                    if l2 < j + 1:
                        continue
                    pairs.append((min(l1, l2), max(l1, l2)))
                counts = Counter(pairs)
                for pair, count in counts.items():
                    term = HermitSigmaTerm(-self.coeff * count * 0.5, self.idx, self.operand_idx, aux_idxs=pair)
                    res.append((term, k))
        if DEBUG:
            print(f"Operating with max_int={max_int}:")
            print(f"* {self}")
            print(f"* {other}")
            print("Results:")
            print(res)
            print("")
        return res

    def negate(self) -> Term:
        return SigmaTerm(-self.coeff, self.idx, self.operand_idx)

    def __repr__(self) -> str:
        return f"{self.coeff} * sigma_{self.idx}(y^[{self.operand_idx}])"

class JacobSigmaTerm(Term):
    # sigma'_{idx}(y^{(operand_idx)}) y_{aux_idx}
    def __init__(self, coeff: float, idx: int, operand_idx: int, aux_idx: int) -> None:
        self.coeff = coeff
        self.idx = idx
        self.operand_idx = operand_idx
        self.aux_idx = aux_idx

    def can_operate(self, other: Term) -> bool:
        if not isinstance(other, JacobSigmaTerm):
            return False
        if not (self.idx == other.idx):
            return False
        if not (self.aux_idx == other.aux_idx):
            return False
        if self.operand_idx == other.operand_idx:
            return True
        return coeff_negat(self.coeff, other.coeff)

    def operate(self, other: JacobSigmaTerm, max_int: int = 999999) -> List[Tuple[Term, int]]:
        # other is JacobSigmaTerm with the same index, and coefficients are negative to each other
        if self.operand_idx == other.operand_idx:
            if coeff_negat(self.coeff, other.coeff):
                return []
            else:
                term = JacobSigmaTerm(self.coeff + other.coeff, idx=self.idx,
                                      operand_idx=self.operand_idx, aux_idx=self.aux_idx)
                return [(term, 0)]
        elif self.operand_idx < other.operand_idx:
            return other.operate(self)
        else:
            # self has higher operand_idx than other
            i = self.operand_idx
            j = other.operand_idx
            res = []
            if j + 1 > max_int:
                return res
            # the first order terms
            for k in range(j + 1, min(i + 1, max_int + 1)):
                term = HermitSigmaTerm(self.coeff, self.idx, operand_idx=i, aux_idxs=(k, self.aux_idx))
                res.append((term, k))
            return res

    def negate(self) -> Term:
        return JacobSigmaTerm(-self.coeff, self.idx, self.operand_idx, self.aux_idx)

    def __repr__(self) -> str:
        return f"{self.coeff} * sigma'_{self.idx}(y^[{self.operand_idx}]) y_{self.aux_idx}"

class HermitSigmaTerm(Term):
    def __init__(self, coeff: float, idx: int, operand_idx: int, aux_idxs: Tuple[int, int]) -> None:
        self.coeff = coeff
        self.idx = idx
        self.operand_idx = operand_idx
        self.aux_idxs = (min(aux_idxs), max(aux_idxs))

    def can_operate(self, other: Term) -> bool:
        if not isinstance(other, HermitSigmaTerm):
            return False
        if not (self.idx == other.idx):
            return False
        if not (self.aux_idxs == other.aux_idxs):
            return False
        if self.operand_idx == other.operand_idx:
            return True
        return coeff_negat(self.coeff, other.coeff)

    def operate(self, other: JacobSigmaTerm, max_int: int = 999999) -> List[Tuple[Term, int]]:
        # other is JacobSigmaTerm with the same index, and coefficients are negative to each other
        if self.operand_idx == other.operand_idx:
            if coeff_negat(self.coeff, other.coeff):
                return []
            else:
                term = HermitSigmaTerm(self.coeff + other.coeff, idx=self.idx,
                                       operand_idx=self.operand_idx, aux_idxs=self.aux_idxs)
                return [(term, 0)]
        elif self.operand_idx < other.operand_idx:
            return other.operate(self)
        else:
            return []  #  we assume sigma''' is 0

    def negate(self) -> Term:
        return HermitSigmaTerm(-self.coeff, self.idx, self.operand_idx, self.aux_idxs)

    def __repr__(self) -> str:
        return f"{self.coeff} * sigma''_{self.idx}(y^[{self.operand_idx}]) y_{self.aux_idxs[0]} y_{self.aux_idxs[1]}^T"

class MultiTerm:
    def __init__(self, terms: List[Union[Term, MultiTerm]]) -> None:
        self.terms = self._preprocess_terms(terms)

    def __len__(self) -> int:
        return len(self.terms)

    def _preprocess_terms(self, terms: List[Union[Term, MultiTerm]]) -> List[Term]:
        new_terms: List[Term] = []
        for term in terms:
            if isinstance(term, MultiTerm):
                new_terms.extend(term.terms)
            elif isinstance(term, Term):
                new_terms.append(term)
            else:
                raise TypeError(f"{type(term)}")
        return new_terms

    def substitute(self, prim: Primitive, term: Union[None, Term, MultiTerm]) -> MultiTerm:
        # substitute the primitive with a term or multiterm
        # or delete the prim if term == None
        new_terms = []
        for tm in self.terms:
            if isinstance(tm, Primitive) and tm.equal(prim):
                if term is not None:
                    new_terms.append(term)
            else:
                new_terms.append(tm)
        return MultiTerm(new_terms)

    def substitutes(self, prim_term_lst: List[Tuple[Primitive, Union[Term, MultiTerm]]]) -> MultiTerm:
        mt = self
        for prim, term in prim_term_lst:
            mt = mt.substitute(prim, term)
        return mt

    def negate(self) -> MultiTerm:
        return MultiTerm([term.negate() for term in self.terms])

    def simplify(self, max_int: int = 9999999) -> Dict[int, MultiTerm]:
        res_lst: Dict[int, List[Term]] = {}  # res_lst[0] is the remaining terms
        operated = [False for _ in range(len(self.terms))]
        for i0 in range(len(self.terms)):
            # check if it can be operated with another term
            operate_with: Optional[int] = None
            term0 = self.terms[i0]
            if operated[i0]:
                operate_with = -1
                continue
            for i1 in range(i0 + 1, len(self.terms)):
                term1 = self.terms[i1]
                if term0.can_operate(term1):
                    operate_with = i1
                    break
            # if term0 cannot be operated with another term, then list it as remaining terms
            if operate_with is None:
                if 0 not in res_lst:
                    res_lst[0] = []
                res_lst[0].append(term0)
            else:
                terms_ints = term0.operate(self.terms[operate_with], max_int=max_int)
                for term, move_idx in terms_ints:
                    if move_idx not in res_lst:
                        res_lst[move_idx] = []
                    res_lst[move_idx].append(term)
                # mark it as operated
                operated[operate_with] = True
        
        # convert res_lst to MultiTerm
        res = {}
        for key, terms in res_lst.items():
            res[key] = MultiTerm(terms)
        return res

    def __repr__(self) -> str:
        s = ""
        for i, term in enumerate(self.terms):
            if i == 0:
                s += str(term)
            elif term.coeff < 0:
                s += " - " + str(term)[1:]
            else:
                s += " + " + str(term)
        return s

class Equations:
    def __init__(self, eqs: Dict[int, MultiTerm]):
        self.eqs = eqs

    def simplify_once(self, max_order: int) -> Tuple[Equations, bool]:
        res1_lst: Dict[idx, List[Term]] = {}
        for idx, multiterm in self.eqs.items():
            max_int = max_order - idx
            mt_move = multiterm.simplify(max_int=max_int)  # dict: {move_idx: MultiTerm}
            for key, val in mt_move.items():
                key2 = idx + key
                if key2 not in res1_lst:
                    res1_lst[key2] = []
                res1_lst[key2].append(val)
        res = Equations({key: MultiTerm(mt_lst) for key, mt_lst in res1_lst.items() if key <= max_order})
        to_stop = str(self) == str(res)
        if DEBUG:
            print("simplify once results:")
            print(res)
        return res, to_stop

    def substitutes(self, prim_term_lst: List[Tuple[Primitive, Union[Term, MultiTerm]]]) -> Equations:
        eqs = {
            key: multiterm.substitutes(prim_term_lst) for key, multiterm in self.eqs.items()
        }
        return Equations(eqs)

    def simplify(self, max_order: int) -> Equations:
        eqs = self
        while True:
            eqs, to_stop = eqs.simplify_once(max_order=max_order)
            if to_stop:
                break
        return eqs

    def __repr__(self) -> str:
        return str(self.eqs)

################ solving equations ################

def generate_equations(max_order: int, *, mode: int = 1) -> Tuple[Equations, Primitive]:
    # generate the first equations of the given order and the primitive of the max_order
    assert max_order > 0
    if mode == 1:
        eqs = {}
        for ord in range(1, max_order + 1):
            prim = Primitive(f"y{ord}")
            eqs[ord] = MultiTerm([prim, SigmaTerm(coeff=1.0, idx=ord, operand_idx=max_order - 1)])
    elif mode == 2:
        eqs = {}
        for ord in range(1, max_order + 1):
            prim = Primitive(f"y{ord}")
            eqs[ord] = MultiTerm([prim, SigmaTerm(coeff=1.0, idx=ord, operand_idx=max_order - ord)])
    else:
        raise ValueError(f"Mode: {mode}")
    return Equations(eqs), prim

def solve_equations(max_order: int, *, mode: int = 1) -> List[Tuple[Primitive, MultiTerm]]:
    # solve "y{max_order}"
    substit_lst: List[Tuple[Primitive, MultiTerm]] = []

    # solve the equations order-by-order
    for order in range(1, max_order + 1):
        # solving the `order`-th order equation
        eqs_obj, prim = generate_equations(order, mode=mode)
        if DEBUG:
            print("")
            print(f"Solving the {order}-th order")
            print("generated eqs:")
            print(eqs_obj)
        eqs_obj = eqs_obj.substitutes(substit_lst)
        if DEBUG:
            if len(substit_lst) > 0:
                print("substituted eqs:")
                print(eqs_obj)
        eqs = eqs_obj.simplify(max_order=order).eqs
        assert len(eqs) == 1
        multiterm = eqs[list(eqs.keys())[0]]
        multiterm = multiterm.substitute(prim, None)  # remove the primitive
        substit_lst.append((prim, multiterm.negate()))
        if DEBUG:
            print("substitution list:")
            print(substit_lst)

    return substit_lst

####### helper functions #######
def coeff_negat(coeff1: float, coeff2: float) -> bool:
    # check if coeff1 == -coeff2
    return (coeff1 * coeff2 < 1 and abs(abs(coeff1) - abs(coeff2)) < 1e-7)

if __name__ == "__main__":
    DEBUG = False
    eqs = solve_equations(60, mode=1)
    print(eqs)
    print("")
    print(eqs[-1][1])
    print(len(eqs[-1][1]))
