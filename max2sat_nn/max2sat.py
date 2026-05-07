"""MAX-2SAT formula representation and brute-force solver."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class Literal:
    """A literal: 1-based variable index ``var`` and ``negated`` flag."""

    var: int
    negated: bool

    def evaluate(self, assignment: Sequence[int]) -> int:
        v = assignment[self.var - 1]
        return 1 - v if self.negated else v

    def __repr__(self) -> str:
        return f"{'¬' if self.negated else ''}x{self.var}"


@dataclass(frozen=True)
class Clause:
    """A 2-literal clause ``l1 OR l2`` (``l1 == l2`` allowed for unit clauses)."""

    l1: Literal
    l2: Literal

    def evaluate(self, assignment: Sequence[int]) -> int:
        return int(self.l1.evaluate(assignment) | self.l2.evaluate(assignment))

    def __repr__(self) -> str:
        return f"({self.l1} ∨ {self.l2})"


@dataclass(frozen=True)
class Formula:
    n_vars: int
    clauses: Tuple[Clause, ...]

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def count_satisfied(self, assignment: Sequence[int]) -> int:
        return sum(c.evaluate(assignment) for c in self.clauses)

    def __repr__(self) -> str:
        return " ∧ ".join(repr(c) for c in self.clauses)


def assignment_from_index(k_idx: int, n: int) -> List[int]:
    """Decode ``k_idx`` in ``[0, 2^n)`` as a Boolean assignment (LSB-first)."""
    return [(k_idx >> (k - 1)) & 1 for k in range(1, n + 1)]


def brute_force_max2sat(phi: Formula) -> Tuple[int, List[int]]:
    """Return ``(MAX-2SAT(phi), best_assignment)`` by enumeration."""
    best = -1
    best_a: List[int] = []
    for k_idx in range(2 ** phi.n_vars):
        a = assignment_from_index(k_idx, phi.n_vars)
        s = phi.count_satisfied(a)
        if s > best:
            best, best_a = s, a
    return best, best_a


def random_formula(n_vars: int, n_clauses: int, rng: random.Random | None = None) -> Formula:
    rng = rng if rng is not None else random.Random(0)
    clauses: List[Clause] = []
    for _ in range(n_clauses):
        if n_vars >= 2:
            a, b = rng.sample(range(1, n_vars + 1), 2)
        else:
            a, b = 1, 1
        l1 = Literal(a, rng.random() < 0.5)
        l2 = Literal(b, rng.random() < 0.5)
        clauses.append(Clause(l1, l2))
    return Formula(n_vars=n_vars, clauses=tuple(clauses))
