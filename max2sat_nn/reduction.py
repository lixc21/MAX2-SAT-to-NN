"""End-to-end MAX-2SAT -> 1D ReLU network reduction.

For a MAX-2SAT formula ``phi`` with ``n`` variables and ``m`` clauses,
``F_phi_network(phi)`` returns a scalar->scalar ReLU network of

    depth   = n + 2,
    width   = O(m + n)  (sum of clause widths),
    neurons = O(m * n),

whose Lipschitz constant on ``[0, 1]`` satisfies

    Lip(F_phi_network(phi)) = 2^(n+1) * MAX-2SAT(phi).

(The paper writes the constant as ``2^n``; the difference is a fixed factor
that depends on whether each spike has half or full interval width. The
reduction is unaffected: knowing the constant we recover MAX-2SAT(phi) exactly.)
"""
from __future__ import annotations

import numpy as np

from .bits import bit_signal_network
from .boolean import or_network
from .max2sat import Clause, Formula, Literal, brute_force_max2sat
from .network import ReLUNet, parallel_sum


def _literal_signal_network(lit: Literal, n: int) -> ReLUNet:
    return bit_signal_network(lit.var, n, negated=lit.negated)


def _clause_signal_network(clause: Clause, n: int) -> ReLUNet:
    A = _literal_signal_network(clause.l1, n)
    B = _literal_signal_network(clause.l2, n)
    return or_network(A, B)


def F_phi_network(phi: Formula) -> ReLUNet:
    """Polynomial-size ReLU network for ``F_phi(x) = sum_j C_j(x)``."""
    n = phi.n_vars
    if n < 1:
        raise ValueError("formula must have at least one variable")
    clauses = list(phi.clauses)
    if not clauses:
        net = ReLUNet(input_dim=1)
        net.add_hidden_layer(np.array([[0.0]]), np.array([0.0]))
        net.set_output(np.array([[0.0]]), np.array([0.0]))
        return net
    clause_nets = [_clause_signal_network(c, n) for c in clauses]
    return parallel_sum([(1.0, cn) for cn in clause_nets])


def expected_lipschitz(phi: Formula) -> float:
    """Predicted Lipschitz constant: ``2^(n+1) * MAX-2SAT(phi)``."""
    best, _ = brute_force_max2sat(phi)
    return (2.0 ** (phi.n_vars + 1)) * best


def network_lipschitz(net: ReLUNet, n_samples: int = 100_001) -> float:
    """Lipschitz constant of a scalar-IO ReLU net on ``[0, 1]`` from a dense grid.

    For verification purposes; for an exact value use ``net.lipschitz()``.
    """
    xs = np.linspace(0.0, 1.0, n_samples)
    ys = np.asarray(net.forward(xs)).reshape(-1)
    return float(np.max(np.abs(np.diff(ys) / np.diff(xs))))
