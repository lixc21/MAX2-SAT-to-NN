"""End-to-end MAX-2SAT -> ReLU network reduction.

For every random formula we check the central identity

    Lip(F_phi_network(phi)) == 2^(n+1) * MAX-2SAT(phi)

and that the network's size grows polynomially in (n, m).
"""
import random

import numpy as np
import pytest

from max2sat_nn import (
    Clause,
    F_phi_network,
    Formula,
    Literal,
    brute_force_max2sat,
    expected_lipschitz,
    network_lipschitz,
    random_formula,
)


# ----------------------------------------------------- handcrafted formula
def test_handcrafted_formula_n3():
    """phi = (x1 v x2) and (~x1 v x3) and (x2 v ~x3); MAX-2SAT = 3."""
    phi = Formula(
        n_vars=3,
        clauses=(
            Clause(Literal(1, False), Literal(2, False)),
            Clause(Literal(1, True), Literal(3, False)),
            Clause(Literal(2, False), Literal(3, True)),
        ),
    )
    best, _ = brute_force_max2sat(phi)
    assert best == 3
    L = F_phi_network(phi).lipschitz()
    assert L == pytest.approx(2.0 ** 4 * best, rel=1e-9)


# --------------------------------------- central identity, fixed seeds
@pytest.mark.parametrize(
    "n,m,seed",
    [
        (2, 2, 0), (2, 4, 1), (2, 6, 2),
        (3, 3, 10), (3, 5, 11), (3, 7, 12), (3, 9, 13),
        (4, 4, 20), (4, 6, 21), (4, 8, 22), (4, 10, 23),
        (5, 5, 30), (5, 8, 31), (5, 10, 32),
    ],
)
def test_lipschitz_identity_random(n, m, seed):
    rng = random.Random(seed)
    phi = random_formula(n, m, rng)
    net = F_phi_network(phi)
    L_exact = net.lipschitz()
    L_grid = network_lipschitz(net, n_samples=50_001)
    L_pred = expected_lipschitz(phi)
    assert L_exact == pytest.approx(L_pred, rel=1e-9), (n, m, seed, phi)
    assert L_grid == pytest.approx(L_pred, rel=1e-4), (n, m, seed, phi)


# ----------------------------------------------- midpoint semantics check
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_midpoints_realise_satisfaction_counts(seed):
    """The 2^n midpoint values of F_phi are a permutation of the per-assignment
    counts of satisfied clauses."""
    from max2sat_nn.bits import midpoint_grid
    from max2sat_nn.max2sat import assignment_from_index

    rng = random.Random(seed)
    n = rng.choice([2, 3, 4])
    m = rng.randint(2, 6)
    phi = random_formula(n, m, rng)
    net = F_phi_network(phi)
    mids = midpoint_grid(n)
    F_vals = np.asarray(net.forward(mids))
    counts = np.array(
        [phi.count_satisfied(assignment_from_index(k, n)) for k in range(2 ** n)],
        dtype=float,
    )
    np.testing.assert_allclose(np.sort(F_vals), np.sort(counts), atol=1e-9)


# ------------------------------------------------------ polynomial size
@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_network_size_is_polynomial(n):
    """Depth = n + 2; total neurons grow as O(m * n)."""
    rng = random.Random(11)
    m = 4
    phi = random_formula(n, m, rng)
    net = F_phi_network(phi)
    assert net.depth == n + 2
    assert net.n_neurons <= 60 * m * (n + 2)


def test_neurons_grow_linearly_in_m_and_n():
    rng = random.Random(2)
    n = 4
    sizes_m = [F_phi_network(random_formula(n, m, rng)).n_neurons for m in [1, 2, 4, 8]]
    diffs_m = np.diff(sizes_m) / np.diff([1, 2, 4, 8])
    np.testing.assert_allclose(diffs_m, diffs_m[0], rtol=0)

    m = 3
    sizes_n = [F_phi_network(random_formula(nn, m, rng)).n_neurons for nn in [2, 3, 4, 5]]
    diffs_n = np.diff(sizes_n)
    assert np.allclose(diffs_n, diffs_n[0])
