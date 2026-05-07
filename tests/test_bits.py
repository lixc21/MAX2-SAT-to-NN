import numpy as np
import pytest

from max2sat_nn.bits import bit_signal_network, midpoint_grid


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_bit_signal_network_size(n):
    """Each b_k has depth n+1 and width 2."""
    for k in range(1, n + 1):
        net = bit_signal_network(k, n)
        assert net.depth == n + 1
        assert max(net.widths) == 2


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bit_signal_midpoint_values_are_binary(n):
    """At interval midpoints b_k is exactly 0 or 1, with half being 1."""
    mids = midpoint_grid(n)
    for k in range(1, n + 1):
        for negated in (False, True):
            v = np.asarray(bit_signal_network(k, n, negated=negated).forward(mids))
            np.testing.assert_allclose(v, np.round(v), atol=1e-10)
            assert int(v.sum()) == 2 ** (n - 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bit_signal_zero_at_interval_boundaries(n):
    """All spikes vanish at every interval boundary."""
    bdry = np.arange(2 ** n + 1) / 2 ** n
    for k in range(1, n + 1):
        for negated in (False, True):
            v = np.asarray(bit_signal_network(k, n, negated=negated).forward(bdry))
            np.testing.assert_allclose(v, 0.0, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bit_signal_assignment_bijection(n):
    """interval -> (b_1, ..., b_n)(midpoint) is a bijection."""
    mids = midpoint_grid(n)
    bits = np.stack(
        [np.asarray(bit_signal_network(k, n).forward(mids)) for k in range(1, n + 1)],
        axis=1,
    ).round().astype(int)
    seen = {tuple(row) for row in bits}
    assert len(seen) == 2 ** n


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bit_signal_lipschitz(n):
    """Per-bit Lipschitz constant equals 2^(n+1)."""
    for k in range(1, n + 1):
        assert pytest.approx(bit_signal_network(k, n).lipschitz(), rel=1e-9) == 2.0 ** (n + 1)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bit_signal_complement(n):
    """b_k(mid) + bit_neg_k(mid) = 1 at every midpoint."""
    mids = midpoint_grid(n)
    for k in range(1, n + 1):
        a = np.asarray(bit_signal_network(k, n, negated=False).forward(mids))
        b = np.asarray(bit_signal_network(k, n, negated=True).forward(mids))
        np.testing.assert_allclose(a + b, 1.0, atol=1e-10)
