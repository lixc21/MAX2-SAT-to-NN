import numpy as np
import pytest

from max2sat_nn.tent import sawtooth_network, tent_network


def test_tent_network_values():
    xs = np.linspace(0.0, 1.0, 101)
    expected = 2.0 * np.minimum(xs, 1.0 - xs)
    np.testing.assert_allclose(tent_network().forward(xs), expected, atol=1e-12)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_sawtooth_network_size_and_breakpoints(n):
    net = sawtooth_network(n)
    assert net.depth == n
    assert all(w == 2 for w in net.widths)
    # T_n alternates 0, 1 at every multiple of 2^-n.
    xs = np.arange(2 ** n + 1) / 2 ** n
    expected = np.array([(i % 2) for i in range(2 ** n + 1)], dtype=float)
    np.testing.assert_allclose(net.forward(xs), expected, atol=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_sawtooth_network_lipschitz(n):
    assert pytest.approx(sawtooth_network(n).lipschitz()) == 2.0 ** n


def test_sawtooth_n0_is_identity():
    net = sawtooth_network(0)
    xs = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(net.forward(xs), xs, atol=1e-12)
