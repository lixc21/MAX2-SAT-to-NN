import numpy as np
import pytest

from max2sat_nn.network import ReLUNet


def _tent_net() -> ReLUNet:
    net = ReLUNet(input_dim=1)
    net.add_hidden_layer(np.array([[2.0], [2.0]]), np.array([0.0, -1.0]))
    net.set_output(np.array([[1.0, -2.0]]), np.array([0.0]))
    return net


def test_tent_forward_values():
    xs = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(_tent_net().forward(xs), [0.0, 0.5, 1.0, 0.5, 0.0], atol=1e-12)


def test_tent_to_pwl_and_lipschitz():
    net = _tent_net()
    f = net.to_pwl(0.0, 1.0)
    assert f.n_pieces == 2
    assert pytest.approx(f.lipschitz()) == 2.0
    xs = np.linspace(0.0, 1.0, 101)
    np.testing.assert_allclose(f(xs), net.forward(xs), atol=1e-12)


def test_widths_and_size():
    net = _tent_net()
    assert net.depth == 1
    assert net.widths == [2]
    assert net.n_neurons == 2
