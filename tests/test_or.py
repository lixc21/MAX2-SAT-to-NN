import random

import numpy as np
import pytest

from max2sat_nn.bits import bit_signal_network
from max2sat_nn.boolean import or_network


@pytest.mark.parametrize("n", [2, 3, 4])
def test_or_network_matches_pointwise_max(n):
    rng = random.Random(123 + n)
    xs = np.linspace(0.0, 1.0, 4001)
    for _ in range(8):
        k1 = rng.randint(1, n)
        k2 = rng.randint(1, n)
        A = bit_signal_network(k1, n, negated=rng.random() < 0.5)
        B = bit_signal_network(k2, n, negated=rng.random() < 0.5)
        out = or_network(A, B)
        a = np.asarray(A.forward(xs))
        b = np.asarray(B.forward(xs))
        np.testing.assert_allclose(np.asarray(out.forward(xs)), np.maximum(a, b), atol=1e-9)
