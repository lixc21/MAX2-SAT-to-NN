"""Bit-extraction signals as polynomial-size ReLU networks.

For ``1 <= k <= n``, ``bit_signal_network(k, n)`` is a depth-``(n+1)``
width-``2`` ReLU network whose forward pass is a PWL function on ``[0, 1]``
with triangular spikes (height 1, width ``2^-n``) marking exactly half of
the ``2^n`` length-``2^-n`` intervals partitioning ``[0, 1]``.

Closed form (matches the paper's bit-extraction figure):

    b_k(x)         = T_k( ReLU(  2 T_{n-k}(x) - 1 ) )           (positive)
    bit_neg_k(x)   = T_k( ReLU(  1 - 2 T_{n-k}(x) ) )           (negated)

The Lipschitz constant of every bit signal is ``2^(n+1)``.

Note: stacking tents folds the interval index in a Gray-code-like order, so
the spike pattern is a permutation of the plain little-endian one. This is
harmless for the MAX-2SAT reduction because brute_force_max2sat enumerates
*every* assignment.
"""
from __future__ import annotations

import numpy as np

from .network import ReLUNet, affine_relu_net, compose
from .tent import sawtooth_network


def bit_signal_network(k: int, n: int, negated: bool = False) -> ReLUNet:
    """Polynomial-size ReLU network for the bit-``k`` spike signal."""
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    ramp = affine_relu_net(-2.0, 1.0) if negated else affine_relu_net(2.0, -1.0)
    T_outer = sawtooth_network(k)
    if n - k == 0:
        return compose(T_outer, ramp)
    T_inner = sawtooth_network(n - k)
    return compose(T_outer, compose(ramp, T_inner))


def midpoint_grid(n: int) -> np.ndarray:
    """Array of the ``2^n`` interval midpoints on ``[0, 1]``."""
    h = 2.0 ** -n
    return (np.arange(2 ** n) + 0.5) * h
