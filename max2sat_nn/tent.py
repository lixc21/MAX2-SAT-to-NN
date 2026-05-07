"""Tent and sawtooth ReLU networks.

The tent function is

    g(x) = 2 * min(x, 1 - x) = ReLU(2x) - 2 * ReLU(2x - 1),

implemented by a depth-1 width-2 ReLU network. The sawtooth ``T_n`` is the
``n``-fold composition ``g ∘ g ∘ ... ∘ g``; it has ``2^n`` linear pieces of
slope ``±2^n`` on ``[0, 1]`` and is implemented by a depth-``n`` width-2
ReLU network by stacking tent layers.
"""
from __future__ import annotations

import numpy as np

from .network import ReLUNet


def tent_network() -> ReLUNet:
    """Depth-1 width-2 ReLU network for the tent ``g(x)``."""
    net = ReLUNet(input_dim=1)
    net.add_hidden_layer(np.array([[2.0], [2.0]]), np.array([0.0, -1.0]))
    net.set_output(np.array([[1.0, -2.0]]), np.array([0.0]))
    return net


def sawtooth_network(n: int) -> ReLUNet:
    """Depth-``n`` width-2 ReLU network for ``T_n``.

    For ``n = 0`` returns the identity (depth 1, width 1).
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    net = ReLUNet(input_dim=1)
    if n == 0:
        net.add_hidden_layer(np.array([[1.0]]), np.array([0.0]))
        net.set_output(np.array([[1.0]]), np.array([0.0]))
        return net

    # Layer 1: x -> [ReLU(2x), ReLU(2x - 1)].
    net.add_hidden_layer(np.array([[2.0], [2.0]]), np.array([0.0, -1.0]))
    # Layers 2..n: input h = [r1, r2] = [ReLU(2y), ReLU(2y-1)] of previous tent;
    # next tent input is y' = r1 - 2 r2, so produce
    #   [ReLU(2 y'), ReLU(2 y' - 1)] = [ReLU(2 r1 - 4 r2), ReLU(2 r1 - 4 r2 - 1)].
    for _ in range(n - 1):
        net.add_hidden_layer(
            np.array([[2.0, -4.0], [2.0, -4.0]]),
            np.array([0.0, -1.0]),
        )
    net.set_output(np.array([[1.0, -2.0]]), np.array([0.0]))
    return net
