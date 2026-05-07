"""Network-level OR (= elementwise max) for scalar-IO ReLU nets.

Uses the identity

    max(a, b) = (a + b + ReLU(a - b) + ReLU(b - a)) / 2 .

Given two equal-depth scalar->scalar ReLU nets ``A, B``, returns a
scalar->scalar ReLU net for ``max(A(x), B(x))`` of depth ``depth(A) + 1``.
"""
from __future__ import annotations

import numpy as np

from .network import ReLUNet, _block_diag, pad_depth, parallel_sum


def _relu_diff_net(A: ReLUNet, B: ReLUNet, sign: float) -> ReLUNet:
    """Depth-``(A.depth + 1)`` net evaluating ``ReLU(sign * (A(x) - B(x)))``."""
    if A.depth != B.depth:
        raise ValueError("A and B must have equal depth")
    if A.input_dim != 1 or B.input_dim != 1:
        raise ValueError("scalar input only")
    if A.output_weight is None or B.output_weight is None:
        raise ValueError("output layers must be set")

    out = ReLUNet(input_dim=1)
    for ell in range(A.depth):
        if ell == 0:
            W = np.vstack([A.hidden_weights[0], B.hidden_weights[0]])
        else:
            W = _block_diag([A.hidden_weights[ell], B.hidden_weights[ell]])
        b = np.concatenate([A.hidden_biases[ell], B.hidden_biases[ell]])
        out.add_hidden_layer(W, b)

    W_diff = sign * np.hstack([A.output_weight, -B.output_weight])
    b_diff = np.array([sign * (float(A.output_bias[0]) - float(B.output_bias[0]))])
    out.add_hidden_layer(W_diff, b_diff)
    out.set_output(np.array([[1.0]]), np.array([0.0]))
    return out


def or_network(A: ReLUNet, B: ReLUNet) -> ReLUNet:
    """ReLU network for ``max(A(x), B(x))`` (depth ``depth(A) + 1``)."""
    if A.depth != B.depth:
        raise ValueError("A and B must have equal depth")
    new_depth = A.depth + 1
    A_pad = pad_depth(A, new_depth)
    B_pad = pad_depth(B, new_depth)
    diff_pos = _relu_diff_net(A, B, +1.0)
    diff_neg = _relu_diff_net(A, B, -1.0)
    return parallel_sum(
        [(0.5, A_pad), (0.5, B_pad), (0.5, diff_pos), (0.5, diff_neg)]
    )
