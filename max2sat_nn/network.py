"""Sequential ReLU network with combinators (compose, pad_depth, parallel_sum)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .pwl import PWLFunc


@dataclass
class ReLUNet:
    """Sequential ReLU network::

        h_0 = x
        h_l = ReLU(W_l h_{l-1} + b_l),     l = 1, ..., L
        F(x) = W_{L+1} h_L + b_{L+1}
    """

    hidden_weights: List[np.ndarray] = field(default_factory=list)
    hidden_biases: List[np.ndarray] = field(default_factory=list)
    output_weight: Optional[np.ndarray] = None
    output_bias: Optional[np.ndarray] = None
    input_dim: int = 1

    # ------------------------------------------------------------ properties
    @property
    def depth(self) -> int:
        return len(self.hidden_weights)

    @property
    def n_neurons(self) -> int:
        return sum(W.shape[0] for W in self.hidden_weights)

    @property
    def widths(self) -> List[int]:
        return [W.shape[0] for W in self.hidden_weights]

    @property
    def output_dim(self) -> int:
        if self.output_weight is None:
            return self.widths[-1] if self.widths else self.input_dim
        return self.output_weight.shape[0]

    # --------------------------------------------------------------- forward
    def forward(self, x):
        x = np.asarray(x, dtype=float)
        scalar_in = self.input_dim == 1
        if scalar_in and x.ndim == 0:
            h = x.reshape(1, 1)
        elif scalar_in and x.ndim == 1:
            h = x.reshape(-1, 1)
        elif x.ndim == 1:
            h = x.reshape(1, -1)
        else:
            h = x
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            h = np.maximum(0.0, h @ W.T + b)
        if self.output_weight is not None:
            h = h @ self.output_weight.T + self.output_bias
        if scalar_in and self.output_dim == 1:
            return h.reshape(-1) if h.shape[0] > 1 else float(h[0, 0])
        return h

    # ---------------------------------------------------------------- to PWL
    def to_pwl(self, a: float = 0.0, b: float = 1.0) -> PWLFunc:
        """Exact scalar->scalar PWL representation on ``[a, b]``.

        Adaptively inserts every x where any hidden pre-activation crosses
        zero, then evaluates the network at those breakpoints.
        """
        if self.input_dim != 1 or self.output_dim != 1:
            raise ValueError("to_pwl requires scalar input and scalar output")
        xs = np.array([a, b], dtype=float)
        # Recompute breakpoints layer by layer (works because each layer's
        # input is itself PWL in x).
        for layer_idx in range(self.depth):
            pre = self._preact_at(xs, layer_idx)
            new_pts: List[float] = []
            for j in range(pre.shape[1]):
                p = pre[:, j]
                for i in range(len(xs) - 1):
                    if (p[i] > 0) != (p[i + 1] > 0) and p[i] != p[i + 1]:
                        t = -p[i] / (p[i + 1] - p[i])
                        if 0.0 < t < 1.0:
                            new_pts.append(xs[i] + t * (xs[i + 1] - xs[i]))
            if new_pts:
                xs = np.unique(np.concatenate([xs, np.asarray(new_pts)]))
                xs = _dedupe(xs)
        ys = np.asarray(self.forward(xs)).reshape(-1)
        return PWLFunc(xs, ys)

    def lipschitz(self, a: float = 0.0, b: float = 1.0) -> float:
        """Exact Lipschitz constant of this scalar->scalar network on ``[a, b]``."""
        return self.to_pwl(a, b).lipschitz()

    # ----------------------------------------------------------- assembly
    def add_hidden_layer(self, W: np.ndarray, b: np.ndarray) -> "ReLUNet":
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        if W.ndim != 2 or b.ndim != 1 or W.shape[0] != b.shape[0]:
            raise ValueError("Inconsistent layer shapes")
        prev = self.widths[-1] if self.widths else self.input_dim
        if W.shape[1] != prev:
            raise ValueError("Layer input dim mismatch")
        self.hidden_weights.append(W)
        self.hidden_biases.append(b)
        return self

    def set_output(self, W: np.ndarray, b: np.ndarray) -> "ReLUNet":
        W = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float)
        last_dim = self.widths[-1] if self.widths else self.input_dim
        if W.shape[1] != last_dim or W.shape[0] != b.shape[0]:
            raise ValueError("Inconsistent output shape")
        self.output_weight = W
        self.output_bias = b
        return self

    # ---------------------------------------------------------------- utils
    def _preact_at(self, xs: np.ndarray, layer_idx: int) -> np.ndarray:
        """Pre-activation values of hidden layer ``layer_idx`` at scalar ``xs``."""
        h = xs.reshape(-1, 1)
        for ell in range(layer_idx):
            W, b = self.hidden_weights[ell], self.hidden_biases[ell]
            h = np.maximum(0.0, h @ W.T + b)
        W, b = self.hidden_weights[layer_idx], self.hidden_biases[layer_idx]
        return h @ W.T + b


# -------------------------------------------------------- helper utilities
def _dedupe(xs: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    xs = np.sort(xs)
    keep = [xs[0]]
    for x in xs[1:]:
        if x - keep[-1] > tol:
            keep.append(x)
    return np.asarray(keep, dtype=float)


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    rows = sum(B.shape[0] for B in blocks)
    cols = sum(B.shape[1] for B in blocks)
    out = np.zeros((rows, cols))
    r = c = 0
    for B in blocks:
        out[r : r + B.shape[0], c : c + B.shape[1]] = B
        r += B.shape[0]
        c += B.shape[1]
    return out


# ----------------------------------------------------------- combinators
def affine_relu_net(a: float, b: float) -> ReLUNet:
    """Depth-1 width-1 net implementing ``y -> ReLU(a*y + b)``."""
    net = ReLUNet(input_dim=1)
    net.add_hidden_layer(np.array([[float(a)]]), np.array([float(b)]))
    net.set_output(np.array([[1.0]]), np.array([0.0]))
    return net


def pad_depth(net: ReLUNet, target_depth: int) -> ReLUNet:
    """Append identity hidden layers (post-ReLU activations are >= 0)."""
    if target_depth < net.depth:
        raise ValueError(f"target_depth={target_depth} < depth={net.depth}")
    if net.output_weight is None:
        raise ValueError("net must have an output layer set")
    out = ReLUNet(input_dim=net.input_dim)
    for W, b in zip(net.hidden_weights, net.hidden_biases):
        out.add_hidden_layer(W.copy(), b.copy())
    last = net.widths[-1]
    for _ in range(target_depth - net.depth):
        out.add_hidden_layer(np.eye(last), np.zeros(last))
    out.set_output(net.output_weight.copy(), net.output_bias.copy())
    return out


def compose(outer: ReLUNet, inner: ReLUNet) -> ReLUNet:
    """Return ``outer ∘ inner`` (apply ``inner`` first, then ``outer``)."""
    if inner.input_dim != 1 or outer.input_dim != 1:
        raise ValueError("compose requires scalar-input nets")
    if inner.output_weight is None or outer.output_weight is None:
        raise ValueError("both nets must have an output layer set")
    if inner.output_dim != 1 or not outer.hidden_weights:
        raise ValueError("inner must be scalar-output, outer must have hidden layers")
    # Merge inner's output linear layer into outer's first hidden layer.
    W_o1, b_o1 = outer.hidden_weights[0], outer.hidden_biases[0]
    merged_W = W_o1 @ inner.output_weight
    merged_b = (W_o1 @ inner.output_bias).reshape(-1) + b_o1

    out = ReLUNet(input_dim=inner.input_dim)
    for W, b in zip(inner.hidden_weights, inner.hidden_biases):
        out.add_hidden_layer(W.copy(), b.copy())
    out.add_hidden_layer(merged_W, merged_b)
    for W, b in zip(outer.hidden_weights[1:], outer.hidden_biases[1:]):
        out.add_hidden_layer(W.copy(), b.copy())
    out.set_output(outer.output_weight.copy(), outer.output_bias.copy())
    return out


def parallel_sum(weighted_nets: Sequence[Tuple[float, ReLUNet]]) -> ReLUNet:
    """Weighted sum of equal-depth scalar-IO nets: ``sum_i w_i * net_i(x)``."""
    if not weighted_nets:
        raise ValueError("need at least one net")
    weights = [float(w) for w, _ in weighted_nets]
    nets = [n for _, n in weighted_nets]
    L = nets[0].depth
    for n in nets:
        if n.input_dim != 1 or n.output_dim != 1:
            raise ValueError("parallel_sum requires scalar I/O nets")
        if n.output_weight is None:
            raise ValueError("each net must have an output layer set")
        if n.depth != L:
            raise ValueError(f"all nets must have equal depth (got {n.depth} vs {L})")

    out = ReLUNet(input_dim=1)
    for ell in range(L):
        if ell == 0:
            W = np.vstack([n.hidden_weights[0] for n in nets])
        else:
            W = _block_diag([n.hidden_weights[ell] for n in nets])
        b = np.concatenate([n.hidden_biases[ell] for n in nets])
        out.add_hidden_layer(W, b)
    out_W = np.hstack([w * n.output_weight for w, n in zip(weights, nets)])
    out_b = np.array([sum(w * float(n.output_bias[0]) for w, n in zip(weights, nets))])
    out.set_output(out_W, out_b)
    return out
