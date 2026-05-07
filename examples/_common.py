"""Build F_phi for a random MAX-2SAT formula at the given n; plot and verify.

Used by example_n3.py / example_n4.py / example_n5.py.
"""
from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np

from max2sat_nn import (
    F_phi_network,
    brute_force_max2sat,
    expected_lipschitz,
    random_formula,
)


def run(n: int, m: int | None = None, seed: int = 2026) -> None:
    if m is None:
        m = 2 * n
    rng = random.Random(seed)
    phi = random_formula(n_vars=n, n_clauses=m, rng=rng)
    print(f"phi = {phi}")
    print(f"  n = {n}, m = {m}, seed = {seed}")

    best, _ = brute_force_max2sat(phi)
    net = F_phi_network(phi)
    pwl = net.to_pwl(0.0, 1.0)
    L = pwl.lipschitz()
    L_pred = expected_lipschitz(phi)
    print(f"\nF_phi network: depth = {net.depth}, neurons = {net.n_neurons}")
    print(f"  PWL trace has {pwl.n_pieces} linear pieces")
    print(f"  Lip(F_phi)             = {L}")
    print(f"  2^(n+1) * MAX-2SAT     = {L_pred}    (MAX-2SAT(phi) = {best})")
    print(f"  identity holds         = {abs(L - L_pred) < 1e-6}")

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(pwl.xs, pwl.ys, lw=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$F_\varphi(x)$")
    ax.set_title(
        rf"$n={n}$, $m={m}$;   MAX-2SAT $={best}$,   "
        rf"Lip $={L:.0f}$,   $2^{{n+1}}\cdot$MAX $={int(L_pred)}$"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"F_phi_n{n}.png")
    fig.savefig(out_path, dpi=140)
    print(f"\nfigure saved to {os.path.relpath(out_path)}")
