"""MAX-2SAT to 1D ReLU network reduction (network-level construction)."""

from .max2sat import (
    Clause,
    Formula,
    Literal,
    assignment_from_index,
    brute_force_max2sat,
    random_formula,
)
from .network import ReLUNet
from .reduction import F_phi_network, expected_lipschitz, network_lipschitz

__all__ = [
    "Clause",
    "Formula",
    "Literal",
    "ReLUNet",
    "F_phi_network",
    "assignment_from_index",
    "brute_force_max2sat",
    "expected_lipschitz",
    "network_lipschitz",
    "random_formula",
]
