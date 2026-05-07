"""Minimal usage example for max2sat_nn."""
import numpy as np

from max2sat_nn import (
    Clause,
    Formula,
    Literal,
    F_phi_network,
    brute_force_max2sat,
    expected_lipschitz,
    network_lipschitz,
)


def main() -> None:
    # phi = (x1 v x2) and (~x1 v x3) and (x2 v ~x3)
    phi = Formula(
        n_vars=3,
        clauses=(
            Clause(Literal(1, False), Literal(2, False)),
            Clause(Literal(1, True), Literal(3, False)),
            Clause(Literal(2, False), Literal(3, True)),
        ),
    )
    print(f"phi = {phi}")
    print(f"  n = {phi.n_vars}, m = {phi.n_clauses}")

    best, assn = brute_force_max2sat(phi)
    print(f"\nbrute-force MAX-2SAT(phi) = {best}  (e.g. assignment {tuple(assn)})")

    net = F_phi_network(phi)
    print(f"\nF_phi network: depth = {net.depth}, neurons = {net.n_neurons}")

    L = net.lipschitz()                    # exact, via PWL trace
    L_grid = network_lipschitz(net)        # numerical, dense grid
    L_pred = expected_lipschitz(phi)

    print(f"  Lip(F_phi) (exact)           = {L}")
    print(f"  Lip(F_phi) (grid estimate)   = {L_grid}")
    print(f"  2^(n+1) * MAX-2SAT(phi)      = {L_pred}")
    print(f"  identity holds               = {abs(L - L_pred) < 1e-6}")


if __name__ == "__main__":
    main()
