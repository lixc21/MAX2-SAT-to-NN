"""F_phi for a random MAX-2SAT formula with n=4 variables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import run

if __name__ == "__main__":
    run(n=4, m=8, seed=2026)
