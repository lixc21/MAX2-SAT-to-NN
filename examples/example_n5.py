"""F_phi for a random MAX-2SAT formula with n=5 variables."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import run

if __name__ == "__main__":
    run(n=5, m=10, seed=2026)
