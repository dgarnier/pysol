"""Test Lyle's formulae for inductance."""

import unittest

import coverage_env  # noqa: F401
import numpy as np

from inductance.self import L_lorentz, L_lyle4, L_maxwell
from inductance.utils import _lyle_terms


class TestSelfInductances(unittest.TestCase):
    """Test the self-inductances routines."""

    def test_L_maxwell(self):
        """Test Maxwell's formula."""
        self.assertAlmostEqual(L_maxwell(1, 1e-4, 1, 1), 20.7463e-7, places=6)

    def test_L_lorentz(self):
        """Test thin-wall Lorentz formula. See Lyle pg. 429."""
        mu0 = 4e-7 * np.pi
        self.assertAlmostEqual(L_lorentz(1, 0.0, 2, 1) / mu0, 1.08137, places=5)
        self.assertAlmostEqual(L_lorentz(1, 0.0, 1, 1), 20.7463e-7, places=6)
        self.assertAlmostEqual(L_lorentz(1, 0.0, 0.5, 1), 28.85335e-7, places=7)

    def test_L_lyle4(self):
        """Test Lyle's formula against Lorentz thin-wall. See Lyle pg. 429."""
        mu0 = 4e-7 * np.pi
        self.assertAlmostEqual(L_lyle4(1, 1e-6, 2, 1) / mu0, 1.07970, places=3)
        self.assertAlmostEqual(L_lyle4(1, 1e-6, 1, 1), 20.7463e-7, places=6)
        self.assertAlmostEqual(L_lyle4(1, 1e-6, 0.5, 1), 28.85335e-7, places=7)

    def test_lyle_terms(self):
        """Test Lyle's formula against his table. See Lyle pg. 429."""
        table1 = np.array(
            [  # Lyle's Table 1
                [0.00, 1.5, 0.223130],
                [0.025, 1.474734, 0.223328],
                [0.05, 1.451005, 0.223455],
                [0.10, 1.407566, 0.223599],
                [0.15, 1.368975, 0.223664],
                [0.20, 1.334799, 0.223686],
                [0.25, 1.304680, 0.223686],
                [0.30, 1.278284, 0.223675],
                [0.35, 1.255312, 0.223658],
                [0.40, 1.235461, 0.223639],
                [0.45, 1.218448, 0.223619],
                [0.50, 1.203998, 0.223601],
                [0.55, 1.191853, 0.223584],
                [0.60, 1.181768, 0.223570],
                [0.65, 1.173516, 0.223558],
                [0.70, 1.166888, 0.223548],
                [0.75, 1.161691, 0.223540],
                [0.80, 1.157752, 0.223534],
                [0.85, 1.154914, 0.223530],
                [0.90, 1.153034, 0.223527],
                [0.95, 1.151987, 0.223525],
                [1.00, 1.151660, 0.223525],
            ]
        )
        for row in table1:
            b, philyle, roverbc = row[0], row[1], row[2]
            c = 1.0
            _, _, _, _, _, phi, GMD = _lyle_terms(b, c)

            self.assertAlmostEqual(phi, philyle, places=5)
            self.assertAlmostEqual(GMD / (b + c), roverbc, places=5)


if __name__ == "__main__":
    unittest.main()
