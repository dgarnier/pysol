"""Test the elliptics module."""

import unittest

import coverage_env  # noqa: F401

from inductance.elliptics import celbd, ellipe, ellipk


class TestElliptics(unittest.TestCase):
    """Test the elliptics module."""

    def setUp(self) -> None:
        """Set up the test data."""
        self.scipy_data = [
            (0.00, 1.5707963267948966, 1.5707963267948966),
            (0.009, 1.574348623485514, 1.567256048466157),
            (0.05, 1.591003453790792, 1.5509733517804725),
            (0.10, 1.6124413487202192, 1.5307576368977633),
            (0.15, 1.63525673226458, 1.5101218320928198),
            (0.20, 1.659623598610528, 1.489035058095853),
            (0.25, 1.685750354812596, 1.4674622093394272),
            (0.30, 1.713889448178791, 1.4453630644126654),
            (0.35, 1.7443505972256133, 1.4226911334908792),
            (0.40, 1.7775193714912534, 1.3993921388974322),
            (0.45, 1.8138839368169826, 1.3754019718711163),
            (0.50, 1.8540746773013719, 1.3506438810476755),
            (0.55, 1.8989249102715537, 1.32502449795823),
            (0.60, 1.9495677498060258, 1.298428035046913),
            (0.65, 2.0075983984243764, 1.2707074796501499),
            (0.70, 2.075363135292469, 1.2416705679458226),
            (0.75, 2.156515647499643, 1.2110560275684594),
            (0.80, 2.257205326820854, 1.1784899243278386),
            (0.85, 2.38901648632558, 1.1433957918831659),
            (0.90, 2.5780921133481733, 1.1047747327040733),
            (0.95, 2.9083372484445524, 1.0604737277662781),
            (0.99, 3.6956373629898747, 1.015993545025223987),
        ]
        return super().setUp()

    def test_ellipke(self):
        """Test the ellipke function."""
        for m, k, e in self.scipy_data:
            # this fails at the 15th place
            self.assertAlmostEqual(ellipk(m), k, places=14)
            self.assertAlmostEqual(ellipe(m), e, places=14)

    def test_celbd(self):
        """Test the celbd function. Which is to be compared with ellipkm1 in scipy.special."""
        mc = 1e-16
        b, d = celbd(mc)
        k = b + d
        self.assertAlmostEqual(19.806975105072258, k, places=14)


if __name__ == "__main__":
    unittest.main()
