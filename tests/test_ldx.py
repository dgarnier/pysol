"""Test filamentation and inducatance of LDX coils."""

import unittest

import coverage_env  # noqa: F401

from inductance.coils import Coil, CompositeCoil


class TestLDXInductance(unittest.TestCase):
    """Test the inductance of the LDX coils.

    We know the coil weighed about 590kg, and had about 0.4H inductance.
    """

    def setUp(self) -> None:
        """Set up the LDX coils."""
        (
            self.HTS_LC,
            self.LC,
            self.CC,
            self.F1,
            self.F2,
            self.F3,
            self.FC,
        ) = define_ldx_coils()
        return super().setUp()

    def test_self_inductances(self):
        """Test the self-inductances routines."""
        self.assertAlmostEqual(self.LC.L_Maxwell(), 0.005397519485316731)
        self.assertAlmostEqual(self.LC.L_Lyle4(), 0.005623887363536865)
        self.assertAlmostEqual(self.LC.L_Lyle6(), 0.005626208118367906)
        self.assertAlmostEqual(self.LC.L_Lyle6A(), 0.005626208118367906)
        self.assertAlmostEqual(self.LC.L_filament(), 0.005625117051066614)

        self.assertAlmostEqual(self.CC.L_Maxwell(), 85.91637858501646)
        self.assertAlmostEqual(self.CC.L_Lyle4(), 90.90254315310752)
        self.assertAlmostEqual(self.CC.L_Lyle6(), 90.90927053789774)
        self.assertAlmostEqual(self.CC.L_Lyle6A(), 90.90927053789773)
        self.assertAlmostEqual(self.CC.L_filament(), 90.89118772211005)

    def test_mutual_inductances(self):
        """Test the mutual inductances routines."""
        MFL = self.FC.M_filament(self.LC)
        MFC = self.FC.M_filament(self.CC)
        self.assertAlmostEqual(MFL, 0.611729e-3)
        self.assertAlmostEqual(MFC, 1.686291576361124)

    def test_fcoil_selfinductance(self):
        """Test the self-inductance of the F coils."""
        LF1 = self.F1.L_Lyle6()
        LF2 = self.F2.L_Lyle6()
        LF3 = self.F3.L_Lyle6()
        MF12 = self.F1.M_filament(self.F2)
        MF13 = self.F1.M_filament(self.F3)
        MF23 = self.F2.M_filament(self.F3)
        LFC = LF1 + LF2 + LF3 + 2 * MF12 + 2 * MF13 + 2 * MF23
        self.assertAlmostEqual(LFC, 0.38692937382836284)

    def test_levitation_force(self):
        """Test the levitation force."""
        FZ = self.FC.Fz_filament(self.LC) / 9.81
        self.assertAlmostEqual(FZ, 589.263842397435)


def define_ldx_coils():
    """Define the LDX coils."""
    HTS_LC = {}
    HTS_LC["r1"] = 0.41 / 2
    HTS_LC["r2"] = 1.32 / 2
    HTS_LC["z1"] = 1.610 - 0.018 / 2
    HTS_LC["z2"] = 1.610 + 0.018 / 2
    HTS_LC["nt"] = 2796
    HTS_LC["at"] = 2796 * 105
    HTS_LC = Coil.from_dict(HTS_LC)
    HTS_LC.filamentize(30, 2)

    LC = {}
    LC["r1"] = 0.246
    LC["r2"] = 0.70
    LC["z1"] = 1.525
    LC["z2"] = LC["z1"] + 0.1
    LC["nt"] = 80
    LC["at"] = 3500 * 80
    LC = Coil.from_dict(LC)
    LC.filamentize(20, 4)

    CC = {}
    CC["r1"] = 0.645
    CC["r2"] = 0.787
    CC["z1"] = -0.002 - 0.750 / 2
    CC["z2"] = -0.002 + 0.750 / 2
    CC["nt"] = 8388
    CC["at"] = 8388 * 420
    CC = Coil.from_dict(CC)
    CC.filamentize(3, 7)

    F1 = {}
    F1["r1"] = 0.2717 - 0.01152 / 2
    F1["r2"] = 0.2717 + 0.01152 / 2
    F1["z1"] = -0.0694 / 2
    F1["z2"] = +0.0694 / 2
    F1["nt"] = 26.6
    F1["at"] = 26.6 * 1629
    F1 = Coil.from_dict(F1)
    F1.filamentize(2, 4)

    F2 = {}
    F2["r1"] = 0.28504 - 0.01508 / 2
    F2["r2"] = 0.28504 + 0.01508 / 2
    F2["z1"] = -0.125 / 2
    F2["z2"] = +0.125 / 2
    F2["nt"] = 81.7
    F2["at"] = 81.7 * 1629
    F2 = Coil.from_dict(F2)
    F2.filamentize(3, 7)

    F3 = {}
    F3["r1"] = 0.33734 - 0.08936 / 2
    F3["r2"] = 0.33734 + 0.08936 / 2
    F3["z1"] = -0.1615 / 2
    F3["z2"] = +0.1615 / 2
    F3["nt"] = 607.7
    F3["at"] = 607.7 * 1629
    F3 = Coil.from_dict(F3)
    F3.filamentize(10, 15)

    FC = CompositeCoil([F1, F2, F3])

    return HTS_LC, LC, CC, F1, F2, F3, FC


if __name__ == "__main__":
    unittest.main()
