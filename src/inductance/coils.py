"""Coil inductance calculations.

Defines a coil class to keep track of coil parameters.

Benchmarking against LDX values, which come from
old Mathematica routines and other tests.

Filaments are defined as an numpy 3 element vector
 - r, z, and n.
"""
from __future__ import annotations

import numpy as np

from .filaments import (
    filament_coil,
    mutual_inductance_of_filaments,
    radial_force_of_filaments,
    self_inductance_by_filaments,
    vertical_force_of_filaments,
)
from .self import (
    L_long_solenoid_butterworth,
    L_lorentz,
    L_lyle4,
    L_lyle6,
    L_lyle6_appendix,
    L_maxwell,
    dLdR_lyle6,
)


class Coil:
    """Rectangular coil object to keep track of coil parameters."""

    def __init__(self, r, z, dr, dz, nt=1, at=1, nr=0, nz=0):
        """Create a rectangular coil object.

        Args:
            r (float): radial center of coil
            z (float): vertical center of coil
            dr (float): radial width of coil
            dz (float): axial height of coil
            nt (int): number of turns in coil
            nr (int, optional): Number of radial sections to filament coil. Defaults to 0.
            nz (int, optional): Number of axial sections to filament coil. Defaults to 0.
            at (float, optional): Amperage of coil. Defaults to 0.
        """
        self.r = r
        self.z = z
        self.dr = dr
        self.dz = dz
        self.nt = nt
        self.at = at
        self.fils = None

        if (nr > 0) and (nz > 0):
            self.nr = nr
            self.nz = nz
            self.filamentize(nr, nz)

    @classmethod
    def from_dict(cls, d):
        """Create a coil from a dictionary."""
        if "r1" in d:
            return cls.from_bounds(**d)
        else:
            return cls(**d)

    @classmethod
    def from_bounds(cls, r1, r2, z1, z2, nt=1, at=1, nr=0, nz=0):
        """Create a coil from bounds instead of center and width."""
        return cls(
            (r1 + r2) / 2, (z1 + z2) / 2, r2 - r1, z2 - z1, nt=nt, at=at, nr=nr, nz=nz
        )

    @property
    def r1(self):  # noqa: D102
        return self.r - self.dr / 2

    @property
    def r2(self):  # noqa: D102
        return self.r + self.dr / 2

    @property
    def z1(self):  # noqa: D102
        return self.z - self.dz / 2

    @property
    def z2(self):  # noqa: D102
        return self.z + self.dz / 2

    def filamentize(self, nr, nz):
        """Create an array of filaments to represent the coil."""
        self.nr = nr
        self.nz = nz
        self.fils = filament_coil(self.r, self.z, self.dr, self.dz, self.nt, nr, nz)

    def L_Maxwell(self):
        """Inductance by Maxwell's formula."""
        return L_maxwell(self.r, self.dr, self.dz, self.nt)

    def L_Lyle4(self):
        """Inductance by Lyle's formula, 4th order."""
        return L_lyle4(self.r, self.dr, self.dz, self.nt)

    def L_Lyle6(self):
        """Inductance by Lyle's formula, 6th order."""
        return L_lyle6(self.r, self.dr, self.dz, self.nt)

    def L_Lyle6A(self):
        """Inductance by Lyle's formula, 6th order, appendix."""
        return L_lyle6_appendix(self.r, self.dr, self.dz, self.nt)

    def L_filament(self):
        """Inductance by filamentation."""
        return self_inductance_by_filaments(
            self.fils, conductor="rect", dr=self.dr / self.nr, dz=self.dz / self.nz
        )

    def L_long_solenoid_butterworth(self):
        """Inductance by Butterworth's formula."""
        return L_long_solenoid_butterworth(self.r, self.dr, self.dz, self.nt)

    def L_lorentz(self):
        """Inductance by Lorentz's formula."""
        return L_lorentz(self.r, self.dr, self.dz, self.nt)

    def dLdR_Lyle6(self):
        """Derivative of inductance by Lyle's formula, 6th order."""
        return dLdR_lyle6(self.r, self.dr, self.dz, self.nt)

    def M_filament(self, C2: Coil) -> float:
        """Mutual inductance of two coils by filamentation."""
        return mutual_inductance_of_filaments(self.fils, C2.fils)

    def Fz_filament(self, C2: Coil) -> float:
        """Vertical force of two coils by filamentation."""
        F_a2 = vertical_force_of_filaments(self.fils, C2.fils)
        return self.at / self.nt * C2.at / C2.nt * F_a2

    def Fr_self(self) -> float:
        """Radial force of coil on itself."""
        dLdR = dLdR_lyle6(self.r, self.dr, self.dz, self.nt)
        return (self.at / self.nt) ** 2 / 2 * dLdR

    def Fr_filament(self, C2: Coil) -> float:
        """Radial force of two coils by filamentation."""
        F_r2 = radial_force_of_filaments(self.fils, C2.fils)
        return self.at / self.nt * C2.at / C2.nt * F_r2


class CompositeCoil(Coil):
    """A coil made of multiple rectangular coils."""

    def __init__(self, coils: list[Coil]):
        """Create a composite coil from a list of _filamented_ coils."""
        self.coils = coils
        self.nt = sum(coil.nt for coil in coils)
        self.at = sum(coil.at for coil in coils)
        self.r = sum(coil.r * coil.nt for coil in coils) / self.nt
        self.z = sum(coil.z * coil.nt for coil in coils) / self.nt
        self.fils = np.concatenate([coil.fils for coil in coils])
