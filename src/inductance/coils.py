"""Coil inductance calculations.

Defines a coil class to keep track of coil parameters.

Benchmarking against LDX values, which come from
old Mathematica routines and other tests.

Filaments are defined as an numpy 3 element vector
 - r, z, and n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .filaments import (
    filament_coil,
    mutual_inductance_of_filaments,
    radial_force_of_filaments,
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
    self_inductance_by_filaments,
)


class Shape(Enum):
    """Enum for conductor shapes."""

    round = "round"
    hollow_round = "hollow_round"
    rect = "rectangle"


@dataclass
class Conductor:
    """Conductor object to keep track of conductor parameters."""

    shape: Shape | str = Shape.round
    r: float = 0.0
    dr: float = 0.0
    dz: float = 0.0
    r_i: float = 0.0
    rho: float = 0.0

    def __post_init__(self):
        """Post init."""
        if isinstance(self.shape, str):
            self.shape = Shape(self.shape)


class Coil:
    """Rectangular coil object to keep track of coil parameters."""

    def __init__(self, r, z, dr, dz, nt=1, at=None, nr=0, nz=0, theta=0, **kwargs):
        """Create a rectangular coil object.

        Args:
            r (float): radial center of coil
            z (float): vertical center of coil
            dr (float): radial width of coil
            dz (float): axial height of coil
            nt (int, optional): number of turns in coil
            nr (int, optional): Number of radial sections to filament coil. Defaults to 0.
            nz (int, optional): Number of axial sections to filament coil. Defaults to 0.
            at (float, optional): Amperage of coil. Defaults to nt Amps.
            theta (float, optional): Rotation angle in radians. Defaults to 0.
            **kwargs: Additional arguments to store in the coil object.
        """
        self.r = r
        self.z = z
        self.dr = dr
        self.dz = dz
        self.nt = nt
        if at is None:
            at = nt
        self.at = at
        self.theta = theta
        self.fils = None
        self._L = None
        self._r_c = None
        self._z_c = None

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

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

    @property
    def r_c(self):  # noqa: D102
        return self.r if self._r_c is None else self._r_c

    @property
    def z_c(self):  # noqa: D102
        return self.z if self._z_c is None else self._z_c

    @property
    def conductor_length(self):  # noqa: D102
        return self.r * self.nt * 2 * math.pi

    def L_best(self):
        """Inductance by best formula."""
        if self.dz < 0.2 * self.r:
            # Lyle's formula is not valid for long coils
            # as its an expansion in dz/r
            return self.L_Lyle6()
        else:
            return self.L_filament()

    @property
    def L(self):  # noqa: D102
        if self._L is None:
            self._L = self.L_best()
        return self._L

    @L.setter
    def L(self, value):
        self._L = value

    def filamentize(self, nr, nz):
        """Create an array of filaments to represent the coil."""
        self.nr = nr
        self.nz = nz
        self.fils = filament_coil(
            self.r, self.z, self.dr, self.dz, self.nt, nr, nz, theta=self.theta
        )

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

    def L_filament(self, nr=0, nz=0):
        """Inductance by filamentation."""
        if nr != 0 and nz != 0:
            self.filamentize(nr, nz)
        cond = getattr(self, "conductor", None)
        if cond is None:
            return self_inductance_by_filaments(
                self.fils,
                conductor=Shape.rect.name,
                dr=self.dr / self.nr,
                dz=self.dz / self.nz,
            )
        elif "round" in cond.shape.value:
            return self_inductance_by_filaments(
                self.fils, conductor=cond.shape.name, a=cond.r
            )
        elif "rect" in cond.shape.value:
            return self_inductance_by_filaments(
                self.fils, conductor=cond.shape.name, dr=cond.dr, dz=cond.dz
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

    def __init__(self, coils: list[Coil], **kwargs):
        """Create a composite coil from a list of _filamented_ coils."""
        self.coils = coils
        nt = sum(coil.nt for coil in coils)
        at = sum(coil.at for coil in coils)
        r1 = min(coil.r - coil.dr for coil in coils)
        r2 = max(coil.r + coil.dr for coil in coils)
        r = (r1 + r2) / 2
        dr = r2 - r1
        z1 = min(coil.z - coil.dz for coil in coils)
        z2 = max(coil.z + coil.dz for coil in coils)
        z = (z1 + z2) / 2
        dz = z2 - z1
        Coil.__init__(self, r, z, dr, dz, nt, at, **kwargs)
        self._r_c = sum(coil.r * coil.nt for coil in coils) / nt
        self._z_c = sum(coil.z * coil.nt for coil in coils) / nt
        self.fils = np.concatenate([coil.fils for coil in coils])

    @property
    def conductor_length(self):  # noqa: D102
        return sum(coil.conductor_length for coil in self.coils)

    def L_best(self):
        """Inductance of composite coils by best formula."""
        L = 0
        for i, c1 in enumerate(self.coils):
            for j, c2 in enumerate(self.coils):
                if i == j:
                    L += c1.L
                else:
                    L += c1.M_filament(c2)
        return L


def coilset_mutual_inductance(coils: list[Coil]):
    """Get the inductance matrix of a set of coils.

    Args:
        coils (list[Coil]): set of coils

    Returns:
        np.ndarray: 2D array of inductances
    """
    muts = np.zeros([len(coils), len(coils)], dtype=float)
    for i, ci in enumerate(coils):
        for j, cj in enumerate(coils):
            if i == j:
                muts[i, j] = ci.L
            else:
                muts[i, j] = ci.M_filament(cj)
    return muts


def coilset_Fz_greens(coils: list[Coil]):
    """Calculate the vertical force matrix of a set of coils.

    Args:
        coils (list[Coil]): set of coils

    Returns:
        np.ndarray: 2D array of vertical forces per amp**2
    """
    mfz = np.zeros([len(coils), len(coils)], dtype=float)
    for i, ci in enumerate(coils):
        for j, cj in enumerate(coils):
            if i != j:
                mfz[i, j] = ci.Fz_filament(cj)
    return mfz


def coilset_Fr_greens(coils: list[Coil]):
    """Calculate the radial force matrix of a set of coils.

    Args:
        coils (list[Coil]): set of coils

    Returns:
        np.ndarray: 2D array of radial forces per amp**2
    """
    muts = np.zeros([len(coils), len(coils)], dtype=float)
    for i, ci in enumerate(coils):
        for j, cj in enumerate(coils):
            if i == j:
                muts[i, j] = ci.Fr_self()
            else:
                muts[i, j] = ci.Fr_filament(cj)
    return muts
