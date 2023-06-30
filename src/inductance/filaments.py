"""Filament coil calculations.

This is a dictionary based API, making it easier to define coils.
probably it should be made into a proper object class
but really I only use it for benchmarking against LDX values
and I wanted to copy from old Mathematica routines and other tests.

Filaments are defined as an numpy 3 element vector
 - r, z, and n.

A coil is defined as an dictionary with
r1, r2, z1, and z2 and nt turns
"""

from .inductance import (
    L_long_solenoid_butterworth,
    L_lyle4,
    L_lyle6,
    L_lyle6_appendix,
    L_maxwell,
    L_thin_wall_babic_akyel,
    L_thin_wall_lorentz,
    dLdR_lyle6,
    filament_coil,
    mutual_inductance_of_filaments,
    radial_force_of_filaments,
    self_inductance_by_filaments,
    vertical_force_of_filaments,
)


def FilamentCoil(C: dict, nr: int, nz: int):
    """Break a coil into filaments, as nr by nz array. Adds the filaments to the coil.

    Args:
        C (dict): Coil dictionary (class?)
        nr (int): number of columns to break the coil into
        nz (int): number of rows to break the coil into

    Returns:
        dict: Coil dictionary with filaments added
    """
    C["fil"] = filament_coil(
        float((C["r1"] + C["r2"]) / 2),
        float((C["z2"] + C["z1"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
        nr,
        nz,
    )
    C["dr"] = (C["r2"] - C["r1"]) / nr
    C["dz"] = (C["z2"] - C["z1"]) / nz
    return C


def TotalM0(C1, C2):
    """Mutual inductance of two coils by filamentation."""
    return mutual_inductance_of_filaments(C1["fil"], C2["fil"])


def TotalFz(C1, C2):
    """Vertical force of two coils by filamentation."""
    F_a2 = vertical_force_of_filaments(C1["fil"], C2["fil"])
    return C1["at"] / C1["nt"] * C2["at"] / C2["nt"] * F_a2


def TotalFrOn1(C1, C2):
    """Radial force of on coil 1 by filamentation from second coil and self force."""
    Fr_11 = (
        0.5
        * C1["at"]
        / C1["nt"]
        * dLdR_lyle6(
            (C1["r2"] + C1["r1"]) / 2.0,
            C1["r2"] - C1["r1"],
            C1["z2"] - C1["z1"],
            C1["nt"],
        )
    )

    Fr_12 = (
        C1["at"]
        / C1["nt"]
        * C2["at"]
        / C2["nt"]
        * radial_force_of_filaments(C1["fil"], C2["fil"])
    )
    return Fr_11 + Fr_12


def LMaxwell(C):
    """Inductance by Maxwell's formula."""
    return L_maxwell(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def LLyle4(C):
    """Inductance by Lyle's formula, 4th order."""
    return L_lyle4(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def LLyle6(C):
    """Inductance by Lyle's formula, 6th order."""
    return L_lyle6(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def LLyle6A(C):
    """Inductance by Lyle's formula, 6th order, appendix."""
    return L_lyle6_appendix(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def Lfil(Coil):
    """Inductance by filamentation.

    Args:
        Coil (dict): coil dictionary, must have run FilamentCoil first.

    Returns:
        float: inductance of coil in Henries
    """
    return self_inductance_by_filaments(
        Coil["fil"], conductor="rect", dr=Coil["dr"], dz=Coil["dz"]
    )


def LLS(C):
    """Inductance by Butterworth's formula for a long solenoid."""
    return L_long_solenoid_butterworth(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def LBA(C):
    """Inductance by Babic and Akyel's formula for a thin wall solenoid."""
    return L_thin_wall_babic_akyel(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )


def LL(C):
    """Inductance by Lorentz's formula for a thin wall solenoid."""
    return L_thin_wall_lorentz(
        float((C["r1"] + C["r2"]) / 2),
        float(C["r2"] - C["r1"]),
        float(C["z2"] - C["z1"]),
        C["nt"],
    )
