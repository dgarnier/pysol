"""Mutual inductance calculations for coils.

author: Darren Garnier <garnier@mit.edu>


"""
import math

import numpy as np

from ._numba import njit
from .filaments import mutual_inductance_fil, mutual_inductance_of_filaments
from .utils import section_coil


@njit
def mutual_rayleigh(r1, z1, dr1, dz1, n1, r2, z2, dr2, dz2, n2):
    """Mutual inductance of two coils by Rayleigh's Quadrature Method.

    reproduced in :
    Rosa and Grover, "Formulas for the Mutual Inductance of
    Coaxial Circular Coils of Rectangular Section,"
    Bull. Natl. Bur. Stand., vol 8, no. 1, p. 34-35; 1911.

    Args:
        r1 (float): inner radius of coil 1
        z1 (float): inner height of coil 1
        dr1 (float): radial width of coil 1
        dz1 (float): height of coil 1
        n1 (int): number of turns in coil 1
        r2 (float): inner radius of coil 2
        z2 (float): inner height of coil 2
        dr2 (float): radial width of coil 2
        dz2 (float): height of coil 2
        n2 (int): number of turns in coil 2

    Returns:
        float: mutual inductance of the two coils
    """
    m_ray = 0
    # define the quadrature points
    rzn1 = np.array(
        [
            [r1, z1, 1],
            [r1 - dr1 / 2, z1, 1],
            [r1, z1 - dz1 / 2, 1],
            [r1 + dr1 / 2, z1, 1],
            [r1, z1 + dz1 / 2, 1],
        ]
    )
    rzn2 = np.array(
        [
            [r2, z2, 1],
            [r2 - dr2 / 2, z2, 1],
            [r2, z2 - dz2 / 2, 1],
            [r2 + dr2 / 2, z2, 1],
            [r2, z2 + dz2 / 2, 1],
        ]
    )
    # apply Rayleigh's Quadrature Method
    m_ray = -mutual_inductance_fil(rzn1[0, :], rzn2[0, :]) * 2
    m_ray += mutual_inductance_fil(rzn1[1, :], rzn2[0, :])
    m_ray += mutual_inductance_fil(rzn1[2, :], rzn2[0, :])
    m_ray += mutual_inductance_fil(rzn1[3, :], rzn2[0, :])
    m_ray += mutual_inductance_fil(rzn1[4, :], rzn2[0, :])
    m_ray += mutual_inductance_fil(rzn1[0, :], rzn2[1, :])
    m_ray += mutual_inductance_fil(rzn1[0, :], rzn2[2, :])
    m_ray += mutual_inductance_fil(rzn1[0, :], rzn2[3, :])
    m_ray += mutual_inductance_fil(rzn1[0, :], rzn2[4, :])
    return n1 * n2 * m_ray / 6


@njit
def lyle_equivalent_filaments(r, z, dr, dz, nt, fils):
    """Compute the equivalent filament locations for Lyle's method.

    Using Lyle's Method of Equivalent Filaments.

    originally from:
        Lyle, Phil. Mag., 3, p. 310; 1902.

    reproduced in:
        Rosa and Grover, "Formulas for the Mutual Inductance of
        Coaxial Circular Coils of Rectangular Section,"
        Bull. Natl. Bur. Stand., vol 8, no. 1, p. 38-39; 1911.

    Args:
        r (float): inner radius of coil
        z (float): inner height of coil
        dr (float): radial width of coil
        dz (float): height of coil
        nt (float): number of turns in coil
        fils (numpy.ndarray): array of filaments 2 x (r, z, n)
    """
    if dr < dz:
        req = r * (1 + dr**2 / (24 * r**2))
        beta = math.sqrt((dz**2 - dr**2) / 12)
        fils[:, 0] = req
        fils[:, 1] = z - beta, z + beta
    elif dr > dz:
        req = r * (1 + dz**2 / (24 * r**2))
        delta = math.sqrt((dr**2 - dz**2) / 12)
        fils[:, 0] = req - delta, req + delta
        fils[:, 1] = z
    else:
        req = r * (1 + dz**2 / (24 * r**2))
        fils[:, 0] = req
        fils[:, 1] = z
    fils[:, 2] = 0.5 * nt


@njit
def mutual_lyles_method(r1, z1, dr1, dz1, nt1, r2, z2, dr2, dz2, nt2):
    """Mutual inductance of two coils by Lyle's method.

    Using Lyle's Method of Equivalent Filaments.

    originally from:
    Lyle, Phil. Mag., 3, p. 310; 1902.

    reproduced:
    Rosa and Grover, "Formulas for the Mutual Inductance of
    Coaxial Circular Coils of Rectangular Section,"
    Bull. Natl. Bur. Stand., vol 8, no. 1, p. 38-39; 1911.

    Args:
        r1 (float): inner radius of coil 1
        z1 (float): inner height of coil 1
        dr1 (float): radial width of coil 1
        dz1 (float): height of coil 1
        nt1 (int): number of turns in coil 1
        r2 (float): inner radius of coil 2
        z2 (float): inner height of coil 2
        dr2 (float): radial width of coil 2
        dz2 (float): height of coil 2
        nt2 (int): number of turns in coil 2

    Returns:
        float: mutual inductance of the two coils
    """
    fils1 = np.zeros(2, 3)
    fils2 = np.zeros(2, 3)

    lyle_equivalent_filaments(r1, z1, dr1, dz1, nt1, fils1)
    lyle_equivalent_filaments(r2, z2, dr2, dz2, nt2, fils1)
    return mutual_inductance_of_filaments(fils1, fils2)


@njit
def lyle_equivalent_subcoil_filaments(subcoils):
    """Compute the equivalent filament locations for set of subcoils."""
    fils = np.zeros((subcoils.shape[0], 2, 3))
    for i in range(subcoils.shape[0]):
        lyle_equivalent_filaments(*subcoils[i, :], fils[i, :, :])
    return fils


def mutual_sectioning_lyle(
    r1, z1, dr1, dz1, nt1, nr1, nz1, r2, z2, dr2, dz2, nt2, nr2, nz2
):
    """Mutual inductance by sectioning of two coils by Lyle's Method.

    Section cois into subcoils and compute mutual inductance of each set
    of subcoil using Lyle's method of equivalent filaments.

    Args:
        r1 (float): inner radius of coil 1
        z1 (float): inner height of coil 1
        dr1 (float): radial width of coil 1
        dz1 (float): height of coil 1
        nt1 (float): number of turns in coil 1
        nr1 (int): number of radial sections in coil 1
        nz1 (int): number of vertical sections in coil 1
        r2 (float): inner radius of coil 2
        z2 (float): inner height of coil 2
        dr2 (float): radial width of coil 2
        dz2 (float): height of coil 2
        nt2 (float): number of turns in coil 2
        nr2 (int): number of radial sections in coil 2
        nz2 (int): number of vertical sections in coil 2

    Returns:
        float: mutual inductance of the two coils
    """
    subs_1 = section_coil(r1, z1, dr1, dz1, nt1, nr1, nz1)
    subs_2 = section_coil(r2, z2, dr2, dz2, nt2, nr2, nz2)

    fils_1 = lyle_equivalent_subcoil_filaments(subs_1).reshape(-1, 3)
    fils_2 = lyle_equivalent_subcoil_filaments(subs_2).reshape(-1, 3)

    return mutual_inductance_of_filaments(fils_1, fils_2)
