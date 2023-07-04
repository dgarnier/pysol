"""Mutual inductance calculations for coils.

author: Darren Garnier <garnier@mit.edu>


"""
import math

import numpy as np

from ._numba import njit

# from .elliptics import ellipke
from .filaments import mutual_inductance_fil, mutual_inductance_of_filaments

MU0 = 4e-7 * math.pi  # permeability of free space


@njit
def mutual_rayleigh(r1, z1, dr1, dz1, n1, r2, z2, dr2, dz2, n2):
    """Mutual inductance of two coils by Rayleigh's Quadrature Method.

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
def mutual_lyle(r1, z1, dr1, dz1, n1, r2, z2, dr2, dz2, n2):
    """Mutual inductance of two coils by Lyle's Method of Equivalent Filaments.

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
    fils1 = _lyle_equivalent_filaments(r1, z1, dr1, dz1)
    fils2 = _lyle_equivalent_filaments(r2, z2, dr2, dz2)
    return n1 * n2 * mutual_inductance_of_filaments(fils1, fils2)


@njit
def _lyle_equivalent_filaments(r, z, dr, dz):
    """Compute the equivalent filament locations for Lyle's method.

    Args:
        r (float): inner radius of coil
        z (float): inner height of coil
        dr (float): radial width of coil
        dz (float): height of coil

    Returns:
        numpy.ndarray: equivalent filament locations
    """
    if dr < dz:
        req = r * (1 + dr**2 / (24 * r**2))
        beta = math.sqrt((dz**2 - dr**2) / 12)
        fils = [[req, z - beta, 0.5], [req, z + beta, 0.5]]
    elif dr > dz:
        req = r * (1 + dz**2 / (24 * r**2))
        delta = math.sqrt((dz**2 - dr**2) / 12)
        fils = [[req - delta, z, 0.5], [req + delta, z, 0.5]]
    else:
        req = r * (1 + dz**2 / (24 * r**2))
        fils = [[req, z, 1]]

    return np.array(fils)


def section_coil(r, z, dr, dz, nt, nr, nz):
    """Create an array of filaments, each with its own radius, height, and amperage.

    r : Major radius of coil center.
    z : Vertical center of coil.
    dr : Radial width of coil.
    dz : Height of coil.
    nt : number of turns in coil
    nr : Number of radial slices
    nz : Number of vertical slices

    Returns:    Array of shape (nr*nz) x 5 of r, z, dr, dz, n for each section

    """
    rd = np.linspace(-dr * (nr - 1) / nr / 2, dr * (nr - 1) / nr / 2, nr)
    zd = np.linspace(-dz * (nz - 1) / nz / 2, dz * (nz - 1) / nz / 2, nz)
    Rg, Zg = np.meshgrid(rd, zd)

    R = r + Rg
    Z = z + Zg

    DR = np.full_like(R, dr / nr)
    DZ = np.full_like(R, dz / nz)
    NT = np.full_like(R, float(nt) / (nr * nz))

    return np.dstack([R, Z, DR, DZ, NT]).reshape(nr * nz, 5)


def mutual_sectioning_lyle(r1, z1, dr1, dz1, n1, r2, z2, dr2, dz2, n2):
    """Mutual inductance by sectioning of two coils by Lyle's Method of Equivalent Filaments.

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
    # FIXME
    # use section_coil
    # need to be clever to keep it numba compatible

    fils1 = _lyle_equivalent_filaments(r1, z1, dr1, dz1)
    fils2 = _lyle_equivalent_filaments(r2, z2, dr2, dz2)
    return n1 * n2 * mutual_inductance_of_filaments(fils1, fils2)
