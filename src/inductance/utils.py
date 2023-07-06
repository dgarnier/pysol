"""Utility functions for inductance calculations.

This module contains utility functions for inductance calculations.
"""
import math

import numpy as np

from ._numba import njit


@njit
def _lyle_terms(dz, dr):
    #  helper functions for most self inductance equations.
    # b : cylindrical height of coil
    # c : radial width of coil

    # when b/c or c/b is very large or small, this diverges
    # and gives inaccurate results when b/c ~ 1e6 or 1e-6
    # now added a check for this and use a series expansion
    # phi should approach 1.5 and GMD should approach (b+c)*exp(-1.5)

    # for small b/c, u and wp need to be calculated with more precision
    # for small c/b, v and w need more precision
    # will try to use limit functions

    if dz < dr:
        b, c = dz, dr
    else:
        c, b = dr, dz

    boc = b / c
    boc2 = boc**2

    if boc == 0:
        u, v, w, p = 0, 1, 0, 1
    elif boc < 1e-5:
        u = -boc2 * math.log(boc2) + boc2**2 - boc2**3 / 2
        v = 1 - boc2 / 2 + boc2**2 / 3
        w = math.pi / 2 * (b / c) - boc2 + boc2**2 / 3
        p = 1 - boc2 / 3 + boc2**2 / 5
    else:
        u = ((b / c) ** 2) * math.log(1 + (c / b) ** 2)
        v = ((c / b) ** 2) * math.log(1 + (b / c) ** 2)
        w = (b / c) * math.atan2(c, b)
        p = (c / b) * math.atan2(b, c)

    if dr > dz:
        u, v = v, u
        w, p = p, w

    d = np.sqrt(b**2 + c**2)  # diagnonal length
    phi = (u + v + 25) / 12 - 2 * (w + p) / 3
    GMD = d * np.exp(-phi)  # geometric mean radius of section GMD
    return d, u, v, w, p, phi, GMD


def rectangle_GMD(dr, dz):
    """Geometric mean radius of a rectangle.

    Args:
        dr (float): width of rectangle
        dz (_type_): height of rectangle
    """
    return _lyle_terms(dz, dr)[-1]


def section_coil(r, z, dr, dz, nt, nr, nz, theta=0):
    """Create an array of subcoils.

    Each coil will have its with its own radius, height,
    number of turns.

    Args:
        r  (float): Major radius of coil center.
        z  (float): Vertical center of coil.
        dr (float): Radial width of coil.
        dz (float): Height of coil.
        nt (float): number of turns in coil
        nr (float): Number of radial slices
        nz (float): Number of vertical slices
        theta (float): Rotation angle in radians

    Returns:
        (np.ndarray) : Array of shape (nr*nz) x 5 of r, z, dr, dz, nt)
        for each section
    """
    rd = np.linspace(-dr * (nr - 1) / nr / 2, dr * (nr - 1) / nr / 2, nr)
    zd = np.linspace(-dz * (nz - 1) / nz / 2, dz * (nz - 1) / nz / 2, nz)

    Rg, Zg = np.meshgrid(rd, zd)

    R = r + Rg * np.cos(theta) - Zg * np.sin(theta)
    Z = z + Rg * np.sin(theta) + Zg * np.cos(theta)

    DR = np.full_like(R, dr / nr)
    DZ = np.full_like(R, dz / nz)
    NT = np.full_like(R, float(nt) / (nr * nz))
    TH = np.full_like(R, theta)

    return np.dstack([R, Z, DR, DZ, NT, TH]).reshape(nr * nz, 6)
