"""Inductance calculations for coils.

author: Darren Garnier <garnier@mit.edu>

basic equations come from analytic approximations of old.
tested in real life with the LDX Fcoil / Ccoil / Lcoil system.

One from Maxwell himself, and better ones from:

Lyle, T. R.  "On the Self-inductance of Circular Coils of
    Rectangular Section".  Roy. Soc. London. A.  V213 (1914) pp 421-435.
    https://doi.org/10.1098/rsta.1914.0009

Unfortunately, Lyle doesn't work that well with large dz/R coils.  Other
approximations are also included.

This code now uses numba to do just-in-time compiliation and parallel execution
to greatly increase speed. Also requires numba-scipy for elliptical functions.
numba-scipy can be fragile and sometimes needs to be "updated" before installation
to the newest version of numba.
"""
import math
import os

import numpy as np

# try to enable use without numba.  Those with guvectorize will not work.
try:
    from numba import guvectorize, jit, njit, prange

    if os.getenv("NUMBA_DISABLE_JIT", "0") == "1":  # pragma: no cover
        raise ImportError

except ImportError:  # pragma: no cover
    from inspect import currentframe, getframeinfo
    from warnings import warn_explicit

    WARNING = (
        "Numba disabled. Mutual inductance calculations "
        + "will not be accelerated and some API will not be available."
    )
    warn_explicit(
        WARNING,
        RuntimeWarning,
        "inductance.py",
        getframeinfo(currentframe().f_back).lineno,  # type: ignore
    )

    def _jit(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # called as @decorator
            return args[0]
        else:
            # called as @decorator(*args, **kwargs)
            return lambda f: f

    def _guvectorize(*args, **kwds):
        def fake_decorator(f):
            warning = f"{f.__name__} requires Numba JIT."
            warn_explicit(
                warning,
                RuntimeWarning,
                "inductance.py",
                getframeinfo(currentframe().f_back.f_back).lineno,  # type: ignore
            )
            return lambda f: None

        return fake_decorator

    guvectorize = _guvectorize
    njit = _jit
    prange = range
    jit = _jit


from .elliptics import ellipke

try:
    from mpmath import mp  # type: ignore

    USE_MPMATH = True
except ImportError:  # pragma: no cove
    USE_MPMATH = False

MU0 = 4e-7 * math.pi  # permeability of free space


@njit
def _lyle_terms(b, c):
    #  helper functions for most self inductance equations.
    # b : cylindrical height of coil
    # c : radial width of coil

    # FIXME:  when b/c or c/b is very large or small, this diverges
    #         and gives inaccurate results when b/c ~ 1e6 or 1e-6
    # phi should approach 1.5 and GMD should approach (b+c)*exp(-1.5)

    d = np.sqrt(b**2 + c**2)  # diagnonal length
    u = ((b / c) ** 2) * 2 * np.log(d / b)
    v = ((c / b) ** 2) * 2 * np.log(d / c)
    w = (b / c) * np.arctan(c / b)
    wp = (c / b) * np.arctan(b / c)
    phi = (u + v + 25) / 12 - 2 * (w + wp) / 3
    GMD = d * np.exp(-phi)  # geometric mean radius of section GMD

    return d, u, v, w, wp, phi, GMD


if USE_MPMATH:  # pragma: no cover

    def _lyle_terms_mp(b, c):
        # b : cylindrical height of coil
        # c : radial width of coil

        # this does NOT fix problem with b/c ~ 1e6 or 1e-6

        d = mp.sqrt(b**2 + c**2)  # diagnonal length
        u = ((b / c) ** 2) * 2 * mp.log(d / b)
        v = ((c / b) ** 2) * 2 * mp.log(d / c)
        w = (b / c) * mp.atan(c / b)
        wp = (c / b) * mp.atan(b / c)
        phi = (u + v + 25) / 12 - 2 * (w + wp) / 3
        GMD = d * mp.exp(-phi)  # geometric mean radius of section GMD
        return d, u, v, w, wp, phi, GMD


@njit
def L_maxwell(r, dr, dz, n):
    """Self inductance of a rectangular coil with constant current density by Maxwell.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, wp, phi, GMD = _lyle_terms(b, c)
    L = MU0 * (n**2) * a * (math.log(8 * a / GMD) - 2)
    return L


@njit
def L_round(r, a, n):
    """Self inductance of a round conductor coil with constant current density.

    Args:
        r (float): coil centerline radius
        a (float): coil conductor radius
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    L = MU0 * (n**2) * r * (math.log(8 * r / a) - 1.75)
    return L


@njit
def L_hollow_round(r, a, n):
    """Self inductance of a round conductor coil with skin current.

    Args:
        r (float): coil centerline radius
        a (float): coil conductor radius
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    L = MU0 * (n**2) * r * (math.log(8 * r / a) - 2)
    return L


@njit
def L_lyle4_eq3(r, dr, dz, n):
    """Self inductance of a rectangular coil via Lyle to 4th order, Eq3.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, wp, phi, GMD = _lyle_terms(b, c)
    p2 = 1 / (2**5 * 3 * d**2) * (3 * b**2 + c**2)
    q2 = (
        1
        / (2**5 * 3 * d**2)
        * (
            1 / 2 * b**2 * u
            - 1 / 10 * c**2 * v
            - 16 / 5 * b**2 * w
            - (3 * b**2 + c**2) * phi
            + 69 / 20 * b**2
            + 221 / 60 * c**2
        )
    )
    p4 = (
        1
        / (2**11 * 3**2 * 5 * d**4)
        * (-90 * b**4 + 105 * (b * c) ** 2 + 22 * c**4)
    )
    q4 = (
        1
        / (2**11 * 3**2 * 5 * d**4)
        * (
            -(-90 * b**4 + 105 * (b * c) ** 2 + 22 * c**4) * phi
            - 69 / 28 * c**4 * v
            - u / 4 * (115 * b**4 - 480 * (b * c) ** 2)
            + 2**8 * w / 7 * (6 * b**4 - 7 * (b * c) ** 2)
            - 1
            / (2**3 * 5 * 7)
            * (36590 * b**4 - 2035 * (b * c) ** 2 - 11442 * c**4)
        )
    )

    ML = np.log(8 * a / GMD)

    # equation #3

    eq3 = (
        MU0
        * (n**2)
        * a
        * (ML - 2 + (d / a) ** 2 * (p2 * ML + q2) + (d / a) ** 4 * (p4 * ML + q4))
    )
    return eq3


# equation 4.. slightly different result... not sure which is better.
# equation 3 above matches what I did for the 6th order.
# and it also seems to match the 4th order in the paper.


@njit
def L_lyle4_eq4(r, dr, dz, n):
    """Self inductance of a rectangular coil via Lyle to 4th order, Eq4.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, wp, phi, GMD = _lyle_terms(b, c)
    p2 = 1 / (2**5 * 3 * d**2) * (3 * b**2 + c**2)
    q2 = (
        1
        / (2**5 * 3 * d**2)
        * (
            1 / 2 * b**2 * u
            - 1 / 10 * c**2 * v
            - 16 / 5 * b**2 * w
            - (3 * b**2 + c**2) * phi
            + 69 / 20 * b**2
            + 221 / 60 * c**2
        )
    )
    p4 = (
        1
        / (2**11 * 3**2 * 5 * d**4)
        * (-90 * b**4 + 105 * (b * c) ** 2 + 22 * c**4)
    )
    q4 = (
        1
        / (2**11 * 3**2 * 5 * d**4)
        * (
            -(-90 * b**4 + 105 * (b * c) ** 2 + 22 * c**4) * phi
            - 69 / 28 * c**4 * v
            - u / 4 * (115 * b**4 - 480 * (b * c) ** 2)
            + 2**8 * w / 7 * (6 * b**4 - 7 * (b * c) ** 2)
            - 1
            / (2**3 * 5 * 7)
            * (36590 * b**4 - 2035 * (b * c) ** 2 - 11442 * c**4)
        )
    )

    m1 = p2
    n1 = -(p2 + q2)
    m2 = p4
    n2 = -(p4 + q4) + 1 / 2 * (m1 - n1) ** 2
    n3 = (m1 - n1) * (m2 - n2 - 1 / 6 * (m1 - n1) * (m1 + 2 * n1))

    A = a * (1 + m1 * (d / a) ** 2 + m2 * (d / a) ** 4)
    R = GMD * (1 + n1 * (d / a) ** 2 + n2 * (d / a) ** 4 + n3 * (d / a) ** 6)

    eq4 = MU0 * (n**2) * A * (np.log(8 * A / R) - 2)
    return eq4


L_lyle4 = L_lyle4_eq3


@njit
def L_lyle6(r, dr, dz, n):
    """Self inductance of a rectangular coil via Lyle to 6th order.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, ww, phi, GMD = _lyle_terms(b, c)
    bd2 = (b / d) ** 2
    cd2 = (c / d) ** 2
    da2 = (d / a) ** 2
    ML = np.log(8 * a / d)

    # after further reduction in mathematica... all the terms.
    f = (
        ML
        + (1 + u + v - 8 * (w + ww)) / 12.0  # 0th order in d/a
        + (
            da2
            * (
                cd2 * (221 + 60 * ML - 6 * v)
                + 3 * bd2 * (69 + 60 * ML + 10 * u - 64 * w)
            )
        )
        / 5760.0  # 2nd order
        + (
            da2**2
            * (
                2 * cd2**2 * (5721 + 3080 * ML - 345 * v)
                + 5 * bd2 * cd2 * (407 + 5880 * ML + 6720 * u - 14336 * w)
                - 10 * bd2**2 * (3659 + 2520 * ML + 805 * u - 6144 * w)
            )
        )
        / 2.58048e7  # 4th order
        + (
            da2**3
            * (
                3 * cd2**3 * (4308631 + 86520 * ML - 10052 * v)
                - 14
                * bd2**2
                * cd2
                * (617423 + 289800 * ML + 579600 * u - 1474560 * w)
                + 21 * bd2**3 * (308779 + 63000 * ML + 43596 * u - 409600 * w)
                + 42 * bd2 * cd2**2 * (-8329 + 46200 * ML + 134400 * u - 172032 * w)
            )
        )
        / 1.73408256e10  # 6th order
    )
    L = MU0 * (n**2) * a * f
    # print("Lyle6 r: %.4g, dr: %.4g, dz: %4g, n: %d, L: %.8g"%(a,c,b,n,L))
    return L


@njit
def dLdR_lyle6(r, dr, dz, n):
    """Radial derivative of self inductance of a rectangular coil via Lyle to 6th order.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: radial derivative of inductance in Henrys/meter
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, ww, phi, GMD = _lyle_terms(b, c)
    bd2 = (b / d) ** 2
    cd2 = (c / d) ** 2
    da2 = (d / a) ** 2

    ML = np.log(8 * a / d)

    f = (
        ML
        + (13 + u + v - 8 * w - 8 * ww) / 12.0  # zero
        + (
            da2
            * (
                cd2 * (-161 - 60 * ML + 6 * v)
                - 3 * bd2 * (9 + 60 * ML + 10 * u - 64 * w)
            )
        )
        / 5760.0  # 2nd order
        + (
            da2**2
            * (
                -2 * cd2**2 * (14083 + 9240 * ML - 1035 * v)
                - 15 * bd2 * cd2 * (-1553 + 5880 * ML + 6720 * u - 14336 * w)
                + 30 * bd2**2 * (2819 + 2520 * ML + 805 * u - 6144 * w)
            )
        )
        / 2.58048e7  # 4th order
        + (
            da2**3
            * (
                -3 * cd2**3 * (4291327 + 86520 * ML - 10052 * v)
                + 14
                * bd2**2
                * cd2
                * (559463 + 289800 * ML + 579600 * u - 1474560 * w)
                - 21 * bd2**3 * (296179 + 63000 * ML + 43596 * u - 409600 * w)
                - 42 * bd2 * cd2**2 * (-17569 + 46200 * ML + 134400 * u - 172032 * w)
            )
        )
        / 3.46816512e9  # 6th order
    )

    dLdR = 4e-7 * np.pi * (n**2) * f
    return dLdR


@njit
def L_lyle6_appendix(r, dr, dz, n):
    """Self inductance of a rectangular coil via Lyle to 6th order, appendix.

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    d, u, v, w, wp, phi, GMD = _lyle_terms(b, c)

    p6 = (
        1
        / (2**16 * 3 * 5 * 7 * d**6)
        * (525 * b**6 - 1610 * b**4 * c**2 + 770 * b**2 * c**4 + 103 * c**6)
    )
    q6 = (
        1
        / (2**16 * 3 * 5 * 7 * d**6)
        * (
            0
            + (3633 / 10 * b**6 - 3220 * b**4 * c**2 + 2240 * b**2 * c**4) * u
            - (359 / 30) * c**6 * v
            - 2**11
            * (5 / 3 * b**6 - 4 * b**4 * c**2 + 7 / 5 * b**2 * c**4)
            * w
            + 2161453 / (2**3 * 3 * 5 * 7) * b**6
            - 617423 / (2**2 * 3**2 * 5) * b**4 * c**2
            - 8329 / (2**2 * 3 * 5) * b**2 * c**4
            + 4308631 / (2**3 * 3 * 5 * 7) * c**6
        )
    )

    # just add the correction to the 4th order solution
    L6 = L_lyle4_eq3(r, dr, dz, n) + 4e-7 * np.pi * (n**2) * a * (d / a) ** 6 * (
        p6 * np.log(8 * a / d) + q6
    )
    # print("Lyle6A r: %.4g, dr: %.4g, dz: %4g, n: %d, L: %.8g"%(a,c,b,n,L6))
    return L6


if USE_MPMATH:

    def L_lyle6_mp(r, dr, dz, n):
        """Self inductance of a rectangular coil via Lyle to 6th order.

        This function uses the mpmath arbitrary precision library to
        calculate to arbitrary precision.  The default precision is 30.

        Args:
            r (float): coil centerline radius
            dr (float): coil radial width
            dz (float): coil height
            n (int): number of turns

        Returns:
            float: coil self inductance in Henrys
        """
        mp.dps = 30
        a = mp.mpf(r)
        b = mp.mpf(dz)
        c = mp.mpf(dr)
        d, u, v, w, wp, phi, GMD = _lyle_terms_mp(b, c)
        ML = mp.log(8 * a / d)

        f = (
            ML
            + (u + v + 1) / 12
            - mp.mpf(2) / 3 * (w + wp)
            + 1
            / (2**5 * 3 * a**2)
            * (
                (3 * b**2 + c**2) * ML
                + mp.mpf(1) / 2 * b**2 * u
                - 1.0 / 10 * c**2 * v
                - 16.0 / 5 * b**2 * w
                + 69.0 / 20 * b**2
                + 221.0 / 60 * c**2
            )
            + mp.mpf(1)
            / (2**11 * 3 * 5 * a**4)
            * (
                (-30 * b**4 + 35 * b**2 * c**2 + mp.mpf(22) / 3 * c**4) * ML
                - (115 * b**4 - 480 * b**2 * c**2) / 12 * u
                - mp.mpf(23) / 28 * c**4 * v
                + (6 * b**4 - 7 * b**2 * c**2) / 21 * 2**8 * w
                - (36590 * b**4 - 2035 * b**2 * c**2 - 11442 * c**4)
                / (2**3 * 3 * 5 * 7)
            )
            + (mp.mpf(1) / (2**16 * 3 * 5 * 7 * a**6))
            * (
                (
                    525 * b**6
                    - 1610 * b**4 * c**2
                    + 770 * b**2 * c**4
                    + 103 * c**6
                )
                * ML
                + (
                    mp.mpf(3633) / 10 * b**6
                    - 3220 * b**4 * c**2
                    + 2240 * b**2 * c**4
                )
                * u
                - mp.mpf(359) / 30 * c**6 * v
                - 2**11
                * ((25 * b**6 - 60 * b**4 * c**2 + 21 * b**2 * c**4) / 15)
                * w
                + mp.mpf(2161453) / (2**3 * 3 * 5 * 7) * b**6
                - mp.mpf(617423) / ((2**2) * (3**2) * 5) * b**4 * c**2
                - mp.mpf(8329) / ((2**2) * 3 * 5) * b**2 * c**4
                + mp.mpf(4308631) / (2**3 * 3 * 5 * 7) * c**6
            )
        )
        L = 4e-7 * mp.pi * (n**2) * a * f
        # print("Lyle6mp r: %.4g, dr: %.4g, dz: %4g, n: %d, L: %.8g"%(a,c,b,n,L))
        return L


@njit
def L_long_solenoid(r, _dr, dz, n):
    """Self inductance of a long solenoid.

    Args:
        r (float): coil centerline radius
        _dr (float): coil radial width (ignored)
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    # c = float(dr)
    L1 = MU0 * np.pi * (n**2) * a**2 / b
    return L1


@njit
def L_long_solenoid_butterworth(r, dr, dz, n):
    """Self inductance of a long solenoid by Butterworth's formula.

    As written in Grover, Bulletin of the Bureau of Standards, Vol. 14 pg. 558
    https://nvlpubs.nist.gov/nistpubs/bulletin/14/nbsbulletinv14n4p537_A2b.pdf

    Args:
        r (float): coil centerline radius
        dr (float): coil radial width
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(r)
    b = float(dz)
    c = float(dr)
    L = 4e-7 * np.pi * (n**2) * a / b

    k2 = 4 * a**2 / (4 * a**2 + b**2)
    kp2 = b**2 / (4 * a**2 + b**2)
    k = np.sqrt(k2)
    kp = np.sqrt(kp2)

    # assume dz > 2*r
    ell = k2 / ((1 + kp) * (1 + np.sqrt(kp)) ** 2)
    q = ell / 2 + 2 * (ell / 2) ** 5 + 15 * (ell / 2) ** 9  # + ....

    delta = 2 * q - 2 * q**4 + 2 * q**9
    gamma = q - 4 * q**4 + 9 * q**9
    beta = q**2 + 3 * q**6 + 6 * q**12
    alpha = q**2 + q**6 + q**12

    K = (
        2.0
        / (3 * (1 - delta) ** 2)
        * (1 + 8 * beta / (1 + alpha) + kp2 / k2 * 8 * gamma / (1 - delta))
        - 4 / (3 * np.pi) * k / kp
    )

    D = (
        -1
        / 3
        * (c / a)
        * (
            1
            - c / (4 * a)
            - 1 / (2 * np.pi) * (c / b) * (np.log(8 * a / c) - 23.0 / 12)
            + 1
            / (160 * np.pi)
            * (c / a) ** 3
            * (a / b)
            * (np.log(8 * a / c) - 1.0 / 20)
            - 1.0
            / 4
            * (c / a)
            * (a / b) ** 2
            * (1 - 7 / 4 * (a / b) ** 2 + 17 / 4 * (a / b) ** 4)
            - 1.0 / 96 * (c / a) ** 3 * (a / b) ** 2 * (1 - 39.0 / 10 * (a / b) ** 2)
        )
    )
    D
    # hmm.. not using D, wonder why I bothered to calculate it.. lets look back at this function
    L = 4e-7 * np.pi * (n**2) * (a / b) * (K)
    return L


@njit
def L_thin_wall_babic_akyel(r, _dr, dz, n):
    """Self inductance thin wall solenoid by Babic and Akyel's formula.

    Follow formulae from:
    S. Babic and C. Akyel, "Improvement in calculation of the self- and mutual inductance
    of thin-wall solenoids and disk coils," in IEEE Transactions on Magnetics,
    vol. 36, no. 4, pp. 1970-1975, July 2000, doi: 10.1109/TMAG.2000.875240.

    Args:
        r (float): coil centerline radius
        _dr (float): coil radial width (ignored)
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    a = float(dz) / 2
    # R1 = float(r) - float(dr) / 2
    # R2 = float(r) + float(dr) / 2
    # alpha = R2 / R1
    # lets double check why I'm not using alpha...
    beta = a / r

    k2 = 1 / (1 + beta**2)
    elk, ele = ellipke(k2)
    k = np.sqrt(k2)
    tk = (
        4.0
        / (3 * np.pi * beta * k**3)
        * ((2 * k2 - 1) * ele + (1 - k2) * elk - k**3)
    )
    eq8 = MU0 * np.pi * (n**2) * r / (2 * beta) * tk  # eq 8
    return eq8


@njit
def L_thin_wall_lorentz(r, _dr, dz, n):
    """Self inductance of a thin wall solenoid by Lorentz's formula.

    Args:
        r (float): coil centerline radius
        _dr (float): coil radial width (ignored)
        dz (float): coil height
        n (int): number of turns

    Returns:
        float: coil self inductance in Henrys
    """
    k2 = 4 * r**2 / (4 * r**2 + dz**2)
    elk, ele = ellipke(k2)
    k = math.sqrt(k2)
    f = dz / (k * r) * elk - (dz**2 - 4 * r**2) / (k * r * dz) * ele - 4 * r / dz

    L = MU0 * 2 * (n**2) * r**2 / (3 * dz) * f
    return L


# MUTUAL INDUCTANCE
#  just filament the coils and use elliptical functions.
#  the more you filament, the better your accuracy
#  the filamentation should match your current density
#  and watch out for putting two filaments on top of eachother!


@njit
def mutual_inductance_fil(rzn1, rzn2):
    """Mutual inductance of two filaments.

    Args:
        rzn1 (array): (r, z, n) of first filament
        rzn2 (array): (r, z, n) of second filament

    Returns:
        float: mutual inductance in Henrys
    """
    r1 = rzn1[0]
    r2 = rzn2[0]
    z1 = rzn1[1]
    z2 = rzn2[1]
    n1 = rzn1[2]
    n2 = rzn2[2]

    k2 = 4 * r1 * r2 / ((r1 + r2) ** 2 + (z1 - z2) ** 2)
    elk, ele = ellipke(k2)
    amp = 2 * np.pi * r1 * 4e-7 * r2 / np.sqrt((r1 + r2) ** 2 + (z1 - z2) ** 2)
    M0 = n1 * n2 * amp * ((2 - k2) * elk - 2 * ele) / k2
    return M0


@njit
def vertical_force_fil(rzn1, rzn2):
    """Vertical force  between two filaments per conductor amp.

    Args:
        rzn1 (array): (r, z, n) of first filament
        rzn2 (array): (r, z, n) of second filament

    Returns:
        float: force in Newtons/Amp^2
    """
    r1 = rzn1[0]
    r2 = rzn2[0]
    z1 = rzn1[1]
    z2 = rzn2[1]
    n1 = rzn1[2]
    n2 = rzn2[2]

    BrAt1 = BrGreen(r1, z1, r2, z2)
    F = n1 * n2 * 2 * np.pi * r1 * BrAt1
    return F


@njit
def radial_force_fil(rzn1, rzn2):
    """Radial force  between two filaments per conductor amp.

    Args:
        rzn1 (array): (r, z, n) of first filament
        rzn2 (array): (r, z, n) of second filament

    Returns:
        float: force in Newtons/Amp^2
    """
    r1 = rzn1[0]
    r2 = rzn2[0]
    z1 = rzn1[1]
    z2 = rzn2[1]
    n1 = rzn1[2]
    n2 = rzn2[2]

    BzAt1 = BzGreen(r1, z1, r2, z2)
    F = n1 * n2 * 2 * np.pi * r1 * BzAt1
    return F


#
#  Green's functions for filaments
#


@njit
def AGreen(r, z, a, b):
    """Psi at position r, z, due to a filament with radius a, and z postion b.

    Args:
        r (float): radial position of a point
        z (float): vertical position of a point
        a (float): radius of a filament
        b (float): vertical position of a filament

    Returns:
        float: Psi at r, z in Weber / Amp
    """
    k2 = 4 * a * r / ((r + a) ** 2 + (z - b) ** 2)
    elk, ele = ellipke(k2)
    amp = 4e-7 * a / np.sqrt((r + a) ** 2 + (z - b) ** 2)
    return amp * ((2 - k2) * elk - 2 * ele) / k2


@njit
def BrGreen(r, z, a, b):
    """Br at position r, z, due to a filament with radius a, and z postion b.

    Args:
        r (float): radial position of a point
        z (float): vertical position of a point
        a (float): radius of a filament
        b (float): vertical position of a filament

    Returns:
        float: Br at r, z in Tesla / Amp
    """
    k2 = 4 * a * r / ((r + a) ** 2 + (z - b) ** 2)
    elk, ele = ellipke(k2)
    amp = -2e-7 / np.sqrt((r + a) ** 2 + (z - b) ** 2)
    Gr = (a**2 + r**2 + (z - b) ** 2) / ((r - a) ** 2 + (z - b) ** 2)
    return amp * (z - b) / r * (Gr * ele - elk)


@njit
def BzGreen(r, z, a, b):
    """Bz at position r, z, due to a filament with radius a, and z postion b.

    Args:
        r (float): radial position of a point
        z (float): vertical position of a point
        a (float): radius of a filament
        b (float): vertical position of a filament

    Returns:
        float: Bz at r, z in Tesla / Amp
    """
    k2 = 4 * a * r / ((r + a) ** 2 + (z - b) ** 2)
    elk, ele = ellipke(k2)
    amp = -2e-7 / np.sqrt((r + a) ** 2 + (z - b) ** 2)
    Gz = (a**2 - r**2 - (z - b) ** 2) / ((r - a) ** 2 + (z - b) ** 2)
    return amp * (Gz * ele + elk)


@njit(parallel=True)
def green_sum_over_filaments(gfunc, fil, r, z):
    """Calculate a greens function at position r, z, to an array of filaments.

    Args:
        gfunc (function): Green's function to use
        fil (array): filament array N x (r, z, n)
        r (float): radial position of a point
        z (float): vertical position of a point

    Returns:
        float: sum of the greens function at r, z, due to all filaments
    """
    tmp = np.zeros_like(r)
    tshp = tmp.shape
    tmp = tmp.reshape((tmp.size,))
    rf = r.reshape((tmp.size,))
    zf = z.reshape((tmp.size,))
    for j in prange(len(tmp)):
        for i in range(fil.shape[0]):
            tmp.flat[j] += fil[i, 2] * gfunc(rf[j], zf[j], fil[i, 0], fil[i, 1])
    tmp = tmp.reshape(tshp)
    return tmp


def segment_path(pts, ds=0, close=False):
    """Segment a path into equal length segments.

    Args:
        pts (array): N x 3 array of points
        ds (float, optional): length between points. Defaults to minimun in pts.
        close (bool, optional): Should the path close the points. Defaults to False.

    Returns:
        array: M x 3 array of points along the path
        array: M array of length along the path
    """
    # this will make random points have equal spaces along path
    # from scipy.interpolate import interp1d
    # assume path is x,y,z and the 0 axis is the segment
    # so pts[*,0] is x, pts[*,1] is y, pts[*,2] is z

    if close:
        pts = np.vstack([pts, pts[0]])
    dx = pts[1:] - pts[:-1]
    dsv = np.linalg.norm(dx, axis=1)
    if ds == 0:
        ds = dsv.min()
    s = np.insert(np.cumsum(dsv), 0, 0)
    slen = s[-1]  # length
    snew = np.linspace(0, slen, int(slen / ds))

    # use np.interp instead of scipy.interpolation.interp1d
    segs = np.column_stack([np.interp(snew, s, pts[:, i]) for i in range(3)])
    return segs, snew


@njit
def _loop_segmented_mutual(r, z, pts):
    # segments is array of n x 3 (x,y,z)
    # segments should contain first point at start AND end.
    # r, z is r & z of loop
    M = float(0)
    for i in range(pts.shape[0] - 1):
        midp = (pts[i, :] + pts[i + 1, :]) / 2
        delta = pts[i, :] - pts[i + 1, :]
        rs = np.sqrt(midp[0] ** 2 + midp[1] ** 2)
        zs = midp[2]
        rdphi = (delta[0] * midp[1] - delta[1] * midp[0]) / rs
        M += AGreen(r, z, rs, zs) * rdphi

    return M


@njit(parallel=True)
def mutual_filaments_segmented(fils, pts):
    """Mutual inductance between a set of axisymmetric filaments and a segmented path."""
    M = float(0)
    for i in prange(fils.shape[0]):
        M += fils[i, 2] * _loop_segmented_mutual(fils[i, 0], fils[i, 1], pts)
    return M


def M_filsol_path(fil, pts, n, ds=0):
    """Mutual inductance between a set of axisymmetric filaments and a path from pts."""
    segs, s = segment_path(pts, ds)
    return n * mutual_filaments_segmented(fil, segs)


@njit(parallel=True)
def segmented_self_inductance(pts, s, a):
    """Self inductance of a segmented path by double integral.

    Args:
        pts (N x 3 array): points along the path
        s (array): length along the path
        a (float): radius of wire

    Returns:
        float: self inductance of the path

    Neumann's formula.. double curve integral of
    mu_0/(4*pi) * int * int ( dx1 . dx2 / norm(x1-x2)))
    do all points except where x1-x2 blows up.
    instead follow Dengler https://doi.org/10.7716/aem.v5i1.331
    which makes a approximation for that is good O(mu_0*a)
    lets just assume the thing is broken into small pieces and neglect end points

    this code doesn't work very well.. comparison test sorta fails
    phi = np.linspace(0,2*np.pi,100)
    test_xyz = np.array([[np.cos(p), np.sin(p), 0] for p in phi])
    L_seg = L_approx_path_rect(test_xyz, .01, .01, 1, .1)
    L_maxwell(1, .01, .01, 1), L_lyle6(1, .01, .01, 1), L_seg
    (6.898558527897293e-06, 6.8985981243869525e-06, 6.907313505254537e-06) # Y=1/4 hmmm...
    (6.898558527897293e-06, 6.8985981243869525e-06, 7.064366776069971e-06) # Y=1/2
    """
    ds = s[1] - s[0]  # the real ds and thus the real b
    dx = pts[1:] - pts[:-1]
    x = (pts[1:] + pts[:-1]) / 2
    npts = x.shape[0]
    slen = s[-1]
    b = ds / 2  # does not depend on ds, because of this correction.
    LS = 2 * slen * (np.log(2 * b / a) + 0.25)  # dengler equation 6 correction.
    # seems to work much better when its +.125
    L = 0  # numba REQUIRES prange parallel variable to start at 0 (bug!)
    for i in prange(npts):
        for j in range(npts):
            if i != j:
                L += np.dot(dx[i], dx[j]) / np.linalg.norm(x[i] - x[j])
    return 1e-7 * (L + LS)


def L_approx_path_rect(pts, b, c, n, ds=1):
    """Approximate self inductance of a path of points with a rectangular cross section.

    take a path of points n x 3, with a cross section b x c
    and approximate self inductance using Dengler
    """
    _, _, _, _, _, _, a = _lyle_terms(
        b, c
    )  # get Maxwell mean radius to approximate "wire" radius
    ds *= a
    segs, s = segment_path(pts, ds)
    L = n**2 * segmented_self_inductance(segs, s, a)
    return L


# for some reason, these are slightly slower than the above.
# but they work for any shape r & z as long as they are the same
@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:])"],
    "(p, q),(n),(n)->(n)",
    target="parallel",
)
def BrGreenFil(fil, r, z, gr):
    """Radial magnetic field greens functions from a set of filaments over a set of points.

    Args:
        fil (_type_): filament array (r,z,n)
        r (_type_): radial positions of points
        z (_type_): vertical positions of points
        gr (_type_): greens function values at points
    """
    for j in range(r.shape[0]):
        tmp = 0.0
        for i in range(len(fil)):
            tmp += fil[i, 2] * BrGreen(r[j], z[j], fil[i, 0], fil[i, 1])
        gr[j] = tmp


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:])"],
    "(p, q),(n),(n)->(n)",
    target="parallel",
)
def BzGreenFil(fil, r, z, gr):
    """Vertical magnetic field greens functions from a set of filaments over a set of points.

    Args:
        fil (_type_): filament array (r,z,n)
        r (_type_): radial positions of points
        z (_type_): vertical positions of points
        gr (_type_): greens function values at points
    """
    for j in range(r.shape[0]):
        tmp = 0.0
        for i in range(len(fil)):
            tmp += fil[i, 2] * BzGreen(r[j], z[j], fil[i, 0], fil[i, 1])
        gr[j] = tmp


@guvectorize(
    ["void(float64[:,:], float64[:], float64[:], float64[:])"],
    "(p, q),(n),(n)->(n)",
    target="parallel",
)
def AGreenFil(fil, r, z, gr):
    """Psi greens functions from a set of filaments over a set of points.

    Args:
        fil (_type_): filament array (r,z,n)
        r (_type_): radial positions of points
        z (_type_): vertical positions of points
        gr (_type_): greens function values at points
    """
    for j in range(r.shape[0]):
        tmp = 0.0
        for i in range(len(fil)):
            tmp += fil[i, 2] * AGreen(r[j], z[j], fil[i, 0], fil[i, 1])
        gr[j] = tmp


# cant jit this... meshgrid not allowed
def filament_coil(r, z, dr, dz, nt, nr, nz, rot=0):
    """Create an array of filaments, each with its own radius, height, and amperage.

    r : Major radius of coil center.
    z : Vertical center of coil.
    dr : Radial width of coil.
    dz : Height of coil.
    nt : number of turns in coil
    nr : Number of radial slices
    nz : Number of vertical slices
    rot : Rotation of coil about phi axis

    Returns:    Array of shape (nr*nz) x 3 of R, Z, N for each filament

    """
    rd = np.linspace(-dr * (nr - 1) / nr / 2, dr * (nr - 1) / nr / 2, nr)
    zd = np.linspace(-dz * (nz - 1) / nz / 2, dz * (nz - 1) / nz / 2, nz)
    Rg, Zg = np.meshgrid(rd, zd)

    R = r + Rg * np.cos(rot) - Zg * np.sin(rot)
    Z = z + Rg * np.sin(rot) + Zg * np.cos(rot)

    N = np.full_like(R, float(nt) / (nr * nz))
    return np.dstack([R, Z, N]).reshape(nr * nz, 3)


@njit(parallel=True)
def sum_over_filaments(func, f1, f2):
    """Apply a function and sum over all combinations of two sets of filaments.

    Args:
        func (function): function to apply to each pair of filaments
        f1 (array): first filament array
        f2 (array): second filament array

    Returns:
        float: result of summing over all combinations of func(f1[i], f2[j])
    """
    result = float(0)
    for i in prange(f1.shape[0]):
        for j in range(f2.shape[0]):
            result += func(f1[i], f2[j])
    return result


@njit(parallel=True)
def mutual_inductance_of_filaments(f1, f2):
    """Mutual inductance of two sets of filaments.

    These should not be the same filament array,
    not setup to handle self inductance of filament.

    Args:
        f1 (array): first filament array
        f2 (array): second filament array

    Returns:
        float: Mutual inductance of filament sets in Henries
    """
    M = float(0)
    for i in prange(f1.shape[0]):
        for j in range(f2.shape[0]):
            M += mutual_inductance_fil(f1[i, :], f2[j, :])
    return M


@njit(parallel=True)
def self_inductance_by_filaments(f, conductor="round", a=0.01, dr=0.01, dz=0.01):
    """Self inductance of filament set.

    Args:
        f (array): first filament array
        conductor (str, optional): conductor shape. Defaults to "round".
        a (float, optional): conductor radius. Defaults to 0.01.
        dr (float, optional): conductor radial width. Defaults to 0.01
        dz (float, optional): conductor vertical height. Defaults to 0.01

    Returns:
        float: self inductance of filament set in Henries
    """
    L = float(0)
    for i in prange(f.shape[0]):
        for j in range(f.shape[0]):
            if i != j:
                L += mutual_inductance_fil(f[i, :], f[j, :])
        if conductor == "round":
            L += L_round(f[i, 0], a, f[i, 2])
        elif conductor == "hollow_round":
            L += L_hollow_round(f[i, 0], a, f[i, 2])
        elif conductor == "rect":
            L += L_lyle6(f[i, 0], dr, dz, f[i, 2])
    return L


@njit(parallel=True)
def vertical_force_of_filaments(f1, f2):
    """Vertical force between two sets of filaments.

    These should not be the same filament array,
    not setup to handle self inductance of filament.

    Args:
        f1 (array): first filament array
        f2 (array): second filament array

    Returns:
        float: Vertical force in Newtons/Amp^2
    """
    F = float(0)
    for i in prange(f1.shape[0]):
        for j in range(f2.shape[0]):
            F += vertical_force_fil(f1[i, :], f2[j, :])
    return F


@njit(parallel=True)
def radial_force_of_filaments(f1, f2):
    """Radial force between two sets of filaments.

    These should not be the same filament array,
    not setup to handle self inductance of filament.

    Args:
        f1 (array): first filament array
        f2 (array): second filament array

    Returns:
        float: Radial force in Newtons/Amp^2
    """
    F = float(0)
    for i in prange(f1.shape[0]):
        for j in range(f2.shape[0]):
            F += radial_force_fil(f1[i, :], f2[j, :])
    return F
