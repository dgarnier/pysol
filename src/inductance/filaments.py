"""Filamentary inductance calculations."""

import numpy as np

from ._numba import guvectorize, njit, prange
from .elliptics import ellipke
from .self import L_hollow_round, L_lyle6, L_round, _lyle_terms


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


def M_filsol_path(fil, pts, nt, ds=0):
    """Mutual inductance between a set of axisymmetric filaments and a path from pts."""
    segs, _ = segment_path(pts, ds)
    return nt * mutual_filaments_segmented(fil, segs)


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
