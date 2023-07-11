"""See if we can derive an expression for the gmd of a triangle."""
import numpy as np

from inductance import _numba as nb


@nb.njit
def normalized_triangle_pts(a, b):
    """Return a triangle with normalized sides a and b.

    The triangle is normalized so that the half perimeter is 1.
    This means all shapes are represented by the two sides a and b
    which can have a range of values from 0 to 1.  This will
    capture all possible triangles that have the same shape. Of course,
    one must allow for change is scale, location, and orientation, and
    reflection.

    Will return points in a counter-clockwise order only. Starting at origin.
    and the side a along the x-axis.

    This equation can be found by setting a + b + c = 2 and using the law of
    cosines to solve for the cosine of the angle opposite side c, gamma.

                                    + (a-mx-b*cos(g), b*sin(g)-my)
                                  .  .
                                .     .
                           c  .        .  b
                            .           .
                          .            g .
             (-mx, -my) + - - - - - - - - + (a-mx, -my)
                                a

    Args:
        a (float): length of side a
        b (float): length of side b

    Returns:
        tuple(tuple(float)): points of triangle in counter-clockwise order
    """
    pts = np.zeros((6), dtype=np.float64)
    cos_g = (2 * (a + b) - (2 + a * b)) / (a * b)
    sin_g = np.sqrt(1 - cos_g**2)

    mx = (2 * a - b * cos_g) / 3
    my = b * sin_g / 3

    pts[:] = [-mx, -my, a - mx, -my, a - mx - b * cos_g, b * sin_g - my]
    return pts


def normalize_triangle(pts):
    """Convert from points to normalized sides."""
    a = np.sqrt((pts[0] - pts[2]) ** 2 + (pts[1] - pts[3]) ** 2)
    b = np.sqrt((pts[2] - pts[4]) ** 2 + (pts[3] - pts[5]) ** 2)
    c = np.sqrt((pts[4] - pts[0]) ** 2 + (pts[5] - pts[1]) ** 2)
    s = (a + b + c) / 2
    return np.array([a / s, b / s])


def _point_in_triangle(pt, tri):
    """Return True if pt is in tri, False otherwise.

    Assumes tri is a triangle in counter-clockwise order.
    Assumes the first side is along the x-axis.
    """
    if pt[0] < 0 or pt[1] < 0:
        return False

    result = (pt[1] - tri[3]) * (tri[4] - tri[2]) - (pt[0] - tri[2]) * (tri[5] - tri[3])
    if result < 0.0:
        return False
    result = (pt[1] - tri[5]) * (tri[0] - tri[4]) - (pt[0] - tri[4]) * (tri[1] - tri[5])
    return result >= 0.0
