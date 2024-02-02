import numpy as np


def m1_m2_from_M_q(M, q):
    """Compute individual masses from total mass and mass ratio.

    Choose m1 >= m2.

    Arguments:
        M {float} -- total mass
        q {mass ratio} -- mass ratio, 0.0< q <= 1.0

    Returns:
        (float, float) -- (mass_1, mass_2)
    """

    m1 = M / (1.0 + q)
    m2 = q * m1

    return m1, m2


def m1_m2_from_M_Chirp_q(M_Chirp, q):
    q = 1 / q
    eta = q / (1 + q) ** 2
    M = M_Chirp * eta ** (-3 / 5)
    return m1_m2_from_M_q(M, 1 / q)


def AET(X, Y, Z):
    return [
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    ]


def XYZ(A, E, T):
    return [
        (-A / np.sqrt(2.0) + T / np.sqrt(2.0)),
        (A / np.sqrt(6.0) - 2 * E / np.sqrt(6.0) + T / np.sqrt(6.0)),
        (A / np.sqrt(3.0) + E / np.sqrt(3.0) + T / np.sqrt(3.0)),
    ]
