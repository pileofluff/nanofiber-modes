"""
definitions, equations and formulae for TM modes
"""

import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import j0, j1, jv, k0, k1, kv


# cylindrical coordinate (rho, phi, z)
# cartesian coordinate (x, y, z)
# subscripts 1 and 2 sygnify in/of the core and the cladding respectively



# eigenvalue equation
def eve_lhs(u, w):
    return j1(u) / (u * j0(u))


def eve_rhs(u, w, n_1, n_2):
    return -(n_2 / n_1) ** 2 * (k1(w) / (w * k0(w)))


def eve(u, w, n_1, n_2):
    return eve_lhs(u, w) - eve_rhs(u, w, n_1, n_2)



# r < a
def Er_core(r, phase, beta, a, u, A):
    return 1j * beta * (a / u) * A * j1(u * r / a) * np.exp(1j * phase)


def Ez_core(r, phase, a, u, A):
    return A * j0(u * r / a) * np.exp(1j * phase)


def Hp_core(r, phase, omega, a, n_1, u, A):
    return 1j * omega * epsilon_0 * n_1 ** 2 * (a / u) * A * j1(u * r / a) * np.exp(1j * phase)


# r < a
def Er_clad(r, phase, beta, a, u, w, A):
    return -1j * beta * (a / w) * (j0(u) / k0(w)) * A * k1(w * r / a) * np.exp(1j * phase)


def Ez_clad(r, phase, a, u, w, A):
    return (j0(u) / k0(w)) * A * k0(w * r / a) * np.exp(1j * phase)


def Hp_clad(r, phase, omega, a, n_2, u, w, A):
    return (
        -1j * omega * epsilon_0 * n_2 ** 2 * (a / w) * (j0(u) / k0(w)) * A
        * k1(w * r / a) * np.exp(1j * phase)
    )



# power
def calc_P_core(omega, a, n_1, beta, u, w, A):
    return (
        (np.pi / 2) * omega * epsilon_0 * n_1 ** 2 * beta * np.abs(A) ** 2
        * (a ** 4 / u ** 2) * (j1(u) ** 2 - j0(u) * jv(2, u))
    )


def calc_P_clad(omega, a, n_2, beta, u, w, A):
    return (
        (np.pi / 2) * omega * epsilon_0 * n_2 ** 2 * beta * np.abs(A) ** 2
        * ((a ** 4 * j0(u) ** 2) / (w ** 2 * k0(w) ** 2))
        * (k0(w) * kv(2, w) - k1(w) ** 2)
    )



# amp
def calc_A(P, omega, a, n_1, n_2, beta, u, w):
    P1 = calc_P_core(omega, a, n_1, beta, u, w, A=1)
    P2 = calc_P_clad(omega, a, n_2, beta, u, w, A=1)
    return np.sqrt(P / (P1 + P2))
