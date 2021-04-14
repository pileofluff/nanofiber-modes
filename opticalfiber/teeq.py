"""
definitions, equations and formulae for TE modes
"""

import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.special import j0, j1, k0, k1, kv


# cylindrical coordinate (rho, phi, z)
# cartesian coordinate (x, y, z)
# subscripts 1 and 2 sygnify in/of the core and the cladding respectively



# eigenvalue equation
def eve_lhs(u, w):
    return j1(u) / (u * j0(u))


def eve_rhs(u, w):
    return -k1(w) / (w * k0(w))


def eve(u, w):
    return eve_lhs(u, w) - eve_rhs(u, w)



# r < a
def Ep_core(r, phase, omega, a, u, A):
    return -1j * omega * mu_0 * (a / u) * A * j1(u * r / a) * np.exp(1j * phase)


def Hr_core(r, phase, beta, a, u, A):
    return 1j * beta * (a / u) * A * j1(u * r / a) * np.exp(1j * phase)


def Hz_core(r, phase, a, u, A):
    return A * j0(u * r / a) * np.exp(1j * phase)


# r > a
def Ep_clad(r, phase, omega, a, u, w, A):
    return 1j * omega * mu_0 * (a / w) * (j0(u) / k0(w)) * A * k1(w * r / a) * np.exp(1j * phase)


def Hr_clad(r, phase, beta, a, u, w, A):
    return -1j * beta * (a / w) * (j0(u) / k0(w)) * A * k1(w * r / a) * np.exp(1j * phase)


def Hz_clad(r, phase, a, u, w, A):
    return (j0(u) / k0(w)) * A * k0(w * r / a) * np.exp(1j * phase)



# power
def calc_P_core(omega, a, beta, u, w, A):
    return (
        (np.pi / 2) * omega * mu_0 * beta * np.abs(A) ** 2
        * (a ** 4 / u ** 2) * j1(u) ** 2
        * (1 + (w ** 2 / u ** 2) * ((k0(w) * kv(2, w)) / k1(w) ** 2))
    )


def calc_P_clad(omega, a, beta, u, w, A):
    return (
        (np.pi / 2) * omega * mu_0 * beta * np.abs(A) ** 2
        * (a ** 4 / u ** 2) * j1(u) ** 2
        * (((k0(w) * kv(2, w)) / k1(w) ** 2) - 1)
    )



# amp
def calc_A(P, omega, a, beta, u, w):
    P1 = calc_P_core(omega, a, beta, u, w, A=1)
    P2 = calc_P_clad(omega, a, beta, u, w, A=1)
    return np.sqrt(P / (P1 + P2))
