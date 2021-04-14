import numpy as np
import scipy.integrate as integrate
from scipy.constants import epsilon_0, mu_0
from scipy.special import jv, jvp, kv, kvp



# eve
def eve_eh(u, w, l, k, a, n_1, n_2):
    """
    Eq. 23
    """
    beta = np.sqrt((k * n_1) ** 2 - u ** 2)
    return (
        eve_lhs(u, l) - eve_rhs1(u, w, l, n_1, n_2)
        - R(u, w, l, k, n_1, n_2, beta)
    )


def eve_he(u, w, l, k, a, n_1, n_2):
    """
    Eq. 23
    """
    beta = np.sqrt((k * n_1) ** 2 - (u / a) ** 2)
    return (
        eve_lhs(u, l) - eve_rhs1(u, w, l, n_1, n_2)
        + R(u, w, l, k, n_1, n_2, beta)
    )


def eve_lhs(u, l):
    return jv(l-1, u) / (u * jv(l, u))


def eve_rhs1(u, w, l, n_1, n_2):
    return (
        (n_1 ** 2 + n_2 **2) / (4 * n_1 ** 2)
        * (kv(l-1, w) + kv(l+1, w)) / (w * kv(l, w))
        + l / u ** 2
    )


def R(u, w, l, k, n_1, n_2, beta):
    return np.sqrt(_R(u, w, l, k, n_1, n_2, beta))


def _R(u, w, l, k, n_1, n_2, beta):
    return (
        (n_1 ** 2 - n_2 ** 2) ** 2 / (4 * n_1 ** 2) ** 2
        * ((kv(l-1, w) + kv(l+1, w)) / (w * kv(l, w))) ** 2
        + ((l * beta) / (n_1 * k)) ** 2 * ((1 / u) ** 2 + (1 / w) ** 2) ** 2
    )



# r < a
def Er_core(r, p, phase, pol, l, a, beta, u, s, A):
    return (
        -1j * A * beta * (a / u)
        * ((1 - s) / 2 * jv(l-1, u * r / a)
        - (1 + s) / 2 * jv(l+1, u * r / a))
        * np.cos(l * p + pol) * np.exp(1j * phase)
    )


def Ep_core(r, p, phase, pol, l, a, beta, u, s, A):
    return (
        1j * A * beta * (a / u)
        * ((1 - s) / 2 * jv(l-1, u * r / a)
        + (1 + s) / 2 * jv(l+1, u * r / a))
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


def Ez_core(r, p, phase, pol, l, a, u, A):
    return (
        A * jv(l, u / a * r) * np.cos(l * p + pol)
        * np.exp(1j * phase)
    )


def Hr_core(r, p, phase, pol, l, omega, a, n_1, u, s_1, A):
    return (
        -1j * A * omega * epsilon_0 * n_1 ** 2 * (a / u)
        * ((1 - s_1) / 2 * jv(l-1, u * r / a)
        + (1 + s_1) / 2 * jv(l+1, u * r / a))
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


def Hp_core(r, p, phase, pol, l, omega, a, n_1, u, s_1, A):
    return (
        -1j * A * omega * epsilon_0 * n_1 ** 2 * (a / u)
        * ((1 - s_1) / 2 * jv(l-1, u * r / a)
        - (1 + s_1) / 2 * jv(l+1, u * r / a))
        * np.cos(l * p + pol) * np.exp(1j * phase)
    )


def Hz_core(r, p, phase, pol, l, omega, a, beta, u, s, A):
    return (
        -A * (beta / (omega * mu_0)) * s * jv(l, u * r / a)
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


# r > a
def Er_clad(r, p, phase, pol, l, a, beta, u, w, s, A):
    return (
        -1j * A * beta * ((a * jv(l, u)) / (w * kv(l, w)))
        * ((1 - s) / 2 * kv(l-1, w * r / a)
        + (1 + s) / 2 * kv(l+1, w * r / a))
        * np.cos(l * p + pol) * np.exp(1j * phase)
    )


def Ep_clad(r, p, phase, pol, l, a, beta, u, w, s, A):
    return (
        1j * A * beta * ((a * jv(l, u)) / (w * kv(l, w)))
        * ((1 - s) / 2 * kv(l-1, w * r / a)
        - (1 + s) / 2 * kv(l+1, w * r / a))
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


def Ez_clad(r, p, phase, pol, l, a, u, w, A):
    return (
        A * jv(l, u) / kv(l, w) * kv(l, w * r / a) * np.cos(l * p + pol)
        * np.exp(1j * phase)
    )


# r > a
def Hr_clad(r, p, phase, pol, l, omega, a, n_2, u, w, s_2, A):
    return (
        -1j * A * omega * epsilon_0 * l_0 ** 2
        * ((a * jv(l, u)) / (w * kv(l, w)))
        * ((1 - s_2) / 2 * kv(l-1, w * r / a)
        - (1 + s_2) / 2 * kv(l+1, w * r / a))
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


def Hp_clad(r, p, phase, pol, l, omega, a, n_2, u, w, s_2, A):
    return (
        -1j * A * omega * epsilon_0 * l_0 ** 2
        * ((a * jv(l, u)) / (w * kv(l, w)))
        * ((1 - s_2) / 2 * kv(l-1, w * r / a)
        + (1 + s_2) / 2 * kv(l+1, w * r / a))
        * np.sin(l * p + pol) * np.exp(1j * phase)
    )


def Hz_clad(r, p, phase, pol, l, omega, a, beta, u, w, s, A):
    return (
        -A * beta / (omega * mu_0) * s * (jv(l, u) / kv(l, w))
        * kv(l, w * r / a) * np.sin(l * p + pol)
        * np.exp(1j * phase)
    )



# power
def calc_P_core(l, omega, a, n_1, beta, u, s, s_1, A):
    return (
        (np.pi / 4) * omega * epsilon_0 * n_1 ** 2 * beta * np.abs(A) ** 2 * (a / u) ** 2
        * ((1 - s) * (1 - s_1)
        * integrate.quad(lambda r: jv(l-1, r) ** 2, 0, a)
        + (1 + s) * (1 + s_1)
        * integrate.quad(lambda r: jv(l+1, r) ** 2, 0, a))
    )


def calc_P_clad(l, omega, a, n_2, beta, u, w, s, s_2, A):
    return (
        (np.pi / 4) * omega * epsilon_0 * n_2 ** 2 * beta * np.abs(A) ** 2
        * ((a * jv(l, u)) / (w * kv(l, w))) ** 2
        * ((1 - s) * (1 - s_2)
        * integrate.quad(lambda r: kv(l-1, r) ** 2, a, np.inf)
        + (1 + s) * (1 + s_2)
        * integrate.quad(lambda r: kv(l+1, r) ** 2, a, np.inf))
    )



# amp
def calc_A(P, l, omega, a, n_1, n_2, beta, u, w, s, s_1, s_2):
    P1 = calc_P_core(l, omega, a, n_1, beta, u, s, s_1, A=1)
    P2 = calc_P_clad(l, omega, a, n_2, beta, u, w, s, s_2, A=1)
    return np.sqrt(P / (P1 + P2))
