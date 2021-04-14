import numpy as np
from scipy.optimize import root_scalar
from scipy.special import jv, jvp, kv, kvp

import opticalfiber.teeq as teeq
import opticalfiber.tmeq as tmeq
import opticalfiber.hyeq as hyeq



# consts
def calc_V(k, a, n_1, n_2):
    return k * a * np.sqrt(n_1 ** 2 - n_2 ** 2)


def calc_beta(k, a, n_1, u):
    return np.sqrt(k ** 2 * n_1 ** 2 - (u / a) ** 2)


def calc_w(V, u):
    return np.sqrt(V ** 2 - u ** 2)


def calc_s(l, u, w):
    return (
        l * (1 / u ** 2 + 1 / w ** 2)
        / (jvp(l, u) / (u * jv(l, u)) + kvp(l, w) / (w * kv(l, w)))
    )


def calc_s_1(k, n_1, beta, s):
    return (beta / (k * n_1)) ** 2 * s


def calc_s_2(k, n_2, beta, s):
    return (beta / (k * n_2)) ** 2 * s



# eve solvers
_window_size = 1.


def solve_eve(eve, V, *eve_args):
    sol = []
    u_arr = np.linspace(0, V, 1000)[1:-1]
    eve_arr = eve(u_arr, calc_w(V, u_arr), *eve_args)
    zc_ind = np.where(eve_arr[0:-1] * eve_arr[1:] <= 0)[0]

    for i in zc_ind:
        if abs(eve_arr[i] - eve_arr[i+1]) > _window_size:
            continue

        if eve_arr[i] == 0.:
            sol.append(u_arr[i])
        elif eve_arr[i+1] == 0.:
            sol.append(u_arr[i+1])
        else:
            sol.append(root_scalar(lambda u: eve(u, calc_w(V, u), *eve_args),
                                   bracket=[u_arr[i], u_arr[i + 1]]).root)

    return sol


def solve_te(m, lam, a, n_1, n_2):
    k = 2 * np.pi / lam
    V = calc_V(k, a, n_1, n_2)
    sol = solve_eve(teeq.eve, V)
    return sol[m-1]


def solve_tm(m, lam, a, n_1, n_2):
    k = 2 * np.pi / lam
    V = calc_V(k, a, n_1, n_2)
    sol = solve_eve(tmeq.eve, V, n_1, n_2)
    return sol[m-1]


def solve_eh(l, m, lam, a, n_1, n_2):
    k = 2 * np.pi / lam
    V = calc_V(k, a, n_1, n_2)
    sol = solve_eve(hyeq.eve_eh, V, l, k, a, n_1, n_2)
    return sol[m-1]


def solve_he(l, m, lam, a, n_1, n_2):
    k = 2 * np.pi / lam
    V = calc_V(k, a, n_1, n_2)
    sol = solve_eve(hyeq.eve_he, V, l, k, a, n_1, n_2)
    return sol[m-1]


def propagatable_modes(lam, a, n_1, n_2):
    modes_dict = {"TE": [], "TM": [], "EH": [], "HE": []} 
    k = 2 * np.pi / lam
    V = calc_V(k, a, n_1, n_2)

    modes = solve_eve(teeq.eve, V)
    for i in range(len(modes)):
        m = i + 1
        modes_dict["TE"].append(f"0{m}")

    modes = solve_eve(tmeq.eve, V, *[n_1, n_2])
    for i in range(len(modes)):
        m = i + 1
        modes_dict["TM"].append(f"0{m}")

    modes = _solve_hybrid_all(hyeq.eve_eh, V, *[k, a, n_1, n_2])
    for i in range(len(modes)):
        l = i + 1
        for j in range(len(modes[i])):
            m = j + 1
            modes_dict["EH"].append(f"{l}{m}")

    modes = _solve_hybrid_all(hyeq.eve_he, V, *[k, a, n_1, n_2])
    for i in range(len(modes)):
        l = i + 1
        for j in range(len(modes[i])):
            m = j + 1
            modes_dict["HE"].append(f"{l}{m}")

    return modes_dict


def _solve_hybrid_all(eve, V, *eve_args):
    sol = []
    l = 1

    while True:
        sol_l = solve_eve(eve, V, l, *eve_args)

        if len(sol_l) == 0:
            return sol

        sol.append(sol_l)
        l += 1

    return sol
