import numpy as np
from scipy.constants import c

import opticalfiber.hyeq as hyeq
import opticalfiber.utils as utl



class HE:
    __slots__ = [
        "_l", "_m", "_lam", "_omega", "_k", "_a", "_n_1", "_n_2", "_beta",
        "_n_eff", "_V", "_u", "_w", "_s", "_s_1", "_s_2", "_A", "_P"
        ]


    def __init__(self, wavelength, radius, n_1, n_2, l=1, m=1, power=None, direction=1):

        lam = wavelength
        a = radius
        k = 2 * np.pi / lam
        V = utl.calc_V(k, a, n_1, n_2)
        u = utl.solve_he(l, m, lam, a, n_1, n_2)
        w = utl.calc_w(V, u)
        beta = utl.calc_beta(k, a, n_1, u)
        s = utl.calc_s(l, u, w)
        A = 1 if power is None else hyeq.calc_A(P, l, omega, a, n_1)

        self._l = l
        self._m = m
        self._lam = wavelength
        self._omega = c * k
        self._k = k * direction
        self._a = radius
        self._n_1 = n_1
        self._n_2 = n_2
        self._beta = beta * direction
        self._n_eff = beta / k
        self._V = V
        self._u = u
        self._w = w
        self._s = s
        self._s_1 = utl.calc_s_1(k, n_1, beta, s)
        self._s_2 = utl.calc_s_2(k, n_2, beta, s)
        self._A = A
        self._P = power


    # r < a
    def Er_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Er_core(r, p, phase, pol, self.l, self.a, self.beta, self.u, self.s, self.A)

    def Ep_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Ep_core(r, p, phase, pol, self.l, self.a, self.beta, self.u, self.s, self.A)

    def Ez_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Ez_core(r, p, phase, pol, self.l, self.a, self.u, self.A)

    def Ex_core(self, r, p, *, phase=0, pol=0):
        return (
            self.Er_core(r, p, phase=phase, pol=pol) * np.cos(p) 
            - self.Ep_core(r, p, phase=phase, pol=pol) * np.sin(p)
        )

    def Ey_core(self, r, p, *, phase=0, pol=0):
        return (
            self.Er_core(r, p, phase=phase, pol=pol) * np.sin(p) 
            + self.Ep_core(r, p, phase=phase, pol=pol) * np.cos(p)
        )

    def Hr_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Hr_core(r, p, phase, pol, self.l, self.omega, self.a, self.n_1, self.u, self.s_1, self.A)

    def Hp_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Hp_core(r, p, phase, pol, self.l, self.omega, self.a, self.n_1, self.u, self.s_1, self.A)

    def Hz_core(self, r, p, *, phase=0, pol=0):
        return hyeq.Hz_core(r, p, phase, pol, self.l, self.omega, self.a, self.u, self.s, self.A)

    def Hx_core(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr_core(r, p, phase=phase , pol=pol) * np.cos(p) 
            - self.Hp_core(r, p, phase=phase , pol=pol) * np.sin(p)
        )

    def Hy_core(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr_core(r, p, phase=phase , pol=pol) * np.sin(p) 
            + self.Hp_core(r, p, phase=phase , pol=pol) * np.cos(p)
        )


    # r > a
    def Er_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Er_clad(r, p, phase, pol, self.l, self.a, self.beta, self.u, self.w, self.s, self.A)

    def Ep_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Ep_clad(r, p, phase, pol, self.l, self.a, self.beta, self.u, self.w, self.s, self.A)


    def Ez_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Ez_clad(r, p, phase, pol, self.l, self.a, self.u, self.w, self.A)

    def Ex_clad(self, r, p, *, phase=0, pol=0):
        return (
            self.Er_clad(r, p, phase=phase, pol=pol) * np.cos(p) 
            - self.Ep_clad(r, p, phase=phase, pol=pol) * np.sin(p)
        )

    def Ey_clad(self, r, p, *, phase=0, pol=0):
        return (
            self.Er_clad(r, p, phase=phase, pol=pol) * np.sin(p) 
            + self.Ep_clad(r, p, phase=phase, pol=pol) * np.cos(p)
        )

    def Hr_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Hr_clad(r, p, phase, pol, self.l, self.omega, self.a, self.n_2, self.u, self.w, self.s_2, self.A)

    def Hp_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Hp_clad(r, p, phase, pol, self.l, self.omega, self.a, self.n_2, self.u, self.w, self.s_2, self.A)

    def Hz_clad(self, r, p, *, phase=0, pol=0):
        return hyeq.Hz_clad(r, p, phase, pol, self.l, self.omega, self.a, self.beta, self.u, self.w, self.A)

    def Hx_clad(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr_clad(r, p, phase=phase, pol=pol) * np.cos(p) 
            - self.Hp_clad(r, p, phase=phase, pol=pol) * np.sin(p)
        )

    def Hy_clad(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr_clad(r, p, phase=phase, pol=pol) * np.sin(p) 
            + self.Hp_clad(r, p, phase=phase, pol=pol) * np.cos(p)
        )

    def Er(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Er_clad(r, p, phase=phase, pol=pol),
            self.Er_core(r, p, phase=phase, pol=pol)
        )
    
    def Ep(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Ep_clad(r, p, phase=phase, pol=pol),
            self.Ep_core(r, p, phase=phase, pol=pol)
        )

    def Ez(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Ez_clad(r, p, phase=phase, pol=pol),
            self.Ez_core(r, p, phase=phase, pol=pol)
        )
    
    def Ex(self, r, p, *, phase=0, pol=0):
        return (
            self.Er(r, p, phase=phase, pol=pol) * np.cos(p)
            - self.Ep(r, p, phase=phase , pol=pol) * np.sin(p)
        )
    
    def Ey(self, r, p, *, phase=0, pol=0):
        return (
            self.Er(r, p, phase=phase , pol=pol) * np.sin(p)
            + self.Ep(r, p, phase=phase , pol=pol) * np.cos(p)
        )
    
    def Hr(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Hr_clad(r, p, phase=phase, pol=pol),
            self.Hr_core(r, p, phase=phase, pol=pol)
        )

    def Hp(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Hp_clad(r, p, phase=phase, pol=pol),
            self.Hp_core(r, p, phase=phase, pol=pol)
        )

    def Hz(self, r, p, *, phase=0, pol=0):
        return np.where(
            r > self.a,
            self.Hz_clad(r, p, phase=phase, pol=pol),
            self.Hz_core(r, p, phase=phase, pol=pol)
        )
    
    def Hx(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr(r, p, phase=phase, pol=pol) * np.cos(p)
            - self.Hp(r, p, phase=phase, pol=pol) * np.sin(p)
        )

    def Hy(self, r, p, *, phase=0, pol=0):
        return (
            self.Hr(r, p, phase=phase, pol=pol) * np.sin(p)
            + self.Hp(r, p, phase=phase, pol=pol) * np.cos(p)
        )


    @property
    def l(self):
        return self._l

    @property
    def m(self):
        return self._m

    @property
    def lam(self):
        return self._lam

    @property
    def wavelength(self):
        return self._lam

    @property
    def omega(self):
        return self._omega

    @property
    def k(self):
        return self._k

    @property
    def a(self):
        return self._a

    @property
    def radius(self):
        return self._a

    @property
    def n_1(self):
        return self._n_1

    @property
    def n_2(self):
        return self._n_2

    @property
    def beta(self):
        return self._beta

    @property
    def propagation_constant(self):
        return self._beta

    @property
    def n_eff(self):
        return self._n_eff

    @property
    def effective_index(self):
        return self._n_eff

    @property
    def V(self):
        return self._V

    @property
    def normalized_frequency(self):
        return self._V

    @property
    def u(self):
        return self._u

    @property
    def w(self):
        return self._w

    @property
    def s(self):
        return self._s

    @property
    def s_1(self):
        return self._s_1

    @property
    def s_2(self):
        return self._s_2

    @property
    def A(self):
        return self._A

    @property
    def P(self):
        return self._P

    @property
    def P_core(self):
        p = hyeq.calc_P_core(
            self.l, self.omega, self.a, self.n_1, self.beta, self.u, self.s,
            self.s_1, self.A
        )
        return p / self._P

    @property
    def P_clad(self):
        p = hyeq.calc_P_clad(
            self.l, self.omega, self.a, self.n_2, self.beta, self.u, self.w, 
            self.s, self.s_2, self.A
        )
        return p / self._P
