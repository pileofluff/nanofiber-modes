import numpy as np
from scipy.constants import c

import opticalfiber.teeq as teeq
import opticalfiber.utils as utl



class TE:
    __slots__ = ["_m", "_lam", "_omega", "_k", "_a", "_n_1", "_n_2", "_beta",
                 "_n_eff", "_V", "_u", "_w", "_A", "_P"]


    def __init__(self, m, wavelength, radius, n_1, n_2, power=None, direction=1):
        lam = wavelength
        a = radius
        k = 2 * np.pi / lam
        V = utl.calc_V(k, a, n_1, n_2)
        u = utl.solve_te(m, lam, a, n_1, n_2)
        w = utl.calc_w(V, u)
        beta = utl.calc_beta(k, a, n_1, u)
        A = 1 if power is None else teeq.calc_A(P, omega, a, beta, u, w)

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
        self._A = A
        self._P = power


    @classmethod
    def lowest_order(cls, wavelength, radius, n_1, n_2, power=None, direction=1):
        return cls(1, wavelength, radius, n_1, n_2, power, direction)


    # r < a
    def Er_core(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Ep_core(self, r, *, phase=0):
        return teeq.Ep_core(r, phase, self.omega, self.a, self.u, self.A)

    def Ez_core(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Ex_core(self, r, p, *, phase=0):
        return -self.Ep_core(r, phase=phase) * np.sin(p)

    def Ey_core(self, r, p, *, phase=0):
        return self.Ep_core(r, phase=phase) * np.cos(p)

    def Hr_core(self, r, *, phase=0):
        return teeq.Hr_core(r, phase, self.beta, self.a, self.u, self.A)

    def Hp_core(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Hz_core(self, r, *, phase=0):
        return teeq.Hz_core(r, phase, self.a, self.u, self.A)

    def Hx_core(self, r, p, *, phase=0):
        return self.Hr_core(r, phase=phase) * np.cos(p) 

    def Hy_core(self, r, p, *, phase=0):
        return self.Hr_core(r, phase=phase) * np.sin(p) 

    # r > a
    def Er_clad(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Ep_clad(self, r, *, phase=0):
        return teeq.Ep_clad(r, phase, self.omega, self.a, self.u, self.w, self.A)

    def Ez_clad(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Ex_clad(self, r, p, *, phase=0):
        return -self.Ep_clad(r, phase=phase) * np.sin(p)

    def Ey_clad(self, r, p, *, phase=0):
        return self.Ep_clad(r, phase=phase) * np.cos(p)

    def Hr_clad(self, r, *, phase=0):
        return teeq.Hr_clad(r, phase, self.beta, self.a, self.u, self.w, self.A)

    def Hp_clad(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Hz_clad(self, r, *, phase=0):
        return teeq.Hz_clad(r, phase, self.a, self.u, self.w, self.A)

    def Hx_clad(self, r, p, *, phase=0):
        return self.Hr_clad(r, phase=phase) * np.cos(p) 

    def Hy_clad(self, r, p, *, phase=0):
        return self.Hr_clad(r, phase=phase) * np.sin(p) 


    def Er(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Ep(self, r, *, phase=0):
        return np.where(r > self.a, self.Ep_clad(r, phase=phase), self.Ep_core(r, phase=phase))
    
    def Ez(self, r, *, phase=0):
        return np.zeros(r.shape)
    
    def Ex(self, r, p, *, phase=0):
        return -self.Ep(r, phase=phase) * np.sin(p)

    def Ey(self, r, p, *, phase=0):
        return self.Ep(r, phase=phase) * np.cos(p)

    def Hr(self, r, *, phase=0):
        return np.where(r > self.a, self.Hr_clad(r, phase=phase), self.Hr_core(r, phase=phase))

    def Hp(self, r, *, phase=0):
        return np.zeros(r.shape)

    def Hz(self, r, *, phase=0):
        return np.where(r > self.a, self.Hz_clad(r, phase=phase), self.Hz_core(r, phase=phase))
    
    def Hx(self, r, p, *, phase=0):
        return self.Hr(r, phase=phase) * np.cos(p)

    def Hy(self, r, p, *, phase=0):
        return self.Hr(r, phase=phase) * np.sin(p)


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
    def A(self):
        return self._A

    @property
    def P(self):
        return self._P

    @property
    def P_core(self):
        p = teeq.calc_P_core(self.omega, self.a, self.beta, self.u, self.w, self.A)
        return p / self._P

    @property
    def P_clad(self):
        p = teeq.calc_P_clad(self.omega, self.a, self.beta, self.u, self.w, self.A)
        return p / self._P
