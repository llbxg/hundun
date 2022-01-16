# 力学系(dynamical systems)

from abc import ABC as _ABC, abstractmethod as _abstractmethod

import numpy as _np

from ._tu import TU as _TU


class DynamicalSystems(_ABC):

    def __init__(self, t=None, u=None):
        self.dim = 0
        self.parameter()

        if self.dim == 0:
            class_name = self.__class__.__name__
            msg = (f"need to set {class_name}'s dimension "
                   f"({class_name}.dim=? in parameter())")
            raise NotImplementedError(msg)

        self.t = t or 0
        self.u = u if u is not None else _np.zeros(self.dim)

        self._t_seq, self._u_seq = [], []

    @property
    def inf(self):
        return any(_np.isinf(self.u))

    @property
    def internal_state(self):
        return _TU(self.t, self.u)

    @property
    def t_seq(self):
        return _np.array(self._t_seq)

    @property
    def u_seq(self):
        return _np.array(self._u_seq)

    @classmethod
    def on_attractor(cls, t0=None, u0=None, h=0.01, *, T_0=5000, **params):
        c = cls(t0, u0)
        c.parameter(**params)
        c.settle_on_attractor(t0, u0, h=h, T_0=T_0)
        return c

    @_abstractmethod
    def equation(self, t, u):
        """equation"""

    def j(self, **params):
        return _np.array(self.jacobian(**params))

    def jacobian(self):
        """jacobian"""
        return None

    def make_inital(self):
        return _np.random.rand(self.dim)

    def parameter(self):
        """set parameter for equation"""

    def reset_u_seq(self):
        self._u_seq = []

    def settle_on_attractor(self, t0=None, u0=None,
                            *, T_0=5000, notsave=True, **params):
        self.u = self.make_inital() if u0 is None else u0
        self.t = t0 or 0

        for _ in range(T_0):
            self.solve(*self.internal_state, **params)

        if notsave:
            self._u_seq, self._t_seq = [], []

        if t0 is None:
            self.t = 0

        self.t, self.u = self.internal_state

        return self.internal_state

    def solve(self, *args, **kwargs):
        tu = self._solve(*args, **kwargs)

        if kwargs.get('save', True):
            self._u_seq.append(tu.u)
            self._t_seq.append(tu.t)

        return tu

    def solve_n_times(self, n):
        for _ in range(n):
            self.solve(*self.internal_state)
        return self.t_seq, self.u_seq

    def __call__(self, t, u):
        return _np.array(self.equation(t, u))

    def __repr__(self):
        v = vars(self)

        p = ', '.join(f'{key}={_np.round(v[key], 3)}'
                      for key in v.keys()
                      if ('_' not in key) and (key not in ['t', 'u']))

        name = self.__class__.__name__
        return f'{name}({p})'

    @_abstractmethod
    def _solve(self, t, u):
        self.t, self.u = t, u
        return _TU(self.t, self.u)
