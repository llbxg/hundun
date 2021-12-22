import numpy as _np

from ._systems import DynamicalSystems as _DynamicalSystems
from ._solver import RK4 as _RK4
from ._tu import TU as _TU


class Differential(_DynamicalSystems):

    def _solve(self, t, u, *, h=0.01, solver=_RK4, f=None, **params):
        sol = solver(f or self, h=h, **params)
        self.u = sol(t, _np.array(u))
        self.t = t+sol.h

        return _TU(self.t, self.u)
