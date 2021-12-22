from ._systems import DynamicalSystems as _DynamicalSystems
from ._tu import TU as _TU


class Difference(_DynamicalSystems):

    def _solve(self, t, u, *, h=1, **params):
        self.u = self(t, u, **params)
        self.t = t+h

        return _TU(self.t, self.u)
