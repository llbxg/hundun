from ..systems._difference import Difference as _Difference

class Logistic(_Difference):

    def parameter(self, a=4.0):
        self.a = a
        self.dim = 1

    def equation(self, t, u):
        return self.a * u * (1 - u)

    def jacobian(self):
        j = self.a*(1-2*self.u)
        return j
