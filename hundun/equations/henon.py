from ..systems._difference import Difference as _Difference

class Henon(_Difference):

    def parameter(self, a=1.4, b=0.3):
        self.a, self.b = a, b
        self.dim = 2

    def equation(self, t, u):
        x, y = u

        x_next = 1-self.a*x**2 + y
        y_next = self.b*x

        return x_next, y_next

    def jacobian(self):
        x, _ = self.u

        j = [[-self.a*2*x, 1],
             [self.b,      0]]

        return j

