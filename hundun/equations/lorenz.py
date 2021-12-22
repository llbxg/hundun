from ..systems._differential import Differential as _Differential


class Lorenz(_Differential):

    def parameter(self, s=10, r=28, b=8/3):
        self.s, self.r, self.b = s, r, b
        self.dim = 3

    def equation(self, t, u):
        s, r, b = self.s, self.r, self.b

        x, y, z = u

        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z

        return x_dot, y_dot, z_dot

    def jacobian(self):
        s, r, b = self.s, self.r, self.b
        x, y, z = self.u

        j = [[-s, s, 0],
             [r-z, -1, -x],
             [y, x, -b]]

        return j
