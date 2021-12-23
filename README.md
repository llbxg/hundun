# hundun

hundun is a python library for the exploration of chaos.   
Please note that this library is in beta phase.

## Example

Import the package's equation object.

```python
from hundun import Differential
```

Crating a differential equation is easy using `Differential`.
The important thing is to define `parameter()` and `equation()` as methods.

![\begin{array}{l}
    \dot{x}=\sigma (y-x) \\
    \dot{y}=rx - y - xz \\
    \dot{z}=xy - bz
\end{array}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Barray%7D%7Bl%7D%0A++++%5Cdot%7Bx%7D%3D%5Csigma+%28y-x%29+%5C%5C%0A++++%5Cdot%7By%7D%3Drx+-+y+-+xz+%5C%5C%0A++++%5Cdot%7Bz%7D%3Dxy+-+bz%0A%5Cend%7Barray%7D)

```python
class Lorenz(Differential):

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
```

Various methods can be used by creating an instance of `Lorenz`.
As a test, use `.solve_n_times` to solve the equation in 5000 steps.
(This method uses RK4 by default.) Also, you can easily draw by using `Drawing` of utils.

```python
from hundun.utils import Drawing

l = Lorenz.on_attractor()
l.solve_n_times(5000)
u_seq = l.u_seq

d = Drawing(three=True)
d[0,0].plot(u_seq[:, 0], u_seq[:, 1], u_seq[:, 2])
d[0,0].set_axis_label('x', 'y', 'z')
d.show()
```
![fig:lorenz](docs/img/sample_lorenz.png)

It is possible to calculate the Lyapunov exponent(spectrum) from the orbit using `Lorenz` above.
In addition, a calculation method based on QR decomposition can be used by defining `jacobian()`(Jacoby matrix).

```python
class Lorenz2(Lorenz):
    def jacobian(self):
        s, r, b = self.s, self.r, self.b
        x, y, z = self.u

        j = [[-s, s, 0],
             [r-z, -1, -x],
             [y, x, -b]]

        return j
```

`calc_les` automatically determines and calculates.

```python
from hundun.lyapunov import calc_les

les_seq, les = calc_les(Lorenz2)

d = Drawing()
for i in range(3):
    d[0,0].plot(les_seq[:, i], label=fr'$\lambda_{i+1}=${les[i]:>+8.3f}')
d[0,0].legend()
d[0,0].set_axis_label('step', '\lambda')
d.show()
```

![fig:les](docs/img/sample_les.png)

Currently, time series analysis methods are being added!


## Installation

hundun can be installed via pip from PyPI.

```bash
pip install hundun
```

To use the latest code (unstable), checkout the dev branch and run above command in the main hundun directory.

```python
pip install .
```

## Dependencies

[[ Numpy ]](https://numpy.org)
[[ Scipy ]](https://scipy.org)
[[ Matplotlib ]](https://matplotlib.org)