# hundun

hundun is a python library for the exploration of chaos.   
Please note that this library is in beta phase.

## Example

Import the package's equation object.

```python
from hundun import Differential
```

The important thing is to define `parameter()` and `equation()` as methods.
Creating a differential equation is easy using `Differential`.

![\begin{array}{l}
    \dot{x}=\sigma (y-x) \\
    \dot{y}=rx - y - xz \\
    \dot{z}=xy - bz
\end{array}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Barray%7D%7Bl%7D%0A++++%5Cdot%7Bx%7D%3D%5Csigma+%28y-x%29+%5C%5C%0A++++%5Cdot%7By%7D%3Drx+-+y+-+xz+%5C%5C%0A++++%5Cdot%7Bz%7D%3Dxy+-+bz%0A%5Cend%7Barray%7D)

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
(This method uses RK4 by default.) 

```python
l = Lorenz.on_attractor()
l.solve_n_times(5000)
```

At this time, you can get the time and orbit by using `.t_seq` and `.u_seq`.


It is possible to calculate the **Lyapunov exponent**(spectrum) from the orbit using `Lorenz` above.
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
```


Also, you can easily draw by using `Drawing` of utils.

```python
from hundun.utils import Drawing

d = Drawing(1, 2, three=1, number=True)

d[0,0].plot(l.u_seq[:,0], l.u_seq[:,1], l.u_seq[:,2])
d[0,0].set_axis_label('x', 'y', 'z')

for i in range(3):
    d[0,1].plot(les_seq[:, i], label=fr'$\lambda_{i+1}=${les[i]:>+8.3f}')
d[0,1].legend(loc='center right')
d[0,1].set_axis_label('step', r'\lambda')

d.show()
```

![fig:lorenz](docs/img/sample_lorenz_les.png)


Currently, time series analysis methods are being added!


# Documentation

## Installation

hundun can be installed via pip from PyPI.

```bash
pip install hundun
```

To use the latest code (unstable), checkout the dev branch and run above command in the main hundun directory.

```python
pip install .
```

## exploration
### Introduction
The following example uses a 1-dim time series (x) obtained from the Lorenz equation. Equation were numerically integrated by Runge-Kutta method with a time with h=0.01 for 5000 time steps.

![fig:embedding](docs/img/sample_lorenz_data.png)

```python
u_seq = np.load('sample/data/lorenz_x.npy')
```


### Embedding (埋め込み)
Generate a time series using the embedding dimension `D` and the time lag `L`.   

```Python
from hundun.exploration import embedding
```

Generate a time series by using `embedding(u_seq, T, D)` and plot the result.

```python
e_seq = embedding(u_seq, 10, 2)

d = Drawing()
d[0,0].plot(e_seq[:, 0], e_seq[:, 1])
d[0,0].set_axis_label('x(t)', 'x(t+T)')
d.show()
```

![fig:embedding](docs/img/sample_lorenz_embedding_2.png)

The result of calculation with D=3 and shifting T is shown below.

![fig:embedding](docs/img/sample_embedding.png)

## Equations
Some equations have already been defined.   

```python
from hundun.equations import Lorenz, Henon, Logistic
```

### Lorenz equation
[[Lorenz_1963]](#Deterministic-Nonperiodic-Flow)   
By default, `s=10, r=28, b=8/3` is set.

![\begin{array}{l}
    \dot{x}=s (y-x) \\
    \dot{y}=rx - y - xz \\
    \dot{z}=xy - bz
\end{array}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Barray%7D%7Bl%7D%0A++++%5Cdot%7Bx%7D%3Ds+%28y-x%29+%5C%5C%0A++++%5Cdot%7By%7D%3Drx+-+y+-+xz+%5C%5C%0A++++%5Cdot%7Bz%7D%3Dxy+-+bz%0A%5Cend%7Barray%7D)

`Lorenz` is a class that inherits from` Differential`.   
This class has a dimensionless time t and a variable u. For Lorenz, u = (x, y, z).   

#### on_attractor
@classmethod   

Calculate from a random initial position [0,1) and place the trajectory on the attractor.    
By default, 5000 steps are calculated.   
By setting `t0` and `u0`, the initial position can be set arbitrarily.   

```python
attractor = Lorenz.on_attractor()
```


The calculation process is as shown in the figure from the blue point to the orange point.

![fig:on_attractor](docs/img/set_on_attractor.png)


#### solve_n_times
Calculate n times.   
After the calculation, you can get the time and orbit by using `.t_seq` and `.u_seq`.

```python
l = Lorenz.on_attractor()
l.solve_n_times(5000)

u_seq, t_seq = l.u_seq, l.t_seq
```


### Henon map
[[Hénon_1976]](#A-two-dimensional-mapping-with-a-strange-attractor)   
By default, `a=1.4, b=0.3` is set. 

![\begin{array}{l}
    x_{t+1}=y_t +1-ax_t^2 \\
    y_{t+1}=bx_t
\end{array}](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+%5Cbegin%7Barray%7D%7Bl%7D%0A++++x_%7Bt%2B1%7D%3Dy_t+%2B1-ax_t%5E2+%5C%5C%0A++++y_%7Bt%2B1%7D%3Dbx_t%0A%5Cend%7Barray%7D)

Since Henon map has a basin of attraction, it is very important to select the initial value. You can check if it is in orbit by using `.inf`.

```python
while True:
    henon = Henon.on_attractor()
    if not henon.inf:
        break
```

![fig:henon](docs/img/sample_henon.png)

### Logistic map
By default, `a=4.0` is set.


[[May_1976]](#Simple-mathematical-models-with-very-complicated-dynamics)   

![x_{t+1}=ax_t(1-x_t)](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+x_%7Bt%2B1%7D%3Dax_t%281-x_t%29)   

![fig:logistic](docs/img/sample_logistic.png)   


## systems
You can create maps and equations by using `Difference` and `Differential` and analyze the created instance.

### Difference / Differential
There is no big difference between the two. 
The difference in `.solve` is whether to map or use runge-kutta.

The important thing is to define parameter() and equation() as methods.

## Dependencies

[{ Numpy }](https://numpy.org)
[{ Scipy }](https://scipy.org)
[{ Matplotlib }](https://matplotlib.org)


## Reference

### Deterministic Nonperiodic Flow
Edward N. Lorenz   
DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2

### A two-dimensional mapping with a strange attractor
M. Hénon   
DOI: 10.1007/BF01608556

### Simple mathematical models with very complicated dynamics
Robert M. May   
DOI: 10.1038/261459a0