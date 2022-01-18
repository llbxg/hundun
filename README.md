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

![fig:lorenz](https://github.com/llbxg/hundun/blob/main/docs/img/sample_lorenz_les.png?raw=true)


Currently, time series analysis methods are being added!


# Documentation

- [Installation](#Installation)   
- [Exploration](#Exploration)   
    - [Embedding](#Embedding-埋め込み)   
    - [Estimate the time lag](#Estimate-the-time-lag)   
    - [Estimate the generalized dimension](#Estimate-the-generalized-dimension)   
    - [Estimate the acceptable minimum embedding dimension](#Estimate-the-acceptable-minimum-embedding-dimension)   
    - [Visualization](#Visualization)   
- [Equations](#Equations)   
    - [Lorenz equation](#Lorenz-equation)   
    - [Henon map](#Henon-map)   
    - [Logistic map](#Logistic-map)   
- [Dynamical systems](#Dynamical-Systems)   
    - [Difference / Differential](#Difference--Differential)   
- [Lyapunov exponents](#Lyapunov-exponents)   
- [Roadmap](#Roadmap)   
- [Dependencies](#Dependencies)   
- [Reference](#Reference)   



## Installation

hundun can be installed via pip from PyPI.

```bash
pip install hundun
```

To use the latest code (unstable), checkout the dev branch and run above command in the main hundun directory.

```python
pip install .
```

## Exploration
### Introduction
The following example uses a 1-dim time series (x) obtained from the Lorenz equation. Equation were numerically integrated by Runge-Kutta method with a time with h=0.01 for 5000 time steps.

![fig:embedding](https://github.com/llbxg/hundun/blob/main/docs/img/sample_lorenz_data.png?raw=true)

```python
u_seq = np.load('sample/data/lorenz_x.npy')
```


### Embedding (埋め込み)
Generate a time series using the embedding dimension `D` and the time lag `L`.   

```Python
from hundun import embedding
```

Generate a time series by using `embedding(u_seq, T, D)` and plot the result.

```python
e_seq = embedding(u_seq, 10, 2)

d = Drawing()
d[0,0].plot(e_seq[:, 0], e_seq[:, 1])
d[0,0].set_axis_label('x(t)', 'x(t+T)')
d.show()
```

![fig:embedding](https://github.com/llbxg/hundun/blob/main/docs/img/sample_lorenz_embedding_2.png?raw=true)

The result of calculation with D=3 and shifting T is shown below.

![fig:embedding](https://github.com/llbxg/hundun/blob/main/docs/img/sample_embedding.png?raw=true)

### Estimate the time lag

#### Autocorrelation Function
```python
def acf(u_seq, tau):
```
Calculate the autocorrelation function from the time series data. 
Finds a point where the autocorrelation function can be considered 0.

In the example below, the time lag is determined based on Bartlett's formula.
Other conditions include the first local minimums and 1/e or less.

```python
from hundun.exploration import acf, get_minidx_below_seq, bartlett
from hundun.utils import Drawing
import numpy as np


u_seq = np.load('lorenz_x.npy')

rho_seq = acf(u_seq, 400)
var_seq = bartlett(rho_seq)
idx_list = get_minidx_below_seq(rho_seq, var_seq)


d = Drawing()
for i, idx in enumerate(idx_list):
    d[0,0].plot(rho_seq[:, i], label='acf')
    d[0,0].plot(var_seq[:, i], label='standard error')
    d[0,0].scatter(idx, rho_seq[idx, i], zorder=10, marker='o')
d[0,0].legend()
d[0,0].axhline(0, color="black", linewidth=0.5, linestyle='dashed')
d[0,0].set_axis_label('Time~lag', 'Correlation~coefficient')
d.show()
```

![fig:acf](https://github.com/llbxg/hundun/blob/main/docs/img/sample_acf.png?raw=true)


#### Mutual Information
```python
def mutual_info(u_seq, tau):
```
Create a histogram from time series data and calculate mutual information.

```python
from hundun.exploration import mutual_info, get_bottom
from hundun.utils import Drawing
import numpy as np


u_seq = np.load('lorenz_x.npy')

mi_seq = mutual_info(u_seq, 300)
bottom = get_bottom(mi_seq)

d = Drawing()
for i, idx in enumerate(bottom):
    d[0,0].plot(mi_seq[:, i])
    d[0,0].scatter(idx, mi_seq[idx, i])
d[0,0].set_axis_label('Time~lag', 'Mutual~Information~[bit]')
d.show()
```

![fig:mi](https://github.com/llbxg/hundun/blob/main/docs/img/sample_mi.png?raw=true)

### Estimate the generalized dimension

#### Grassberger-Procaccia Algorithm
[[Grassberger_1983]](#Measuring-the-strangeness-of-strange-attractors) [[Grassberger_1983_2]](#Characterization-of-Strange-Attractors)   

```python
def calc_correlation_dimention_w_gp(
    e_seq, base=8, h_r=0.05, loop=200, batch_ave=10, normalize=True):
```

The correlation dimension(D2) is obtained by calculating the correlation integral C(r).   

As an example, The result when fixed at T=1 is shown below.

```python
import numpy as np

from hundun.utils import Drawing
from hundun import embedding
from hundun.exploration import calc_correlation_dimention_w_gp

u_seq = np.load('lorenz_x.npy')

d= Drawing(1, 2)

D_min, D_max = 1, 9
D2s = []
for i in range(D_min, D_max+1):
    e_seq = embedding(u_seq, 1, i)
    D2, rs, crs = calc_correlation_dimention_w_gp(e_seq)
    d[0,0].plot(np.log(rs), np.log(crs), label=f'${i}$: {D2:.3f}')
    D2s.append(D2)

d[0,0].legend()
d[0,0].set_axis_label('\log ~r', '\log ~C(r)')

d[0,1].plot(range(D_min, D_max+1), D2s)
d[0,1].plot([1, D_max], [1, D_max],
            color='black', linewidth=0.5, linestyle='dashed')
d[0,1].set_aspect('equal')
d[0,1].set_axis_label('Embedding ~dimension', 'Correlation ~dimension')

d.show()
```

![fig:gp](https://github.com/llbxg/hundun/blob/main/docs/img/sample_calc_D_gp.png?raw=true)

In the GP-method, D2 is calculated directly from the attractor.
It cannot be evaluated accurately from 1-dim data.
When calculating with 3-dim data, it can be calculated with some accuracy.

```python
l = Lorenz.on_attractor()
l.solve_n_times(5000)
u_seq = l.u_seq
```
![fig:gp_3dim](https://github.com/llbxg/hundun/blob/main/docs/img/sample_calc_D_gp_3dim.png?raw=true)

#### Lyapunov dimension
Calculate using `calc_lyapunov_dimension`.   
See here -> [[Lyapunov dimension]](#Lyapunov-dimension-1)

### Estimate the acceptable minimum embedding dimension

#### False Nearest Neighbors - Algorithm

[[Kennel_1992]](#Determining-embedding-dimension-for-phase-space-reconstruction-using-a-geometrical-construction)

```python
def fnn(u_seq, threshold_R=10, threshold_A=2, T=50, D_max=10):
```

```python
import numpy as np
from hundun.utils import Drawing
from hundun.exploration import fnn

u_seq = np.load('lorenz_x.npy')

percentage_list = fnn(u_seq)

d = Drawing()
d[0,0].plot(range(1, len(percentage_list)+1), percentage_list*100,
            marker='.', markersize=10)
d[0,0].axhline(1, color="black", linewidth=0.5)
d[0,0].set_axis_label('Dimension', 'False~NN~Percentage')
d.show()
```

![fig:fnn](https://github.com/llbxg/hundun/blob/main/docs/img/sample_fnn.png?raw=true)

#### Averaged False Neighbors - Algorithm

[[Cao_1997]](#Practical-method-for-determining-the-minimum-embedding-dimension-of-a-scalar-time-series)

```python
def afn(u_seq, T=1, D_max=10):
```

```python
from itertools import cycle

from hundun.utils import Drawing
from hundun.exploration import afn
import matplotlib as mpl
import numpy as np

color = cycle(mpl.rcParams['axes.prop_cycle'])


u_seq = np.load('lorenz_x.npy')
line, marker = {'E1':'solid', 'E2':'dashed'}, {'E1':'o', 'E2':'s'}

d = Drawing()
for T in [1, 5, 10]:
    Es = afn(u_seq, T=T)
    c = next(color)['color']
    for label, E in zip(['E1', 'E2'], Es):
        d[0,0].plot(range(1, len(E)+1), E,
                    marker=marker[label], markersize=5,
                    label=f'{label}-{T}', linestyle=line[label], color=c)

d[0,0].axhline(1, color="black", linewidth=0.5)

d[0,0].set_axis_label('Dimension', 'E1~&~E2')
d[0,0].legend(loc='lower right')
d.show()
```

![fig:afn](https://github.com/llbxg/hundun/blob/main/docs/img/sample_afn.png?raw=true)

#### Wayland Test
[[Wayland_1993]](#Recognizing-determinism-in-a-time-series)

```python
from hundun.exploration import wayland
from hundun.utils import Drawing
import numpy as np

u_seq = np.load('lorenz_x.npy')

median_e_trans_ave = wayland(u_seq)

d = Drawing()
d[0,0].plot(range(1, len(median_e_trans_ave)+1), median_e_trans_ave)
d[0,0].scatter(median_e_trans_ave.argmin()+1, median_e_trans_ave.min())
d[0,0].set_axis_label('Dimension', 'median(E_{trans})')
d[0,0].set_yscale('log')
d.show()
```

![fig:wayland](https://github.com/llbxg/hundun/blob/main/docs/img/sample_wayland.png?raw=true)

### Visualization
#### Recurrence Plot
```python
def calc_recurrence_plot(u_seq, rule=simple_threshold, *params, **kwargs):

def show_recurrence_plot(u_seq, rule=simple_threshold, cmap=False, *params, **kwargs):
```

Create Recurrence Plot(RP) from time series data. 
`calc_~` returns the result of RP as a matrix.
`show_~` displays the result.


```python
from hundun.exploration import show_recurrence_plot
import numpy as np

u_seq = np.load('lorenz_x.npy')

show_recurrence_plot(u_seq)
```

![fig:rp1](https://github.com/llbxg/hundun/blob/main/docs/img/sample_rp.png?raw=true)

The drawing rule uses the simplest one(`simple_threshold`). 

```python
def simple_threshold(ds, theta=0.5):
    if (d_max:=_np.max(ds))!=0:
        pv = (ds/d_max>theta)*255
        return pv
    return ds
```

This can be changed. The `rule` just takes a matrix and returns the matrix.


```python
from hundun.exploration import show_recurrence_plot
import numpy as np


def new_threshold(ds, func):
    if (d_max:=np.max(ds))!=0:
        pv = func(ds/d_max)*255
        return np.uint8(pv)
    return ds


u_seq = np.load('lorenz_x.npy')

show_recurrence_plot(u_seq, cmap=True, rule=new_threshold, func=np.log)
```


![fig:rp2](https://github.com/llbxg/hundun/blob/main/docs/img/sample_rp_2.png?raw=true)


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
```python
@classmethod
def on_attractor(cls, t0=None, u0=None, h=0.01, *, T_0=5000, **params):
```

Calculate from a random initial position [0,1) and place the trajectory on the attractor.    
By default, 5000 steps are calculated.   
By setting `t0` and `u0`, the initial position can be set arbitrarily.   

```python
attractor = Lorenz.on_attractor()
```


The calculation process is as shown in the figure from the blue point to the orange point.

![fig:on_attractor](https://github.com/llbxg/hundun/blob/main/docs/img/set_on_attractor.png?raw=true)


#### solve_n_times
```python
def solve_n_times(self, n):
```

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

![fig:henon](https://github.com/llbxg/hundun/blob/main/docs/img/sample_henon.png?raw=true)

### Logistic map
By default, `a=4.0` is set.


[[May_1976]](#Simple-mathematical-models-with-very-complicated-dynamics)   

![x_{t+1}=ax_t(1-x_t)](https://render.githubusercontent.com/render/math?math=%5CLarge+%5Cdisplaystyle+x_%7Bt%2B1%7D%3Dax_t%281-x_t%29)   

![fig:logistic](https://github.com/llbxg/hundun/blob/main/docs/img/sample_logistic.png?raw=true)   


## Dynamical Systems
You can create maps and equations by using `Difference` and `Differential` and analyze the created instance.

### Difference / Differential
```python
from hundun import Difference, Differential
```

There is no big difference between the two. 
The difference in `.solve` is whether to map or use runge-kutta.

The important thing is to define `parameter()` and `equation()` as methods.

#### parameter()
You must always define dimention(`self.dim`). (It is used to set the initial value and calculate the Lyapunov exponents.)   
(1) If set `self.xxx`, can be used in `equation()`.

#### equation()
Two arguments, time(`t`) and variable(`u`), are required.   

(2) Since `u` is `self.dim`-dimensional variable, its value can be got using the unpack syntax.   
(3) The return value does not have to be a vector.


#### example
Here is an example of defining Lorenz. This time, set the parameter `self.s` in `parameter()`. 
Refer to [[Lorenz equation]](#Lorenz-equation) for how to use it.

```python
class Lorenz(Differential):

    def parameter(self, s=10):
        self.dim = 3
        self.s = s  #(1)

    def equation(self, t, u):
        r, b = 28, 8/3
        s = self.s  #(1)

        x, y, z = u  #(2)

        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z

        return x_dot, y_dot, z_dot
        # (3) OK:
        #      np.array([x_dot, y_dot, z_dot])
        #      [x_dot, y_dot, z_dot]
```

## Lyapunov exponents
Calculate the Lyapunov exponents (Lyapunov spectrum).

```python
from hundun import calc_les
```

### calc_les
```python
def calc_les(system, **options):
```

Specify a `Difference` or `Differential` object for system. 

#### Difference
Jacobin is always required because it is calculated on the QR base. 
(In the case of 1-dim, a different method is used instead of QR-based to speed up the calculation.)

As an example, search for parameters of Henon. It is possible to estimate the range in which the LEs is positive.

```python
from itertools import product

from hundun.equations.henon import Henon
from matplotlib.colors import Normalize
import numpy as np

from hundun import calc_les
from hundun.utils import Drawing


N_a, N_b = 50, 50
a_list = np.linspace(0, 2.1, N_a)
b_list = np.linspace(0, 1.1, N_b)

les_list = []
for a, b in product(a_list, b_list):
    for _ in range(10):
        try:
            _, les = calc_les(Henon, b=b, a=a)
            les_list.append(les)
            break
        except ValueError:
            pass
    else:
        les_list.append((None, None))
les = np.array(les_list).reshape(N_b, N_a, 2)

d=Drawing(1, 2)
for i in range(2):
    le = les[:,:,i]
    sf = d[0,i].contourf(*np.meshgrid(a_list, b_list), le, cmap='jet',
                         norm=Normalize(vmin=-2, vmax=1))
    cb = d.fig.colorbar(sf, ax=d[0,i], orientation='horizontal')
    d[0,i].set_axis_label('a', 'b')
    d[0,i].set_title(f'$\lambda_{i+1}$')
d.show()
```
![fig:henon_les](https://github.com/llbxg/hundun/blob/main/docs/img/sample_henon_les.png?raw=true)

As an example, calculate the LE for parameter A of Logistic map.

```python
import math

from hundun import calc_les
from hundun.equations import Logistic
from hundun.utils import Drawing


L, dh = 400+1, 0.01

d = Drawing()

le_list = []
for i in range(L):
    a = i*dh
    _, le = calc_les(Logistic, N=500, a=a)
    le_list.append(le)
d[0,0].plot([dh*i for i in range(L)], le_list)

for y in [0, math.log(2)]:
    d[0,0].axhline(y, color="black", linewidth=0.5)

d[0,0].set_axis_label('a', '\lambda')
d[0,0].set_ylim(-5, 1)
d.show()
```

![fig:lm_le](https://github.com/llbxg/hundun/blob/main/docs/img/sample_logistic_le_bif.png?raw=true)


#### Differential

As an example, compare with and without Jacobian Matrix(`jacobian`). 

```python
from hundun import calc_les
from hundun.equations import Lorenz
from hundun.utils import Drawing


class Lorenz_No_Jacobian(Lorenz):
    def jacobian(self):
        return None


u0 = Lorenz.on_attractor().u

d = Drawing(1, 2)
for j, system in enumerate([Lorenz, Lorenz_No_Jacobian]):
    les_seq, les = calc_les(system, u0=u0)
    for i, le in enumerate(les):
        p, = d[0, j].plot(les_seq[:, i],
                          label=fr'$\lambda_{i+1}=$ {le:>+8.3f}')

    d[0,j].legend(loc='center right')
    d[0,j].set_axis_label('step', '\lambda')
    d[0,j].set_ylim(-16, 3)
d.show()
```
![fig:henon](https://github.com/llbxg/hundun/blob/main/docs/img/sample_les_jaco_or_no.png?raw=true)

### Lyapunov dimension

```python
def calc_lyapunov_dimension(les):
```

Calculate the Lyapunov dimension from LEs.

```python
from hundun import calc_les, calc_lyapunov_dimension
from hundun.equations import Lorenz


_, les = calc_les(Lorenz)
D_L = calc_lyapunov_dimension(les)
print(D_L)
```
```bash
2.0673058796702217
```


## To Do

* Synchronization
* Time series
* Baisn of attraction
* sample

## Dependencies

[{ Numpy }](https://numpy.org)
[{ Scipy }](https://scipy.org)
[{ Matplotlib }](https://matplotlib.org)


## Reference

#### Deterministic Nonperiodic Flow
(1963) Edward N. Lorenz   
DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2

#### A two-dimensional mapping with a strange attractor
(1976) M. Hénon   
DOI: 10.1007/BF01608556

#### Simple mathematical models with very complicated dynamics
(1976) Robert M. May   
DOI: 10.1038/261459a0

#### Measuring the strangeness of strange attractors
(1983) Peter Grassberger and Itamar Procaccia   
DOI: 10.1016/0167-2789(83)90298-1

####  Characterization of Strange Attractors
(1983) Peter Grassberger and Itamar Procaccia   
DOI: 10.1103/PhysRevLett.50.346

#### Determining embedding dimension for phase-space reconstruction using a geometrical construction
(1992) Matthew B. Kennel, Reggie Brown, and Henry D. I. Abarbanel   
DOI: 10.1103/PhysRevA.45.3403   

#### Practical method for determining the minimum embedding dimension of a scalar time series
(1997) Liangyue Cao   
DOI: 10.1016/S0167-2789(97)00118-8   

#### Recognizing determinism in a time series
(1993) Wayland, Richard and Bromley, David and Pickett, Douglas and Passamante, Anthony   
DOI: 10.1103/PhysRevLett.70.580