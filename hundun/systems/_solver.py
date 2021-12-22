from functools import partial as _partial

import numpy as _np


class Solver(object):

    A = NotImplemented
    B = NotImplemented
    n_stages = NotImplemented

    def __init__(self, f, *, h=0.01, **params):
        self.f, self.h = f, h
        self.params = params

    def __call__(self, t, u, **params):
        params = params or self.params
        K = _np.zeros((self.n_stages, u.shape[0]), dtype=u.dtype)
        return rk_step(self.f, t, u, self.A, self.B, K, self.h, **params)

    def __repr__(self):
        v = vars(self)
        f_name = v['f'].__name__
        return f"{self.__class__.__name__}(f={f_name}, *, h={v['h']})"


class RK4(Solver):
    A = _np.array([
        [0, 0, 0, 0],
        [1/2, 0, 0, 0],
        [0, 1/2, 0, 0],
        [0, 0, 1, 0]
    ])
    B = _np.array([1/6, 2/6, 2/6, 1/6])
    n_stages = 4


class Runge2(Solver):
    A = _np.array([
        [0, 0],
        [1/2, 0]
    ])
    B = _np.array([0, 1])
    n_stages = 2


def rk_step(f, t, u, A, B, K, h, **params):
    func = _partial(f, **params)
    K[0] = func(t, u)
    for s, a in enumerate(A[1:], start=1):
        c = sum(a[:s])
        K[s] = func(t + c*h, u + h*_np.dot(a[:s], (K[:s])))
    return u + h*_np.dot(K.T, B)


def solve_simple(f, t0, u0, n_loop, *, h=0.01, solver=RK4):
    sol = solver(f, h=h)
    u = _np.array(u0)
    t_seq = [t0 + i*h for i in range(n_loop)]

    u_seq = _np.array([u0, *[u := sol(t, u) for t in t_seq]])
    t_seq = _np.array([*t_seq, t0 + n_loop*h])
    return t_seq, u_seq
