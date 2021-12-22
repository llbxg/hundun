from itertools import accumulate as _accumulate
from operator import add as _add

import numpy as _np


def calc_les_differential(differential, **options):

    if differential().jacobian() is None:
        return calc_les_differential_w_orth(differential, **options)

    else:
        return calc_les_differential_w_qr(differential, **options)


def calc_les_differential_w_qr(differential,
                h=0.001, N=100000, u0=None, n_average=100,
                dynamic_para=None,
                error_func=False,
                **options):

    model = _make_model(differential, h=h, u0=u0, **options)

    if error_func:
        jacobian = model.dynamic_jacobian
        func_solve = lambda t, u, **option : \
            model.solve(t, u, f=model.dynamic_eq, **option)
    else:
        jacobian = model.j
        func_solve = model.solve

    # main
    dim = model.dim

    Q = _np.eye(dim)
    dp = dynamic_para or [dict() for _ in range(N)]

    les = []
    for _, p in enumerate(dp):
        U = (_np.eye(dim)+jacobian(**p)*h)@Q
        Q, R = _np.linalg.qr(U)

        les.append(_np.log(_np.abs(R.diagonal())))

        func_solve(model.t, model.u, h=h, **p)

    les_list = [l/(i*h) for i, l in enumerate(_accumulate(les, _add), 1)]

    return _np.array(les_list), _np.average(les_list[-n_average:], axis=0)


def calc_max_le_differential(differential,
                     h=0.01, N=10000, u0=None, n_average=100, n_split=10,
                     dynamic_para=None,
                     error_func=False,
                     **options):

    model = _make_model(differential, u0=u0, h=h, **options)

    if error_func:
        func_solve = lambda t, u, **option : \
            model.solve(t, u, f=model.dynamic_eq, **option)
    else:
        func_solve = model.solve

    # main
    w_s = _np.random.random(model.dim)*(10**(-3))
    z_0 = _np.linalg.norm(w_s)

    u_tilde = model.u
    u = u_tilde + w_s

    u_tilde_list = [u_tilde]

    dp = dynamic_para or [dict() for _ in range(N)]

    for i, para in enumerate(dp):
        _, u_tilde = func_solve(i*h, u_tilde, h=h, **para)
        u_tilde_list.append(u_tilde)

    w_e = w_s
    v = _np.log(_np.linalg.norm(w_e)*(1/z_0))
    T = 1/n_split
    les_list = [v/T]

    for i, u_tilde in enumerate(u_tilde_list[n_split::n_split]):
        for j in range(n_split):
            d = dp[i*n_split+j]
            _, u = func_solve(i*n_split*h+j*h, u, h=h, **d)

        w_e = u - u_tilde
        v += _np.log(_np.linalg.norm(w_e)*(1/z_0))

        T=(1/n_split)*(i+2)
        les_list.append(v/T)

        w_s = z_0/_np.linalg.norm(w_e)*w_e
        u = u_tilde+w_s

    return les_list, _np.average(les_list[-n_average:])


def calc_les_differential_w_orth(differential,
                  h=0.01, N=10000, u0=None, n_average=100,
                  dynamic_para=None,
                  error_func=False,
                  **options):

    model = _make_model(differential, u0=u0, h=h, **options)

    if error_func:
        func_solve = lambda t, u, **option : \
            model.solve(t, u, f=model.dynamic_eq, **option)
    else:
        func_solve = model.solve

    dim = model.dim

    # main
    d_0 = _np.array([_np.random.rand(dim) for _ in range(dim)])
    d_0 = _np.linalg.qr(d_0.T)[0].T

    u_tilde = model.u
    u_hat = [u_tilde + w for w in d_0]

    lm = [[] for _ in range(dim)]

    dp = dynamic_para or [dict() for _ in range(N)]

    for i, para in enumerate(dp):
        t = i*h
        _, u_tilde = func_solve(t, u_tilde, **para)

        d_tau = []
        for i in range(dim):
            _, u_hat[i] = func_solve(t, u_hat[i], h=h, **para)
            d_tau.append(u_hat[i] - u_tilde)
        d_tau = _np.array(d_tau)

        d_tau_bot = _np.linalg.qr(d_tau.T)[0].T

        for i in range(dim):
            on = _np.linalg.norm(d_0[i])
            d_0[i] =  on * d_tau_bot[i]
            u_hat[i] = u_tilde + d_0[i]

        d_ups, d_downs = _np.ones(3), _np.ones(3)
        for i in range(dim):
            d_ups = _np.linalg.norm(_np.outer(d_ups, d_tau[i]))
            d_downs = _np.linalg.norm(_np.outer(d_downs, d_0[i]))

            lm[i].append(_np.log(d_ups / d_downs))

    calc_l = lambda l : [sum(l[:i])/(i*h) for i in range(1, len(l)+1)]
    les_plus_list = [_np.array(calc_l(l)) for l in lm]

    les_list = []
    les_list.append(les_plus_list[0])
    for i in range(1, dim):
        les_list.append(les_plus_list[i] - les_plus_list[i-1])
    les_average = [_np.average(l[-n_average:]) for l in les_list]

    les_average=sorted(les_average, reverse=True)
    return _np.array(les_list).T, les_average


def _make_model(differential, u0=None, h=0.01, **options):
    if u0 is None:
        model = differential.on_attractor(**options)
    else:
        model = differential(**options)
        model.u, model.t = u0, 0
    return model
