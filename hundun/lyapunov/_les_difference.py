import numpy as _np


def calc_les_difference(difference, N=5000, **options):
    model = _make_model(difference, **options)

    if model.dim == 1:
        return calc_les_difference_dim_1(model, N)
    else:
        return calc_les_difference_dim_geq_2(model, N)


def calc_les_difference_dim_geq_2(difference, N):
    if difference.inf:
        raise ValueError('Initial value is outside basin of attraction')

    Q, R = _np.linalg.qr(difference.j())

    R_list= [_np.diag(R)]

    for _ in range(N):
        difference.solve(*difference.internal_state)
        J = difference.j()
        Q, R = _np.linalg.qr(J@Q)
        R_list.append(_np.diag(R))

    R_sum = R_list[0]
    for R in R_list[1:]:
        R_sum = R_sum + _np.log(_np.abs(R))
    les = R_sum/(N)

    return les


def calc_les_difference_dim_1(difference, N):
    l = 0
    for _ in range(N):
        difference.solve(*difference.internal_state)
        l += _np.log(_np.abs(difference.j()))
    return (l/N)[0]


def _make_model(difference, u0=None, **options):
    if u0 is None:
        model = difference.on_attractor(u0=u0, **options)
    else:
        model = difference(**options)
        model.u, model.t = u0, 0
    return model
