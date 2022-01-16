from itertools import accumulate as _accumulate
from operator import add as _add

import numpy as _np


def calc_les_difference(difference, N=5000, **options):
    model = _make_model(difference, **options)

    if model.dim == 1:
        return calc_les_difference_dim_1(model, N)
    else:
        return calc_les_difference_dim_geq_2(model, N)


def calc_les_difference_dim_geq_2(difference, N, n_average=100):
    if difference.inf:
        raise ValueError('Initial value is outside basin of attraction')

    Q, R = _np.linalg.qr(difference.j())

    R_list= [_np.diag(R)]
    les = [_np.log(_np.abs(_np.diag(R)))]
    for _ in range(N):
        difference.solve(*difference.internal_state)
        J = difference.j()
        Q, R = _np.linalg.qr(J@Q)
        R_list.append(_np.diag(R))
        les.append(_np.log(_np.abs(_np.diag(R))))

    les_list = [l/i for i, l in enumerate(_accumulate(les, _add), 1)]

    return _np.array(les_list), _np.average(les_list[-n_average:], axis=0)


def calc_les_difference_dim_1(difference, N, n_average=100):
    le_list = []
    for _ in range(N):
        difference.solve(*difference.internal_state)
        le_list.append(_np.log(_np.abs(difference.j())))
    le_list = _np.array(
        [le/i for i, le in enumerate(_accumulate(le_list, _add), 1) ])
    return le_list, _np.average(le_list[-n_average:])

def _make_model(difference, u0=None, **options):
    if u0 is None:
        model = difference.on_attractor(u0=u0, **options)
    else:
        model = difference(**options)
        model.u, model.t = u0, 0
    return model
