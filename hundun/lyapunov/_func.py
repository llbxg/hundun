from itertools import accumulate as _accumulate
from operator import add as _add

import numpy as _np


def _calc_lyapunov_dimension(les):
    sum_le = _np.array(list(_accumulate(les, _add)))
    ok = les[sum_le>0]
    d = len(ok)

    return d + sum(ok)/_np.abs(les[d])


def calc_lyapunov_dimension(les):
    if not isinstance(les, _np.ndarray):
        les = _np.array(les)
    return _calc_lyapunov_dimension(les)


if __name__ == '__main__':
    les = [0.943, 0.020, -14.667]
    D_L = calc_lyapunov_dimension(les)
    print(D_L)
