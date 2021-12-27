# Mutual Information

import numpy as _np


def _log2(a):
    return _np.log2(a+1e-15)


def calc_mutual_info(x, y, N, **options):
    H_x = calc_entropy(x, N, **options)
    H_y = calc_entropy(y, N, **options)
    H_xy = calc_joint_entropy(x, y, N, **options)
    return H_x + H_y - H_xy


def calc_entropy(x, N, **options):
    hist, _ = _np.histogram(x, **options)
    ps = hist/N
    H_x = -_np.sum(ps*_log2(ps))
    return H_x


def calc_joint_entropy(x, y, N, **options):
    hist, _, _ = _np.histogram2d(x, y, **options)
    ps = (hist/N).flatten()
    H_xy = -_np.sum(ps*_log2(ps))
    return H_xy
