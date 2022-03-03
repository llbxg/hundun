# Mutual Information

import numpy as _np

from ._utils import (reshape as _reshape, get_bottom as _get_bottom)
from ..utils import Drawing as _Drawing

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


def mutual_info(u_seq, tau):
    u_seq = _reshape(u_seq)

    N, dim = u_seq.shape

    miss = []
    for i in range(dim):
        us = u_seq[:, i]
        mis = [calc_mutual_info(us[:-t], us[t:], N, bins=32)
               for t in range(1, tau+1)]
        miss.append(mis)

    return _np.array(miss).T


def est_lag_w_mi(u_seq, tau_max, plot=True, path_save_plot=None):
    mi_seq = mutual_info(u_seq, tau_max)
    bottom = _get_bottom(mi_seq)

    if plot:
        d = _Drawing()
        for i, idx in enumerate(bottom):
            d[0,0].plot(mi_seq[:, i], label=f'{i}',
                        marker='.', markersize=5, zorder=5)
            d[0,0].scatter(idx, mi_seq[idx, i], s=70, color='red', zorder=10)

        d[0,0].set_axis_label('Time~lag', 'Mutual~Information~[bit]')
        d[0,0].set_xlim(0, tau_max-1)
        d[0,0].legend()
        if path_save_plot is not None:
            d.save(path_save_plot)
        d.show()
        d.close()

    return list(bottom)
