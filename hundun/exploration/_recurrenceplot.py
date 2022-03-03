from math import dist as _dist
from itertools import combinations as _combinations

import numpy as _np

from ..utils._draw import Drawing as _Drawing


def simple_threshold(ds, theta=0.5):
    if (d_max:=_np.max(ds))!=0:
        pv = (ds/d_max>theta)*255
        return pv
    return ds


def calc_recurrence_plot(u_seq, rule=simple_threshold, *params, **kargs):
    size = len(u_seq)
    rp = _np.zeros((size,size))

    ds =  _np.array([_dist(u1, u2) for u1, u2 in _combinations(u_seq, 2)])

    pv = rule(ds, *params, **kargs)

    rp[_np.triu(_np.ones((size,size), dtype=bool), k=1)] = pv

    return rp+rp.T


def show_recurrence_plot(u_seq, rule=simple_threshold, cmap=False,
                         path_save_plot=None,
                         *params, **kargs):
    rp = calc_recurrence_plot(u_seq, rule, *params, **kargs)

    d = _Drawing()
    im = d[0,0].imshow(rp, origin='lower')
    if cmap:
        d.fig.colorbar(im)

    if path_save_plot is not None:
        d.save(path_save_plot)

    d.show()
    d.close()

    return rp
