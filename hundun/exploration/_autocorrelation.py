from itertools import cycle as _cycle

import matplotlib as _mpl
import numpy as _np

from ._utils import (bartlett as _bartlett,
                     get_minidx_below_seq as _get_minidx_below_seq,
                     reshape as _reshape)
from ..utils import Drawing as _Drawing

def _autocovariance_function(dim, u_seq, u_bar, tau, N):

    value = _np.zeros(dim)

    for t in range(N-1-tau):
        value+= (u_seq[t]-u_bar)*(u_seq[t+tau]-u_bar)

    return value/(N-tau)


def _autocorrelation_function(dim, u_seq, u_bar, tau, N):
    gamma_seq = _autocovariance_function(dim, u_seq, u_bar, tau, N)
    return gamma_seq/gamma_seq[0]


def acf(u_seq, tau):
    u_seq = _reshape(u_seq)

    N, dim = u_seq.shape

    u_bar = _np.average(u_seq, axis=0)

    gamma_seq = _np.array([_autocovariance_function(dim, u_seq, u_bar, t, N)
                           for t in range(tau)])

    return gamma_seq/gamma_seq[0]


def est_lag_w_acf(u_seq, tau_max, alpha=0.95,
                  plot=True, path_save_plot=None):

    rho_seq = acf(u_seq, tau_max)
    var_seq = _bartlett(rho_seq, alpha)
    idx_list = _get_minidx_below_seq(rho_seq, var_seq)

    if plot:
        color = _cycle(_mpl.rcParams['axes.prop_cycle'])
        d = _Drawing()
        for i, idx in enumerate(idx_list):
            c = next(color)['color']
            d[0,0].plot(var_seq[:, i],
                        label=f'standard error - {i}', color=c, linestyle='dashed')
            d[0,0].stem(rho_seq[:, i], label=f'acf - {i}',
                        markerfmt='.', linefmt=c, use_line_collection=True)
            d[0,0].scatter(idx, rho_seq[idx, i],
                           zorder=10, marker='o', color='red', s=20)

        d[0,0].legend()
        d[0,0].set_axis_label('Time~lag', 'Correlation~coefficient')
        d[0,0].set_xlim(0, tau_max-1)
        if path_save_plot is not None:
            d.save(path_save_plot)
        d.show()
        d.close()

    return idx_list
