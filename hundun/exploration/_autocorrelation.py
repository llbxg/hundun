import numpy as _np

from ._utils import reshape as _reshape


def _autocovariance_function(dim, u_seq, u_bar, tau, N):

    value = _np.zeros(dim)

    for t in range(N-1-tau):
        value+= (u_seq[t]-u_bar)*(u_seq[t+tau]-u_bar)

    return value/(N-tau)


def _autocorrelation_function(dim, u_seq, u_bar, tau, N):
    gamma_seq = _autocovariance_function(dim, u_seq, u_bar, tau, N)
    return gamma_seq/gamma_seq[0]


def autocorrelation_function(u_seq, tau):
    u_seq = _reshape(u_seq)

    N, dim = u_seq.shape

    u_bar = _np.average(u_seq, axis=0)

    gamma_seq = _np.array([_autocovariance_function(dim, u_seq, u_bar, t, N)
                           for t in range(tau)])

    return gamma_seq/gamma_seq[0]
