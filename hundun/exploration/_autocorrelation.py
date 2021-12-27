import numpy as _np


def autocovariance_function(dim, u_seq, u_bar, tau, N):

    value = _np.zeros(dim)

    for t in range(N-1-tau):
        value+= (u_seq[t]-u_bar)*(u_seq[t+tau]-u_bar)

    return value/(N-tau)


def autocorrelation_function(dim, u_seq, u_bar, tau, N):
    gamma_seq = autocovariance_function(dim, u_seq, u_bar, tau, N)
    return gamma_seq/gamma_seq[0]
