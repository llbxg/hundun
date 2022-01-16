import numpy as _np
from scipy import signal as _signal


def reshape(u_seq):
    if (ndim:=u_seq.ndim) == 1:
        u_seq = u_seq.reshape(len(u_seq), 1)
    elif ndim >= 3:
        raise NotImplementedError(f'{ndim} ndim is not supported')
    return u_seq


def embedding_seq(u_seq, T, D):
    idx = _np.arange(0,D,1)*T
    return _np.array([u_seq[idx+i,:] for i in range(len(u_seq)-(D-1)*T)])


def embedding_seq_1dim(u_seq, T, D):
    idx = _np.arange(0,D,1)*T
    e_seq = _np.array([u_seq[idx+i,:] for i in range(len(u_seq)-(D-1)*T)])
    return e_seq.reshape(len(e_seq), D)


def embedding(u_seq, T, D):
    dim, length = u_seq.ndim, len(u_seq)
    if len(u_seq.shape)==1:
        u_seq = u_seq.reshape(length, dim)

    idx = _np.arange(0,D,1)*T
    e_seq = _np.array([u_seq[idx+i,:] for i in range(length-(D-1)*T)])

    if u_seq.shape[1] == 1:
        e_seq = e_seq.reshape(len(e_seq), D)
    return e_seq


def get_bottom(seq, threshold=float('inf')):
    _, dim = seq.shape

    lags = []
    for i in range(dim):
        us = seq[:, i]
        min_idx = _signal.argrelmin(us, order=1)[0]
        candidate = min_idx[us[min_idx]<=threshold]
        if len(candidate):
            lags.append(candidate[0])
        else:
            lags.append(None)
    return tuple(lags)
