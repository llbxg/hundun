import numpy as _np


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
