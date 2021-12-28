import numpy as _np

def embedding_seq(u_seq, T, D):
    idx = _np.arange(0,D,1)*T
    return _np.array([u_seq[idx+i,:] for i in range(len(u_seq)-(D-1)*T)])
