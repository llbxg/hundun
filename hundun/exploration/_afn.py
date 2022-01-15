# Averaged False Neighbors - Algorithm

import numpy as _np
from scipy.spatial.distance import cdist as _cdist

from ._utils import embedding as _embedding


def _dist(seq):
    return _cdist(seq, seq, metric='chebyshev')


def afn(u_seq, T=1, D_max=10):
    R_A = _np.std(u_seq)

    e_seq_list = [_embedding(u_seq, T, j) for j in range(1, D_max+3)]

    a_list, b_list = [], []
    for e_seq1, e_seq2 in zip(e_seq_list, e_seq_list[1:]):
        dist1 = _dist(e_seq1) + _np.eye(len(e_seq1))*10000
        dist2 = _dist(e_seq2)

        idx_n_list = dist1[:(l2 := len(dist2)), :l2].argmin(axis=0)

        dist1_min = dist1.min(axis=0)
        dist2_min = _np.array([d[idx] for idx, d in zip(idx_n_list, dist2)])

        b = [_np.abs(e_line[-1] - e_seq2[idx][-1])/R_A
             for idx, e_line in zip(idx_n_list, e_seq2)]
        b_bar = _np.average(b)
        b_list.append(b_bar)

        a = [R_Dk2/R_Dk1 for R_Dk1, R_Dk2 in zip(dist1_min, dist2_min)]
        a_bar = _np.average(a)
        a_list.append(a_bar)

    criterions = []
    for c in [a_list, b_list]:
        criterion = []
        for tau, tau_p1 in zip(c, c[1:]):
            criterion.append(tau_p1/tau)
        criterions.append(criterion)

    return criterions
