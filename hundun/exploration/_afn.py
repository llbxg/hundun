# Averaged False Neighbors - Algorithm

import warnings as _warnings

import numpy as _np
from scipy.spatial.distance import cdist as _cdist

from ._utils import embedding as _embedding
from ..utils import Drawing as _Drawing


def _dist(seq):
    return _cdist(seq, seq, metric='chebyshev')


def afn(u_seq, T=1, D_max=10):
    msg = 'It will not be available after version 0.2. ' \
          'Use `est_dimension_w_afn` instead.'
    _warnings.warn(msg)
    return est_dimension_w_afn(u_seq, T=T, D_max=D_max)[1]


def est_dimension_w_afn(u_seq, T, D_max=10,
                        threshold_E1=0.9, threshold_E2=1,
                        plot=True, path_save_plot=None):

    R_A = _np.std(u_seq)

    e_seq_list = [_embedding(u_seq, T, j) for j in range(1, D_max+3)]

    a_list, b_list = [], []
    for e_seq1, e_seq2 in zip(e_seq_list, e_seq_list[1:]):
        dist1 = _dist(e_seq1) + _np.eye(len(e_seq1))*10000
        dist2 = _dist(e_seq2)

        idx_n_list = dist1[:(l2 := len(dist2)), :l2].argmin(axis=0)

        dist1_min = dist1.min(axis=0)
        dist2_min = _np.array([d[idx] for idx, d in zip(idx_n_list, dist2)])

        # E2の計算のためのEの計算
        b = [_np.abs(e_line[-1] - e_seq2[idx][-1])/R_A
             for idx, e_line in zip(idx_n_list, e_seq2)]
        b_bar = _np.average(b)
        b_list.append(b_bar)

        # E1の計算のためのEの計算
        a = [R_Dk2/R_Dk1 for R_Dk1, R_Dk2 in zip(dist1_min, dist2_min)]
        a_bar = _np.average(a)
        a_list.append(a_bar)

    E_list = []
    for c in [a_list, b_list]:
        criterion = []
        for tau, tau_p1 in zip(c, c[1:]):
            criterion.append(tau_p1/tau)
        E_list.append(_np.array(criterion))

    dranges = _np.arange(1, len(E_list[0])+1)

    dimension_list = []
    for E, threshold in zip(E_list, [threshold_E1, threshold_E2]):
        if len(rule:=(E>threshold)) > 0:
            dim = dranges[rule][0]
        else:
            dim = None
        dimension_list.append(dim)

    if plot:
        d = _Drawing()
        for E, threshold, label, dim in zip(E_list,
                                            [threshold_E1, threshold_E2],
                                            ['E1', 'E2'],
                                            dimension_list):
            if dim is not None:
                d[0,0].scatter(dim, E[dim-1],
                               s=70, color='red', zorder=10)

            d[0,0].plot(dranges, E, label=label,
                        marker='.', markersize=5, zorder=5)

            d[0,0].axhline(threshold,
                           color='black', linestyle='dashed', linewidth=0.5,
                           zorder=0)
        d[0,0].legend()
        d[0,0].set_axis_label('Embedding~Dimension', r'E1~&~E2')
        if path_save_plot is not None:
            d.save(path_save_plot)
        d.show()

    return dimension_list, E_list
