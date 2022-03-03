# False Nearest Neighbors - Algorithm

import warnings as _warnings

import numpy as _np
from scipy.spatial.distance import cdist as _cdist

from ._utils import embedding as _embedding
from ..utils import Drawing as _Drawing

def _dist(seq):
    return _cdist(seq, seq, metric='euclidean')


def fnn(u_seq, threshold_R=10, threshold_A=2, T=50, D_max=10):
    msg = 'It will not be available after version 0.2. ' \
          'Use `est_dimension_w_fnn` instead.'
    _warnings.warn(msg)

    est_dimension_w_fnn(u_seq, T=T, D_max=D_max,
                        threshold_R=threshold_R, threshold_A=threshold_A)


def est_dimension_w_fnn(u_seq, T, D_max=10,
                        threshold_R=10, threshold_A=2, threshold_percent=1,
                        plot=True, path_save_plot=None):

    R_A = _np.std(u_seq)

    e_seq_list = [_embedding(u_seq, T, j) for j in range(1, D_max+2)]

    percentage_list = []
    for e_seq1, e_seq2 in zip(e_seq_list, e_seq_list[1:]):
        dist1 = _dist(e_seq1) + _np.eye(len(e_seq1))*10000
        dist2 = _dist(e_seq2)

        idx_n_list = dist1[:(l2 := len(dist2)), :l2].argmin(axis=0)

        dist1_min = dist1.min(axis=0)
        dist2_min = _np.array([d[idx] for idx, d in zip(idx_n_list, dist2)])

        percentage = 0
        for R_Dk1, R_Dk2 in zip(dist1_min, dist2_min):
            d = _np.sqrt((R_Dk2**2-R_Dk1**2)/(R_Dk1**2))
            criterion_1 = d > threshold_R
            criterion_2 = R_Dk2/R_A >= threshold_A
            if criterion_1 and criterion_2:
                percentage += 1
        percentage_list.append(percentage*(1/len(e_seq2)))

    percentages = _np.array(percentage_list)*100
    dranges = _np.arange(1, len(percentages)+1)
    dimension = dranges[percentages<threshold_percent][0]

    if plot:
        d = _Drawing()
        d[0,0].plot(dranges, percentages,
                    marker='.', markersize=5, color='tab:blue', zorder=5)
        d[0,0].axhline(threshold_percent,
                       color='black', linestyle='dashed', linewidth=0.5,
                       zorder=0)
        d[0,0].scatter(dimension, percentages[dimension-1],
                       s=70, color='red', zorder=10)

        d[0,0].set_axis_label('Embedding~Dimension', r'False~NN~Percentage~[\%]')
        if path_save_plot is not None:
            d.save(path_save_plot)
        d.show()

    return dimension, percentages
