# Wayland - Algorithm

from random import sample as _sample
from statistics import median as _median

import numpy as _np

from ._utils import (embedding_seq as _embedding_seq,
                     reshape as _reshape)
from ..utils import Drawing as _Drawing


def wayland(u_seq, K=4, tau=5, Q=10, M = 51, D_max=10, reshapemode=True):
    _, median_e_trans_ave= est_dimension_w_wayland(
        u_seq=u_seq, T=1, D_max=D_max,
        tau=tau, K=K, Q=Q, M=51, reshape=reshapemode,
        plot=False)
    return median_e_trans_ave


def est_dimension_w_wayland(u_seq, T, D_max=10,
                            tau=5, K=4, Q=10, M=51,
                            reshape=True,
                            plot=True, path_save_plot=None):
    if reshape:
        u_seq = _reshape(u_seq)

    median_e_trans_ave = []
    for D in range(1, D_max+1):
        e_seq = _embedding_seq(u_seq, T, D)
        N = len(e_seq)
        median_e_trans = []
        for _ in range(Q):
            e_trans = []
            for idx_n in _sample(range(len(e_seq)), M):
                e_n = e_seq[idx_n]

                idx_distance = ((i, _np.linalg.norm(e))
                                for i, e in enumerate(e_seq[:N-tau]-e_n))
                idx_distance = sorted(idx_distance, key=lambda x: x[1])[1:K+1]

                v_seq = _np.array([
                    e_seq[idx + tau] - e_seq[idx] for idx, _ in idx_distance])
                v_bar = _np.sum(v_seq, axis=0)/(K+1)
                bunshi = [_np.linalg.norm(v)/_np.linalg.norm(v_bar)
                          for v in v_seq - v_bar]
                e_trans.append(sum(bunshi)/(K+1))

            median_e_trans.append(_median(e_trans))
        median_e_trans_ave.append(_np.average(median_e_trans))

    median_e_trans_ave = _np.array(median_e_trans_ave)
    dimension = median_e_trans_ave.argmin()+1

    if plot:
        d = _Drawing()
        d[0,0].plot(range(1, len(median_e_trans_ave)+1), median_e_trans_ave,
                    marker='.', markersize=5, color='tab:blue', zorder=5)
        d[0,0].scatter(median_e_trans_ave.argmin()+1, median_e_trans_ave.min(),
                        s=70, color='red', zorder=10)
        d[0,0].set_axis_label('Dimension', 'median(E_{trans})')
        d[0,0].set_yscale('log')
        if path_save_plot is not None:
            d.save(path_save_plot)
        d.show()

    return dimension, median_e_trans_ave


if __name__ == '__main__':
    from ..equations import Lorenz

    u_seq = Lorenz.get_u_seq(5000)[::5, 0]

    est_dimension_w_wayland(u_seq, T=1)
