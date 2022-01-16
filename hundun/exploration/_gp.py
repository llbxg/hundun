# Grassberger-Procaccia Algorithm (グラスバーガー - プロカッチャ アルゴリズム)

from itertools import combinations as _combinations

import numpy as _np


def _dist(a, b):
    return _np.linalg.norm(a - b)


def _correlation_integrals(r, distance, N):
    return _np.sum(r > distance)/(N**2)


def _calc_cr(e_seq, base_r, h_r, loop):
    distance = _np.array(
        [_dist(x_i, x_j) for x_i, x_j in _combinations(e_seq, 2)])

    rs = _np.array([_np.e**(i*h_r-base_r) for i in range(loop)])[::-1]
    crs = 2*_np.array(
        [_correlation_integrals(r, distance, len(e_seq)) for r in rs])
    return rs, crs


def calc_correlation_dimention_w_gp(e_seq,
                                    base=8, h_r=0.05, loop=200, batch_ave=10,
                                    normalize=True):
    """
    bは相関係数を計算するためのバッチサイズです。
    プロットの直線部分が相関係数0.999以上の場合の最大の勾配をD_2としている。
    """
    if normalize:
        v_max = e_seq.max(axis=0, keepdims=True)
        v_min = e_seq.min(axis=0, keepdims=True)
        e_seq = (e_seq-v_min)/(v_max-v_min)

    rs, crs = _calc_cr(e_seq, base, h_r, loop)

    correlation_max, slope = 0, None
    for i in range(len(crs)-batch_ave):
        log_cr, log_r = _np.log(crs[i:i+batch_ave]), _np.log(rs[i:i+batch_ave])
        correlation = _np.corrcoef(log_cr, log_r)[0,1]
        if (correlation >= 0.999) and (correlation > correlation_max):
            slope, _ = _np.polyfit(log_r, log_cr ,1)
            correlation_max = correlation

    return slope, rs, crs
