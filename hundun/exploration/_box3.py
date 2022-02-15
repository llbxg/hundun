from typing import NamedTuple as _NamedTuple

import numpy as _np

from ..utils import Drawing as _Drawing


class _BoxDimensionResult(_NamedTuple):
    dimension : float  # 容量次元
    idx_dimension : int  # 最も相関係数が高かった時のインデックス

    min_correlation : float #
    correlations : _np.ndarray  # 相関係数
    slopes : _np.ndarray  # 1次関数によるフィッテイング時の傾き
    intercepts: _np.ndarray  # 1次関数によるフィッテイング時の切片

    log_frac_1_eps : _np.ndarray
    log_Ns : _np.ndarray
    dimensions : _np.ndarray  # 定義に沿った容量次元


def calc_dimension_capacity(u_seq, depsilon=0.02, base=7, loop=250,
                            min_correlation=0.999,
                            scale_down=True, batch_ave=10,  plot=True,
                            path_save_plot=None):

    dim, u_seq = _check_dim(u_seq)

    if scale_down:
        u_seq = _scale_down(u_seq)

    result = _calc_capacity_dimension(dim, u_seq, batch_ave,
                                      depsilon, base, loop, min_correlation)

    if plot:
        _check_box_dimension(result, batch_ave, path_save_plot)

    return result.dimension


def _calc_capacity_dimension(dim, u_seq, batch_ave, depsilon, base, loop,
                             min_correlation, mode='most'):
    log_frac_1_eps, log_Ns = _counting(dim, u_seq, depsilon, base, loop)
    dimensions = log_Ns / log_frac_1_eps

    args = (log_Ns, log_frac_1_eps, batch_ave)
    correlations, slopes, intercepts = _get_correlations_and_slopes(*args)

    idx = _decide_idx_ref_mode(correlations, slopes, mode=mode,
                               min_correlation=min_correlation)

    dimension = _np.average(dimensions[idx:idx+batch_ave])

    return _BoxDimensionResult(dimension, idx,
                               min_correlation,
                               correlations, slopes, intercepts,
                               log_frac_1_eps, log_Ns,
                               dimensions)

def _counting(dim, u_seq, depsilon, base, loop):
    '''
    epsilonごとに分割してカウントを行う。
    '''
    epsilon_list = _make_epsilon_list(depsilon, base, loop)

    log_frac_1_ep_list, log_N_list = [], []

    for ep in epsilon_list:
        log_N = _box_counting(dim, u_seq, ep)
        log_N_list.append(log_N)
        log_frac_1_ep_list.append(_np.log(1/ep))

    return _np.array(log_frac_1_ep_list), _np.array(log_N_list)


def _box_counting(dim, u_seq, ep):
    def make_edges(a, ep):
        return _np.arange(_np.min(a)-ep, _np.max(a)+ep, ep)

    def counting(x, y, bins):
        H, _, _ =  _np.histogram2d(x, y, bins=bins)
        return _np.sum(H>0)

    x = u_seq[:, 0]
    xedges = make_edges(x, ep)

    if dim == 1:
        H, _ = _np.histogram(x, bins=xedges)
        N = _np.sum(H>0)

    elif dim == 2:
        y = u_seq[:, 1]
        yedges = make_edges(y, ep)
        N = counting(x, y, (xedges, yedges))

    elif dim ==3:
        y, z = u_seq[:, 1], u_seq[:, 2]
        yedges = make_edges(y, ep)
        zedges = make_edges(z, ep)
        N = 0
        for z_left in zedges:
            z_right = z_left+ep
            new_u_seq = u_seq[(z_left<z) & (z<=z_right)]
            new_x, new_y = new_u_seq[:, 0], new_u_seq[:, 1]
            N += counting(new_x, new_y, (xedges, yedges))

    else:
        N = 0

    return _np.log(N)


def _get_correlations_and_slopes(h_seq, v_seq, batch_ave):
    correlation_list, slope_list, intercept_list = [], [], []

    for i in range(len(h_seq)-batch_ave):
        h_seq_batch, v_seq_batch = h_seq[i:i+batch_ave], v_seq[i:i+batch_ave]

        correlation = _np.corrcoef(h_seq_batch, v_seq_batch)[0, 1]
        correlation_list.append(correlation)

        slope_now, intercept = _np.polyfit(v_seq_batch, h_seq_batch, 1)
        slope_list.append(slope_now)
        intercept_list.append(intercept)

    correlations = _np.array(correlation_list)
    slopes = _np.array(slope_list)
    intercepts = _np.array(intercept_list)

    return correlations, slopes, intercepts


def _decide_idx_ref_mode(correlations, slopes, mode, min_correlation=0.999):
    correlations_over = _np.where(
        correlations>=min_correlation, correlations, 0)
    slopes_over = _np.where(
        correlations>=min_correlation, slopes, 0)

    if mode=='most':
        idx = int(_np.argmax(correlations_over))
    elif mode=='max':
        idx = int(_np.argmax(slopes_over))
    else:
        idx = 0

    return idx


def _check_box_dimension(result, batch, path_save_plot):
    color = {'b':'tab:blue', 'o':'tab:orange', 'g':'tab:green'}

    x = result.log_frac_1_eps
    log_Ns = result.log_Ns
    idx = result.idx_dimension
    a = result.dimension

    def poly(_x):
        b = log_Ns[idx] - a*x[idx]
        return _np.poly1d((a, b))(_x)

    slopes = result.slopes
    dimensions = result.dimensions
    correlations = result.correlations
    min_correlation = result.min_correlation

    x_lr = [x[0], x[-1]]
    y = poly(x_lr)

    s_batch = slice(idx, idx+batch)

    d = _Drawing(3, 2, number=True, figsize=(3.14*1.7*2, 3.14*2),
                number_place=(0.96, 0.04), number_size=10)

    d[0,0].scatter(x, log_Ns, s=3, color=color['b'])
    d[0,0].scatter(x[s_batch], log_Ns[s_batch], s=3, color=color['o'])
    d[0,0].plot(x_lr, y, color=color['o'], linewidth=0.5)

    d[1,0].scatter(x, dimensions, s=3, color=color['b'],
                   label=r'$\frac{\ln N(\epsilon)}{\ln \frac{1}{\epsilon}}$')
    d[1,0].scatter(x[:-batch], slopes, s=3, color='tab:green', label='slope')
    d[1,0].axhline(a, color=color['o'], linewidth=0.5, linestyle='dashed')

    d[2,0].scatter(x[:-batch], correlations, s=3, color=color['b'])
    d[2,0].scatter(x[idx],correlations[idx], color=color['o'])

    d[2,0].axhline(min_correlation, color='black', linestyle='dashed',
                   linewidth=0.5)

    d[0,0].set_ylabel(r'$\ln N(\epsilon)$')
    d[1,0].set_ylabel('$D_0$')
    d[1,0].legend(loc='lower left')
    d[2,0].set_ylim(min_correlation-(1-min_correlation)/10, 1.0001)
    d[2,0].set_axis_label(r'\ln \frac{1}{\epsilon}', r'correlation')

    for i in range(3):
        d[i,0].set_xlim(min(x), max(x))

    if idx<batch:
        s = slice(idx, idx+batch*2)
        s2 = slice(0, idx+batch*2)
    elif idx>len(correlations)-2*batch:
        s = slice(idx-batch, len(correlations))
        s2 = slice(idx-batch, len(x))
    else:
        s = s2 = slice(idx-batch, idx+batch*2)

    x_lr = [x[s2][0], x[s2][-1]]

    d[0,1].scatter(x[s2], log_Ns[s2])
    d[0,1].scatter(x[s_batch], log_Ns[s_batch])
    d[0,1].plot(x_lr, poly(x_lr), color=color['o'])

    d[1,1].scatter(x[s2], dimensions[s2], color=color['b'])
    d[1,1].scatter(x[s], slopes[s], color='tab:green')
    d[1,1].axhline(a, color=color['o'], linestyle='dashed')

    d[2,1].scatter(x[s], correlations[s])
    d[2,1].scatter(x[idx], correlations[idx])
    d[2,1].axhline(1, color='black', linestyle='dashed')
    d[2,1].axhline(min_correlation, color='black', linestyle='dashed')


    d[0,1].set_ylabel(r'$\ln N(\epsilon)$')
    d[1,1].set_ylabel('$D_0$')
    d[2,1].set_axis_label(r'\ln \frac{1}{\epsilon}', r'correlation')
    d[2,1].set_ylim(min_correlation-(1-min_correlation), 1.0001)
    for i in range(3):
        d[i,1].set_xlim(min(x_lr), max(x_lr))

    if path_save_plot is not None:
        d.save(path_save_plot)

    d.show()
    d.close()


def _check_dim(u_seq):
    if len(u_seq.shape)==1:
        u_seq = u_seq.reshape(len(u_seq), 1)
    return u_seq.shape[1], u_seq


def _make_epsilon_list(depsilon=0.02, base=7, loop=250):
    return _np.array([_np.e**(i*depsilon-base) for i in range(loop)])[::-1]


def _scale_down(seq):
    v_max = seq.max(axis=0, keepdims=True)
    v_min = seq.min(axis=0, keepdims=True)
    return seq/_np.max(v_max-v_min)
