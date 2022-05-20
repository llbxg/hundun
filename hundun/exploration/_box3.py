from itertools import combinations as _combinations
from typing import NamedTuple as _NamedTuple

import numpy as _np

from ..utils import Drawing as _Drawing


class _DimensionResult(_NamedTuple):
    dimension : float  # 次元
    idx_dimension : int  # 最も相関係数が高かった時のインデックス

    min_correlation : float #
    correlations : _np.ndarray  # 相関係数
    slopes : _np.ndarray  # 1次関数によるフィッテイング時の傾き
    intercepts: _np.ndarray  # 1次関数によるフィッテイング時の切片

    log_frac_1_eps : _np.ndarray
    log_Ns : _np.ndarray
    dimensions : _np.ndarray  # 定義に沿った容量次元


class CalcDimension(object):

    def __init__(self, u_seq, scale_down, depsilon, base, loop,
                 batch_ave, min_correlation, plot, path_save_plot):
        self.dim, self.u_seq = self.check_dim(u_seq)
        if scale_down:
            self.u_seq = _scale_down(self.u_seq)

        self.batch_ave = batch_ave
        self.plot = plot
        self.min_correlation = min_correlation
        self.config_accuracy = (depsilon, base, loop)
        self.path_save_plot = path_save_plot

        self.length = len(self.u_seq)

    def __call__(self, q=0):
        result = self.main()

        if self.plot:
            _plot(result, self.batch_ave, self.path_save_plot, q=q)

        return result.dimension

    def main(self):
        accuracies, values = self.calc()

        log_accuracies = self.wrap_accuracies_in_main(accuracies)
        new_values =  self.wrap_value_in_main(values)
        dimensions = new_values / log_accuracies

        correlations, slopes, intercepts = \
            self._get_correlations_and_slopes(new_values, log_accuracies)
        self.correlations = correlations
        self.slopes = slopes

        idx = self._decide_idx_ref_mode(correlations)

        dimension = self.decide_dimension(idx, dimensions)

        return _DimensionResult(float(dimension), idx, self.min_correlation,
                                correlations, slopes, intercepts,
                                log_accuracies, new_values, dimensions)

    def calc(self):
        '''
        accuracyごとに計算(func)を行う.
        '''
        accuracies = self.make_accuracies(*self.config_accuracy)
        value_list = [self.func(epsilon) for epsilon in accuracies]
        return accuracies, _np.array(value_list)

    def func(self, epsilon):
        '''
        value = func(epsilon)
        '''

        x = self.u_seq[:, 0]
        xedges = self.make_edges(x, epsilon)

        if self.dim == 1:
            value = self.func_for_1dim(x, xedges)

        elif self.dim == 2:
            y = self.u_seq[:, 1]
            yedges = self.make_edges(y, epsilon)
            value = self.func_for_2dim(x, y, xedges, yedges)

        elif self.dim == 3:
            y, z = self.u_seq[:, 1], self.u_seq[:, 2]
            yedges = self.make_edges(y, epsilon)
            zedges = self.make_edges(z, epsilon)
            value_list = []
            for z_left in zedges:
                z_right = z_left + epsilon
                new_u_seq = self.u_seq[(z_left<z) & (z<=z_right)]
                new_x, new_y = new_u_seq[:, 0], new_u_seq[:, 1]
                value_list.append(self.func_for_2dim(new_x, new_y, xedges, yedges))
            value = self.wrap_value_3dim(value_list)

        else:
            value = 0

        return self.wrap_value(value)

    def func_for_1dim(self, x, xedges):
        return 0

    def func_for_2dim(self, x, y, xedges, yedges):
        return 0

    def decide_dimension(self, idx, dimensions):
        return _np.average(dimensions[idx:idx+self.batch_ave])

    def _decide_idx_ref_mode(self, correlations):
        correlations_over = _np.where(
            correlations>=self.min_correlation, correlations, 0)

        idx = int(_np.argmax(correlations_over))

        return idx

    def _get_correlations_and_slopes(self, h_seq, v_seq):
        batch_ave = self.batch_ave
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

    @staticmethod
    def wrap_value(value):
        return value

    @staticmethod
    def wrap_accuracies_in_main(accuracies):
        return _np.log(1/accuracies)

    @staticmethod
    def wrap_value_3dim(value_list):
        return sum(value_list)

    @staticmethod
    def wrap_value_in_main(values):
        return _np.log(values)

    @staticmethod
    def make_edges(a, ep):
        return _np.arange(_np.min(a)-ep, _np.max(a)+2*ep, ep)

    @staticmethod
    def check_dim(u_seq):
        if len(u_seq.shape)==1:
            u_seq = u_seq.reshape(len(u_seq), 1)
        return u_seq.shape[1], u_seq

    @staticmethod
    def make_accuracies(depsilon=0.02, base=7, loop=250):
        '''
        eベースでのaccuracyを作成する. 刻み幅はdepsilonとし,
        最小はe^(-base), 最大はe^((loop-1)*depsilon-base)となる.
        デフォルトではe^(-2)からe^(-7)のaccuracyのリストを返す.
        '''
        epsilon_list = [_np.e**(i*depsilon-base) for i in range(loop+1)]
        return _np.array(epsilon_list)[::-1]


class Capacity(CalcDimension):

    def func_for_1dim(self, x, xedges):
        H, _ = _np.histogram(x, bins=xedges)
        return _np.sum(H>0)

    def func_for_2dim(self, x, y, xedges, yedges):
        H, _, _ =  _np.histogram2d(x, y, bins=(xedges, yedges))
        return _np.sum(H>0)


class Information(CalcDimension):

    def func_for_1dim(self, x, xedges):
        H, _ = _np.histogram(x, bins=xedges)
        p = H/self.length
        return p[p>0]

    def func_for_2dim(self, x, y, xedges, yedges):
        H, _, _ =  _np.histogram2d(x, y, bins=(xedges, yedges))
        p = H/self.length
        return p[p>0]

    @staticmethod
    def wrap_value(p):
        return -1*_np.sum(_np.multiply(p, _np.log2(p)))

    @staticmethod
    def wrap_value_3dim(value_list):
        return _np.concatenate(value_list)

    @staticmethod
    def wrap_value_in_main(values):
        return values

    @staticmethod
    def wrap_accuracies_in_main(accuracies):
        return _np.log2(1/accuracies)


class Correlation(CalcDimension):
    '''
    Grassberger-Procaccia Algorithm (グラスバーガー - プロカッチャ アルゴリズム)
    '''
    def __init__(self, *args, **kwargs):
        super(Correlation, self).__init__(*args, **kwargs)
        self.distance = _np.array(
            [_dist(x_i, x_j) for x_i, x_j in _combinations(self.u_seq, 2)])

    def calc(self):
        '''
        accuracyごとに計算(func)を行う.
        '''
        accuracies = self.make_accuracies(*self.config_accuracy)
        crs = 2*_np.array(
            [_correlation_integrals(r, self.distance, len(self.u_seq))
             for r in accuracies])

        return accuracies, _np.array(crs)

    def decide_dimension(self, idx, dimensions):
        return _np.average(self.slopes[self.min_correlation<=self.correlations])

    @staticmethod
    def wrap_value_in_main(values):
        return -_np.log(values)

def _dist(a, b):
    return _np.linalg.norm(a - b)


def _correlation_integrals(r, distance, N):
    return _np.sum(r > distance)/(N**2)


def calc_dimension_capacity(u_seq, depsilon=0.02, base=7, loop=250,
                            min_correlation=0.999, scale_down=True,
                            batch_ave=10,  plot=True, path_save_plot=None):

    capacity = Capacity(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation,
        plot=plot, path_save_plot=path_save_plot)

    return capacity()


def calc_dimension_information(u_seq, depsilon=0.02, base=7, loop=250,
                               min_correlation=0.999, scale_down=True,
                               batch_ave=10,  plot=True, path_save_plot=None):

    infomation = Information(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation,
        plot=plot, path_save_plot=path_save_plot)

    return infomation(q=1)


def calc_dimension_correlation(u_seq, depsilon=0.02, base=7, loop=250,
                               min_correlation=0.999, scale_down=True,
                               batch_ave=10,  plot=True, path_save_plot=None):

    correlation = Correlation(
        u_seq, scale_down=scale_down,
        depsilon=depsilon, base=base, loop=loop,
        batch_ave=batch_ave, min_correlation=min_correlation,
        plot=plot, path_save_plot=path_save_plot)

    return correlation(q=2)


def _plot(result, batch, path_save_plot, q):
    color = {'b':'tab:blue', 'o':'tab:orange', 'g':'tab:green'}
    label_vertical_list = [r'$\ln N(\epsilon)$', r'$H(\epsilon)$',
                           r'$- \ln C(\epsilon)$']
    label_eq_list = [
        r'$\frac{\ln {N(\epsilon)}}{\ln \frac{1}{\epsilon}}$',
        r'$\frac{ {H}}{\log_2 \frac{1}{\epsilon}}$',
        r'$\frac{-\ln {C(\epsilon)}}{\ln \frac{1}{\epsilon}}$']

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

    label_D = f'$D_{q}$'

    d[0,0].scatter(x, log_Ns, s=3, color=color['b'])
    d[0,0].scatter(x[s_batch], log_Ns[s_batch], s=3, color=color['o'])
    d[0,0].plot(x_lr, y, color=color['o'], linewidth=0.5)

    if q != 2:
        d[1,0].scatter(x, dimensions, s=3, color=color['b'],
                       label=label_eq_list[q])

    d[1,0].scatter(x[:-batch], slopes, s=3, color='tab:green', label='slope')
    d[1,0].axhline(a, color=color['o'], linewidth=0.5, linestyle='dashed')

    d[2,0].scatter(x[:-batch], correlations, s=3, color=color['b'])
    d[2,0].scatter(x[idx],correlations[idx], color=color['o'])

    d[2,0].axhline(1, color='black', linestyle='dashed',linewidth=0.5)
    d[2,0].axhline(min_correlation, color='black', linestyle='dashed',
                   linewidth=0.5)

    d[0,0].set_ylabel(label_vertical_list[q])
    d[1,0].set_ylabel(label_D)
    d[1,0].legend(loc='lower left')
    d[2,0].set_ylim(min_correlation-(1-min_correlation)/10, 1.0001)
    if q != 1:
        d[2,0].set_axis_label(r'\ln \frac{1}{\epsilon}', r'correlation')
    else:
        d[2,0].set_axis_label(r'\log_2 \frac{1}{\epsilon}', r'correlation')
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

    if q!=2:
        d[1,1].scatter(x[s2], dimensions[s2], color=color['b'])
    d[1,1].scatter(x[s], slopes[s], color='tab:green')
    d[1,1].axhline(a, color=color['o'], linestyle='dashed')

    d[2,1].scatter(x[s], correlations[s])
    d[2,1].scatter(x[idx], correlations[idx])
    d[2,1].axhline(1, color='black', linestyle='dashed')
    d[2,1].axhline(min_correlation, color='black', linestyle='dashed')


    d[0,1].set_ylabel(label_vertical_list[q])
    d[1,1].set_ylabel(label_D)
    if q!=1:
        d[2,1].set_axis_label(r'\ln \frac{1}{\epsilon}', r'correlation')
    else:
        d[2,1].set_axis_label(r'\log_2 \frac{1}{\epsilon}', r'correlation')
    d[2,1].set_ylim(min_correlation-(1-min_correlation), 1.0001)
    for i in range(3):
        d[i,1].set_xlim(min(x_lr), max(x_lr))

    if path_save_plot is not None:
        d.save(path_save_plot)

    d.show()
    d.close()


def _scale_down(seq):
    v_max = seq.max(axis=0, keepdims=True)
    v_min = seq.min(axis=0, keepdims=True)
    return seq/_np.max(v_max-v_min)
