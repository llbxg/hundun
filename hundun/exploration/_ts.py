import numpy as _np
from scipy import signal as _signal

from ..exploration._autocorrelation import (
    autocovariance_function as _autocovariance_function,)
from ..exploration._mutualinfo import calc_mutual_info as _calc_mutual_info
from ..exploration._recurrenceplot import (
    calc_recurrence_plot as _calc_recurrence_plot,
    show_recurrence_plot as _show_recurrence_plot)
from ..exploration._utils import (embedding_seq as _embedding_seq)
from ..exploration._gp import (
    calc_correlation_dimention_w_gp as _calc_correlation_dimention_w_gp)

class TimeSeries(object):

    def __init__(self, u_seq):
        # TODO: ndim>2の時のdimの変換いる
        if not isinstance(u_seq, _np.ndarray):
            u_seq = _np.array(u_seq)

        if (ndim:=u_seq.ndim) == 1:
            u_seq = u_seq.reshape(len(u_seq), 1)
        elif ndim >= 3:
            raise NotImplementedError(f'{ndim} ndim is not supported')

        self.ndim = ndim
        self.N, self.dim = u_seq.shape

        self.u_seq = u_seq

    def __array__(self):
        return self.u_seq

    def __array_wrap__(self, out_arr, context=None):
        return TimeSeries(out_arr)

    def __repr__(self):
        return f'TimeSeries(u_seq={self.u_seq})'

    def __getitem__(self, key):
        return self.u_seq[key]

    def __len__(self):
        return self.N

    @property
    def average(self):
        return _np.average(self.u_seq, axis=0)

    @property
    def shape(self):
        return self.u_seq.shape

    def e_seq(self, T, D):
        return _embedding_seq(self.u_seq, T, D)

    def calc_D2_w_gp(self, T, D):
        D2 , _, _ = _calc_correlation_dimention_w_gp(self.e_seq(T, D))
        return D2

    def calc_recurrence_plot(self, *params, **kwargs):
        return _calc_recurrence_plot(self.u_seq, *params, **kwargs)

    def show_recurrence_plot(self, *params, **kwargs):
        return _show_recurrence_plot(self.u_seq, *params, **kwargs)

    def autocovariance_function(self, tau=100):
        gamma_seq = [
            _autocovariance_function(self.dim, self.u_seq, self.average,
                                     t, self.N)
            for t
            in range(tau)
        ]
        return _np.array(gamma_seq)

    def autocorrelation_function(self, tau=100):
        gamma_seq = self.autocovariance_function(tau)
        return gamma_seq/gamma_seq[0]

    def mutual_info(self, tau=100):
        miss = []
        for i in range(self.dim):
            us = self.u_seq[:, i]
            mis = [
                _calc_mutual_info(us[:-tau], us[tau:], self.N, bins=32)
                for tau
                in range(1, tau+1)]
            miss.append(mis)
        return _np.array(miss).T

    def calc_lag_w_acrf(self, threshold=0.05):
        rho_seq = _np.abs(self.autocorrelation_function())
        lags = []
        for i in range(self.dim):
            us = rho_seq[:, i]
            min_idx = _signal.argrelmin(us, order=1)[0]
            candidate = min_idx[us[min_idx]<=threshold]
            if len(candidate):
                lags.append(candidate[0])
            else:
                lags.append(None)
        return tuple(lags)

    def calc_lag_w_mi(self, **options):
        mi = self.mutual_info(**options)
        return self._get_bottom(mi)

    def _get_bottom(self, seq, threshold=float('inf')):
        seq = _np.abs(seq)
        lags = []
        for i in range(self.dim):
            us = seq[:, i]
            min_idx = _signal.argrelmin(us, order=1)[0]
            candidate = min_idx[us[min_idx]<=threshold]
            if len(candidate):
                lags.append(candidate[0])
            else:
                lags.append(None)
        return tuple(lags)
