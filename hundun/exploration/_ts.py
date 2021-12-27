import numpy as _np
from scipy import signal as _signal

from ..exploration._recurrenceplot import (
    calc_recurrence_plot as _calc_recurrence_plot,
    show_recurrence_plot as _show_recurrence_plot)


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

    def calc_recurrence_plot(self, *params, **kwargs):
        return _calc_recurrence_plot(self.u_seq, *params, **kwargs)

    def show_recurrence_plot(self, *params, **kwargs):
        return _show_recurrence_plot(self.u_seq, *params, **kwargs)
