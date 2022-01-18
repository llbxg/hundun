from abc import ABC as _ABC, abstractmethod as _abstractmethod
from collections import deque as _deque
from inspect import getfullargspec as _getfullargspec
from itertools import cycle as _cycle

import numpy as _np


class Synchronization(_ABC):

    _signal_seq = None

    @_abstractmethod
    def signal(self):
        """sending signal"""

    @_abstractmethod
    def leading(self):
        '''leading model'''

    @_abstractmethod
    def supporting(self):
        '''supporting model'''

    def set_model(self, shift=0):
        dq = _deque([self.leading, self.supporting])
        dq.rotate(shift)
        self._cyc =_cycle(dq)
        return self._cyc

    def _sync(self):
        return next(self._cyc)

    def get_signal(self):
        if self._signal_seq is None:
            self._signal_seq = []

        signal = self.signal()
        self._signal_seq.append(signal)
        return signal

    @property
    def signal_seq(self):
        return _np.array(self._signal_seq)

    @property
    def sync(self):
        try:
            return next(self._cyc)
        except AttributeError:
            raise AttributeError(f'need {self.__class__.__name__}.set_model()')


def _make_parameter(func, params):
    signals = _getfullargspec(func).args
    params = {s:p for s, p in zip(signals[3:], params)}
    params['f'] = func
    return params


def _check_oscillator(oscillators):
    for o in oscillators:
        if not isinstance(o, Synchronization):
            raise TypeError(
                f'{o.__class__.__name__} need to have Synchronization')


def coupling_oneway(o1, o2, N):
    _check_oscillator([o1, o2])

    for _ in range(N):
        signal1 = o1.get_signal()
        signal2 = o2.get_signal()

        params1 = _make_parameter(o1.leading, [signal2])
        params2 = _make_parameter(o2.supporting, [signal1])

        o1.solve(*o1.internal_state, **params1)
        o2.solve(*o2.internal_state, **params2)

    return o1, o2
