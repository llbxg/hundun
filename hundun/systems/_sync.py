from abc import ABC as _ABC, abstractmethod as _abstractmethod
from collections import deque as _deque
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
