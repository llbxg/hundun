from functools import partial as _partial

import matplotlib.pyplot as _plt
from matplotlib import rcParams as _rcParams
import numpy as _np

import string as _string
from itertools import cycle as _cycle


config = {
    'figure.subplot.hspace':0.4,

    'lines.linewidth':0.8,

    'font.family':['Times New Roman', 'sans-serif'],
    'font.size':8,

    'xtick.direction':'in',
    'xtick.major.width':0.5,
    'xtick.major.size':2.5,
    'xtick.major.pad':7,
    'ytick.direction':'in',
    'ytick.major.width':0.5,
    'ytick.major.size':2.5,

    'ytick.right':True,
    'xtick.top':True,

    'axes.linewidth': 0.5,

    'axes.titlesize': 12,

    'axes.labelsize': 12,
    'mathtext.fontset':'cm',

    'scatter.marker':'.',

    'axes.labelpad':10,

    'legend.fancybox':False,
    'legend.edgecolor':'black',
    'legend.borderpad':0.5,
    'patch.linewidth':0.5,

    'grid.linewidth':0.5
    }

_rcParams.update(config)


def _set_axis_label(ax, *xyz_labels, tex=True):
    for axis, label in zip(['x', 'y', 'z'], xyz_labels):
        method = getattr(ax, f'set_{axis}label')
        if tex:
            label = f'${label}$'
        method(label)


class Drawing(object):

    def __init__(self, rows=1, cols=1, number=False, three=False,
                 figsize_w=3.14*1.7*2, figsize_h=3.14*2):
        subplot_kw= dict(projection='3d') if three else {}
        alphabets = _cycle(_string.ascii_lowercase)
        fig, axis = _plt.subplots(nrows=rows, ncols=cols,
                                dpi=150, figsize=(figsize_w, figsize_h),
                                subplot_kw=subplot_kw)

        if not isinstance(axis, _np.ndarray):
            axis = _np.array([[axis]])
        else:
            if rows==1 or cols==1:
                axis = axis.reshape((rows, cols))

        if number:
            for ax in _np.ravel(axis):
                ax.set_title(f'({next(alphabets)})', x=0.95, y=0.92)

        for ax in axis.flatten():
            ax.set_axis_label =  _partial(_set_axis_label, ax)

        self.fig, self.ax = fig, axis

    def __getitem__(self,key):
        return self.ax[key]

    def cmap(self, cmap='viridis'):
        return _plt.cm.get_cmap(cmap)

    def legend(self, **params):
        _plt.legend(**params).get_frame().set_linewidth(0.5)

    def show(self):
        _plt.show()

    def save(self, path, dpi=200):
        _plt.savefig(path, dpi=dpi)

    def close(self):
        _plt.close()
