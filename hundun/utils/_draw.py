from functools import partial as _partial
from os import path as _path

from matplotlib import rcParams as _rcParams
import matplotlib.animation as _anm
import matplotlib.pyplot as _plt
import numpy as _np

import string as _string
from itertools import cycle as _cycle


config_drawing = {
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


def _set_axis_label(ax, *xyz_labels, tex=True):
    for axis, label in zip(['x', 'y', 'z'], xyz_labels):
        method = getattr(ax, f'set_{axis}label')
        if tex:
            label = f'${label}$'
        method(label)


class Drawing(object):

    def __init__(self, rows=1, cols=1, number=False, three=False,
                 figsize=None, dpi=150, space=None, config=None,
                 number_place=None, number_size=None):

        if figsize is None:
            if rows <= cols:
                figsize = (3.14*1.7*2, 3.14*2)
            else:
                figsize = (3.14*2, 3.14*1.7*2)

        _rcParams.update(config or config_drawing)

        alphabets = _cycle(_string.ascii_lowercase)

        if three is True:
            three = tuple(range(1, rows*cols+1))
        elif isinstance(three, int):
            three = tuple([three])
        three = three or ()

        fig = _plt.figure(dpi=dpi, figsize=figsize)

        if three:
            _plt.subplots_adjust(wspace=0.4)

        if space is not None:
            _plt.subplots_adjust(wspace=space[0] ,hspace=space[1])



        axis = []
        for j in range(1, rows*cols+1):
            s = (rows, cols, j)
            kwargs = dict()

            if j in three:
                kwargs['projection']='3d'

            ax = fig.add_subplot(*s, **kwargs)
            axis.append(ax)

        axis = _np.array(axis).reshape((rows, cols))

        if number:
            x = 0.95 if number_place is None else number_place[0]
            y = 0.92 if number_place is None else number_place[1]
            size = (12 if number_size is None else number_size)
            for ax in _np.ravel(axis):
                ax.set_title(f'({next(alphabets)})', x=x, y=y, fontsize=size)

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

    @classmethod
    def plot_a_and_b(cls, u_seq_a , u_seq_b=None, /, legend=True,
                     color=None, name=None, u_seq_more=[], *args, **kwargs):
        d = cls(1, 2, *args, **kwargs)
        ax_label = ['x', 'y', 'z']
        for i, (a, b) in enumerate(zip([0, 2], [1, 1])):
            color_cycle = _cycle(color or ['red', 'blue'])
            name_cycle = _cycle(name or ['a', 'b'])
            new_seq_list = ([u_seq_a] if u_seq_b is None
                            else [u_seq_a, u_seq_b, *u_seq_more])
            for u_seq in new_seq_list:
                c, n = next(color_cycle), next(name_cycle)
                d[0,i].plot(u_seq[:, a], u_seq[:, b], color=c, label=f'${n}$')
                d[0,i].scatter(u_seq[0, a], u_seq[0, b],
                               color=c, s=40, marker='o',
                               zorder=10, edgecolor='white')
                d[0,i].scatter(u_seq[-1, a], u_seq[-1, b],
                               color=c, s=40, marker='s',
                               zorder=10, edgecolor='white')

            d[0,i].set_axis_label(ax_label[a], ax_label[b])

        if legend:
            d[0, 0].legend()

        return d

    @classmethod
    def trajectory_3d(cls, u_seq, shadow=True, *args, **kargs):

        x, y, z = u_seq[:, 0], u_seq[:, 1], u_seq[:, 2]
        u_left = _np.min(u_seq, axis=0)
        u_right = _np.max(u_seq, axis=0)
        scale = (u_right - u_left)/3
        u_left, u_right = u_left - scale, u_right + scale

        config_drawing['axes.labelsize']=20
        config_drawing['axes.labelpad']=0
        kargs['config']=config_drawing

        d = cls(1, 1, three=True, *args, **kargs)
        ln, = d[0,0].plot([], [], [])

        def settings():
            d[0,0].set_xlim(u_left[0], u_right[0])
            d[0,0].set_ylim(u_left[1], u_right[1])
            d[0,0].set_zlim(u_left[2], u_right[2])
            d[0,0].xaxis.pane.set_facecolor("white")
            d[0,0].yaxis.pane.set_facecolor("white")
            d[0,0].zaxis.pane.set_facecolor("white")
            d[0,0].xaxis.pane.set_edgecolor('black')
            d[0,0].yaxis.pane.set_edgecolor('black')
            d[0,0].zaxis.pane.set_edgecolor('black')
            d[0,0].xaxis.line.set_linewidth(0.2)
            d[0,0].yaxis.line.set_linewidth(0.2)
            d[0,0].zaxis.line.set_linewidth(0.2)

            d[0,0].set_axis_label('x', 'y', 'z')
            d[0,0].grid(False)
            d[0,0].set_xticks([])
            d[0,0].set_yticks([])
            d[0,0].set_zticks([])

            d[0,0].view_init(azim=127,elev=27)

            return ln,

        def update(i):
            if i != 0:
                _plt.cla()
            settings()

            s0 = slice(0, i+1)
            s = slice(0,i+5)
            s1 = slice(i, i+5)
            l = len(x[s])

            d[0,0].plot(x[s0], y[s0], z[s0], linewidth=0.2, color='blue')
            d[0,0].scatter(x[s1], y[s1], z[s1],
                           edgecolor=None, color='red', s=10)

            if shadow:
                setting = {'linewidth':0.2, 'color': 'gray'}
                d[0,0].plot(x[s], y[s],  _np.full(l, u_left[2]), **setting)
                d[0,0].plot(x[s],  _np.full(l, u_left[1]), z[s], **setting)
                d[0,0].plot(_np.full(l, u_right[0]), y[s], z[s], **setting)

                setting = {'s':2, 'color':'gray'}
                d[0,0].scatter(x[s1], y[s1],  u_left[2], **setting)
                d[0,0].scatter(x[s1],  u_left[1], z[s1], **setting)
                d[0,0].scatter(u_right[0], y[s1], z[s1], **setting)

        settings()
        ani = _anm.FuncAnimation(d.fig, func=update, init_func=settings,
                                 interval=10, blit=True, save_count=len(x))
        return d, ani
