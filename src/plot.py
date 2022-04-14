# src/plot.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


## This module provide visualisation methods (1-D, 2-D and 3-D) for RBM results

# 1-D line graph
from matplotlib.animation import FuncAnimation


def plot(res, ax = None, plt_kwargs={'color':'blue', 'linewidth':.5}):
    '''
    Plotting the 1-D line graph for RBM results.

    Args:
        res : A two-dimensional numpy array (time by particles ) consisting the locations and time stamps for the
            particles.
        ax: Matplotlib.pyplot axes. If None, plot on the current axes.
        plt_kwargs: Other arguments to pass in function matplotlib.pyplot.plot.

    Returns:
        list of Line2D. A list of lines representing the plotted data.
    '''

    if res.ndim != 2:
        print('"plot" is only available for 1-D particle system, please resort to "animate" for 2-D or 3-D plotting.')
    if ax is None:
        ax = plt.gca()
    for i in range(res.shape[1]):
        l = ax.plot(res[:, i], **plt_kwargs)
    ax.set_xlabel('Time')
    ax.set_ylabel('Position')
    return l

# animate the evolving process of 2-D or 3-D for RBM results
def animate(res, ax = None, scatter_kwargs={}, animate_kwargs={}):
    '''
        Animate the evolving process of 2-D or 3-D for RBM results.

        Args:
            res : A three-dimensional numpy array (time by particles by axes) consisting the locations and time stamps
                for the particles.
            ax: Matplotlib.pyplot axes. If None, plot on the current axes.
            scatter_kwargs: Other arguments to pass in function matplotlib.pyplot.scatter.
            animate_kwargs: Other arguments to pass in function matplotlib.pyplot.animate.

        Returns:
            list of Line2D. A list of lines representing the plotted data.
        '''
    if res.ndim != 3 or res.shape[2] not in [2,3]:
        print('"animate" is only available for 2-D or 3-D particle system.')
    if ax is None:
        ax = plt.gca()
    sc = ax.scatter(res[0, :, 0], res[0, :, 1], cmap = 'viridis', **scatter_kwargs)
    def update_anim(i):
        print(i)
        sc.set_offsets(res[i])
        sc.set_array(np.sqrt(np.sum((res[i]-res[i-1])**2, axis=1)))
        return sc,
    anim = FuncAnimation(plt.gcf(), update_anim, np.arange(1, res.shape[0]), **animate_kwargs)
    plt.show()
    return anim