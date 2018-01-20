"""

Utility functions for conv-nets tutorial

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def generate_dataset(func, n_train, n_test, num_labels, **kwargs):
    """Create synthetic classification data-sets.

    Parameters
    ----------
    func : one of {`make_blobs`, `make_circles`, `make_moons`}
        What kind of data to make.
    n_train : int
        The size of the training set.
    n_test : int
        The size of the test set.
    num_labels : int
        The number of classes.

    Returns
    -------
    train_data, test_data : 2D arrays
        Dimensions: {n_train, n_test} by 2
    train_labels, test_labels: one-hot encoder arrays
        These have dimensions {n_train, n_test} by num_labels
    """
    fvecs, labels = func(n_train + n_test, **kwargs)

    train_data, test_data, train_labels, test_labels = \
        train_test_split(fvecs.astype(np.float32),
                         labels,
                         train_size=n_train,
                         test_size=n_test)
    
    
    return train_data, test_data, train_labels, test_labels


def draw_neural_net(layer_sizes,
                    left=.1, right=.9, bottom=.1, top=.9,
                    ax=None,
                    draw_weights=True,
                    draw_funcs=False):
    """Draw a neural network cartoon using matplotilb.

    Based on: https://gist.github.com/craffel/2d727968c3aaebd10359

    Parameters
    ----------
        ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        left : float
            The center of the leftmost node(s) will be placed here
        right : float
            The center of the rightmost node(s) will be placed here
        bottom : float
            The center of the bottommost node(s) will be placed here
        top : float
            The center of the topmost node(s) will be placed here
        layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    """
    if ax is None:
        fig, ax = plt.subplots(1)
        ax.axis('off')

    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing),
                                v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if draw_funcs and n > 0:
                txt = "$f(X_{%s%s})$" % (n + 1, m + 1)
            else:
                txt = "$X_{%s%s}$" % (n + 1, m + 1)
            t = ax.text(circle.center[0] - 0.02, circle.center[1],
                        txt)
            t.set_zorder(10)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1],
                                                     layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing*(layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(1, layer_size_a + 1):
            for o in range(1, layer_size_b + 1):
                l_x_pos1 = n*h_spacing + left
                l_x_pos2 = (n + 1) * h_spacing + left
                l_y_pos1 = layer_top_a - (m - 1) * v_spacing
                l_y_pos2 = layer_top_b - (o - 1) * v_spacing
                line = plt.Line2D([l_x_pos1, l_x_pos2],
                                  [l_y_pos1, l_y_pos2], c='k')
                ax.add_artist(line)
                if draw_weights:
                    w_x_pos = l_x_pos2 - (v_spacing / 4.) * 1.5
                    w_slope = (l_y_pos2 - l_y_pos1) / (l_x_pos2 - l_x_pos1)
                    w_y_pos = l_y_pos1 + (l_x_pos2 - l_x_pos1) * w_slope * 0.3
                    t = ax.text(w_x_pos,
                                w_y_pos,
                                '$w^{%s}_{%s%s}$' % (n + 2, m, o))
                    t.set_zorder(10)

    return ax
