import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt


def generate_linearly_separated_data(num_samples=100, num_of_features=3, class_separability=1.0):
    x, y_temp = ds.make_classification(n_samples=num_samples,
                                       n_features=num_of_features,
                                       n_informative=num_of_features - 1,
                                       n_redundant=0,
                                       n_clusters_per_class=1,
                                       class_sep=class_separability,
                                       flip_y=0,
                                       shift=None)

    y = np.array([i if i == 1 else -1 for i in y_temp])

    return x, y


def visualize_2d_data(x, y, svm_classifier=None, show_decision_boundary=False):
    colors = {1: 'r', -1: 'b'}
    style = {-1: '.', 1: '+'}

    sv_keys = {} if svm_classifier is None else svm_classifier.get_support_vectors().keys()

    for i in range(x.shape[0]):
        plt.scatter(x[i, 0], x[i, 1], color=colors[y[i]], marker=style[y[i]], s=40)

        if i in sv_keys:
            plt.scatter(x[i, 0], x[i, 1], facecolors='none', edgecolors='r', s=100)

    if show_decision_boundary:
        min_vals = np.amin(x, axis=0)
        max_vals = np.amax(x, axis=0)
        xx, yy = np.meshgrid(np.linspace(min_vals[0] - 0.1, max_vals[0] + 0.1, 100),
                             np.linspace(min_vals[1] - 0.1, max_vals[1] + 0.1, 200))
        pairs = np.c_[xx.ravel(), yy.ravel()]

        z = []
        for i in pairs:
            f_x = svm_classifier.decision_function(i)
            z.append(f_x)

        z_arr = np.array(z)
        z_arr = np.reshape(z_arr, xx.shape)
        plt.contourf(xx, yy, z_arr, cmap=plt.cm.PuBu, alpha=0.5)
        plt.contour(xx, yy, z_arr, levels=[0], linewidths=2, colors='darkred')

    plt.show()

    return
