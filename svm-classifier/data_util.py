import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt


def generate_linearly_separated_data(num_samples=100, num_of_features=3, class_separability=1.0):
    x, y_temp = ds.make_classification(n_samples=num_samples,
                                       n_features=num_of_features,
                                       n_informative=num_of_features,
                                       n_redundant=0,
                                       n_clusters_per_class=2,
                                       class_sep=class_separability,
                                       flip_y=0,
                                       shift=None)

    y = np.array([i if i == 1 else -1 for i in y_temp])

    return x, y


def visualize_2d_data(x, y, svm_classifier=None, show_decision_boundary=False):
    colors = {1: 'r', -1: 'b'}
    style = {-1: '.', 1: '+'}

    for i in range(x.shape[0]):
        plt.scatter(x[i, 0], x[i, 1], color=colors[y[i]], marker=style[y[i]], s=40)

    if show_decision_boundary:
        g = np.linspace(-4, 4, 4)
        f_x = []
        for i in g:
            f_x_i = svm_classifier._svm_function(i)
            f_x.append(f_x_i)

        f_x_arr = np.array(f_x)
        plt.plot(g, f_x_arr, linestyle='-')

    plt.show()

    return
