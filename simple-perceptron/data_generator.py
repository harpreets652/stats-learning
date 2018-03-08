import sklearn.datasets as ds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def generate_linearly_separated_data(num_samples=100, num_of_features=3, class_separability=1.0):
    x, y = ds.make_classification(n_samples=num_samples,
                                  n_features=num_of_features,
                                  n_informative=num_of_features,
                                  n_redundant=0,
                                  n_clusters_per_class=1,
                                  class_sep=class_separability,
                                  flip_y=0,
                                  shift=None)

    return x, y


def visualize_3d_data(x, y, plane_eq, plane_x_range, plane_y_range):
    """
    :param x: 3D data points matrix
    :param y: class labels
    :param plane_eq: [d, x, y, z] where ax + by + cz + d = 0
    :param plane_x_range: range of x mesh
    :param plane_y_range: range of x mesh
    :return:
    """
    colors = ['r', 'b']
    style = ['.', '+']

    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')

    for i in range(x.shape[0]):
        ax.scatter(x[i, 0], x[i, 1], x[i, 2], color=colors[y[i]], marker=style[y[i]], s=40)

    if plane_eq.size:
        plane = figure.gca(projection='3d')
        xx, yy = np.meshgrid(range(plane_x_range[0], plane_x_range[1]), range(plane_y_range[0], plane_y_range[1]))
        z = (-plane_eq[1] * xx - plane_eq[2] * yy - plane_eq[0]) * 1. / plane_eq[3]
        plane.plot_surface(xx, yy, z, alpha=0.5)

    plt.show()

    return


def save_data(output_file, x, y):
    y_vec = np.reshape(y, (y.shape[0], 1))
    data = np.append(y_vec, x, 1)

    np.savetxt(output_file, data, fmt=['%i', '%.5f', '%.5f', '%.5f'])
    return


def main():
    x, y = generate_linearly_separated_data(class_separability=2)
    visualize_3d_data(x, y, np.array(()), (), ())

    output_file = "/Users/harpreetsingh/github/stats-learning/simple-perceptron/resources/data_sep_2.txt"
    save_data(output_file, x, y)

    return


if __name__ == '__main__':
    main()
