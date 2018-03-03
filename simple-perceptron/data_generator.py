import sklearn.datasets as ds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def main():
    x, y = ds.make_classification(n_features=3,
                                  n_informative=3,
                                  n_redundant=0,
                                  n_clusters_per_class=1,
                                  class_sep=3)
    colors = ['r', 'b']

    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')

    for i in range(x.shape[0]):
        ax.scatter(x[i, 0], x[i, 1], x[i, 2], color=colors[y[i]])
    plt.show()

    y_vec = np.reshape(y, (y.shape[0], 1))
    data = np.append(y_vec, x, 1)

    output_file = "/Users/harpreetsingh/github/stats-learning/simple-perceptron/resources/data.txt"
    np.savetxt(output_file, data, fmt=['%i', '%.5f', '%.5f', '%.5f'])

    return


if __name__ == '__main__':
    main()
