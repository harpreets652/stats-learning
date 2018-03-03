import numpy as np
import classifier.perceptron_pla_classifier as pc_pla
import classifier.perceptron_gd_classifier as pc_gd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

training_data = "/Users/harpreetsingh/github/stats-learning/simple-perceptron/resources/data.txt"


def load_training_data(file_name):
    labels = []
    features = []
    with open(file_name, 'r') as file:
        for row in file:
            data_string = row.strip().split()
            data = []
            class_label = -1
            for i in range(len(data_string)):
                if i == 0:
                    class_label = int(data_string[i])
                    continue
                data.append(float(data_string[i]))

            labels.append(class_label)
            features.append(np.array(data))

    return np.array(features), np.array(labels)


def visualize_data(x_mat, y_vec, plane_eq):
    colors = ['r', 'b']

    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    plane = figure.gca(projection='3d')

    for i in range(x_mat.shape[0]):
        ax.scatter(x_mat[i, 0], x_mat[i, 1], x_mat[i, 2], color=colors[y_vec[i]])

    xx, yy = np.meshgrid(range(-7, 2), range(-5, 5))
    z = (-plane_eq[1] * xx - plane_eq[2] * yy - plane_eq[0]) * 1. / plane_eq[3]

    plane.plot_surface(xx, yy, z, alpha=0.5)
    plt.show()

    return


def run_pla_classifier(x, y):
    classifier = pc_pla.PerceptronPLAClassifier(x, y)
    print("weights: ", classifier.get_weights())

    visualize_data(x, y, classifier.get_weights())

    return


def run_gd_classifier(x, y):
    classifier = pc_gd.PerceptronGDClassifier(x, y)
    print("weights: ", classifier.get_weights())

    visualize_data(x, y, classifier.get_weights())

    return


def main():
    x, y = load_training_data(training_data)

    # run_pla_classifier(x, y)
    run_gd_classifier(x, y)

    return


if __name__ == '__main__':
    main()
