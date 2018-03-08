import numpy as np
import classifier.perceptron_pla_classifier as pc_pla
import classifier.perceptron_gd_classifier as pc_gd
import data_generator as dg
import matplotlib.pyplot as plt

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
    x_range = [-5, 5]
    y_range = [-5, 5]
    dg.visualize_3d_data(x_mat, y_vec, plane_eq, x_range, y_range)

    return


def visualize_runtimes(runtime_data):
    plt.scatter(runtime_data[:, 0],
                runtime_data[:, 1],
                facecolors='none',
                linewidths=0.5,
                edgecolors='b',
                s=10)

    plt.xlabel("Sample Size")
    plt.ylabel("Runtime (ms)")

    plt.show()

    return


def run_pla_classifier():
    runtime_data = []
    num_of_data = np.arange(100, 300, 10)

    for sample_size in num_of_data:
        x, y = dg.generate_linearly_separated_data(num_samples=sample_size)

        classifier = pc_pla.PerceptronPLAClassifier(x, y)
        runtime_data.append((sample_size, classifier.get_runtime()))

        print("weights: ", classifier.get_weights())
        # visualize_data(x, y, classifier.get_weights())

    visualize_runtimes(np.array(runtime_data))

    return


def run_gd_classifier():
    x, y = dg.generate_linearly_separated_data()

    classifier = pc_gd.PerceptronGDClassifier(x, y)
    print("weights: ", classifier.get_weights())

    visualize_data(x, y, classifier.get_weights())

    return


def main():
    run_pla_classifier()
    # run_gd_classifier(x, y)

    return


if __name__ == '__main__':
    main()
