import numpy as np
import matplotlib.pyplot as plt
import data_util as du
from random import shuffle
import classifier.neural_network as nn
import classifier.solver as solver

training_data_files = {0: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train0.txt",
                       1: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train1.txt",
                       2: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train2.txt",
                       3: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train3.txt",
                       4: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train4.txt",
                       5: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train5.txt",
                       6: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train6.txt",
                       7: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train7.txt",
                       8: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train8.txt",
                       9: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train9.txt"}

output_file_directory = "/Users/harpreetsingh/github/stats-learning/neural-network/results/"


def get_training_data():
    data_with_labels = []
    pca_transform = {}
    for label, file in training_data_files.items():
        class_data = du.load_class_training_data(file)

        reduced_dim, mean, eigen = du.run_pca(class_data, 16)
        pca_transform[label] = {"mean": mean, "eigen": eigen}

        for i in range(reduced_dim.shape[0]):
            data_with_labels.append((label, reduced_dim[i]))

    shuffle(data_with_labels)

    labels = []
    train_data = []
    for d in data_with_labels:
        labels.append(d[0])
        train_data.append(d[1])

    return np.array(train_data), np.array(labels), pca_transform


def run_test_data(test_data_set, classifier):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((10, 10))

    counter = 0
    for p in test_data_set:
        predicted_class = classifier.classify(p[1])
        confusion_matrix[p[0]][predicted_class] += 1
        counter += 1
        print("count: ", counter)

    return confusion_matrix


def visualize_grad_history(gradient_results):
    plt.plot(gradient_results[:, 0], gradient_results[:, 1], 'r-', linewidth=0.5)

    plt.title("Gradient Descent Progress")
    plt.xlabel("Generation")
    plt.ylabel("Average error")

    plt.show()

    return


def main():
    # setup training data
    x_train, y_train, pca_transform = get_training_data()

    # train neural network
    network_model = nn.FullyConnectedNetwork(16, [18, 12], 10)

    # run validation set on model and print confusion matrix
    network_solver = solver.Solver(network_model,
                                   {"x_train": x_train, "y_train": y_train},
                                   learn_rate=0.1,
                                   num_gen=12)

    network_solver.train()

    visualize_grad_history(np.array(network_solver.get_loss_history()))

    return


if __name__ == '__main__':
    main()
