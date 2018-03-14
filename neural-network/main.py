import numpy as np
import matplotlib.pyplot as plt
import classifier.neural_network as nn
import classifier.solver as solver
import data_util as du

output_file_directory = "/Users/harpreetsingh/github/stats-learning/neural-network/results/"


def run_test_data(test_data_set, classifier):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((10, 10))

    counter = 0
    for p in test_data_set:
        predicted_class = classifier.classify(p[1])
        confusion_matrix[p[0]][predicted_class] += 1
        counter += 1
        print("count: ", counter)

    print("confusion matrix: \n", confusion_matrix)
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
    x_train, y_train, pca_transform = du.get_digits_training_data(num_dimensions=64)
    test_data = du.get_test_data(pca_transform)

    # train neural network
    network_model = nn.FullyConnectedNetwork(64, [18], 10)

    # run validation set on model and print confusion matrix
    network_solver = solver.Solver(network_model,
                                   {"x_train": x_train, "y_train": y_train},
                                   learn_rate=0.0001,
                                   num_gen=40)

    network_solver.train()

    visualize_grad_history(np.array(network_solver.get_loss_history()))

    run_test_data(test_data, network_model)

    return


if __name__ == '__main__':
    main()
