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


def visualize_history(results, title, x_label, y_label):
    plt.plot(results[:, 0], results[:, 1], 'r-', linewidth=0.5)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.show()

    return


def main():
    # configuration
    input_size = 16
    learning_rate = 0.001
    num_gen = 10000
    percent_test_set = 0.20
    online_update = 0

    all_x_train, all_y_train, pca_transform = du.get_digits_training_data(num_dimensions=input_size)
    test_data = du.get_test_data(pca_transform)

    num_val = int(all_x_train.shape[0] * percent_test_set)

    x_train_test = all_x_train[0:num_val]
    y_train_test = all_y_train[0:num_val]
    x_train = all_x_train[num_val:]
    y_train = all_y_train[num_val:]

    # train neural network
    network_model = nn.FullyConnectedNetwork(input_size, [16, 16], 10, activation="relu")

    network_solver = solver.Solver(network_model,
                                   {"x_train": x_train, "y_train": y_train,
                                    "x_test": x_train_test, "y_test": y_train_test},
                                   learn_rate=learning_rate,
                                   num_gen=num_gen,
                                   gradient_update_online=online_update,
                                   log_level=solver.Solver.LogLevel.INFO)

    network_model = network_solver.train()

    # visualize results
    visualize_history(np.array(network_solver.get_loss_history()),
                      "Gradient Descent Progress",
                      "Generation",
                      "Average error")

    visualize_history(np.array(network_solver.get_test_accuracy_history()),
                      "Test Set Accuracy history",
                      "Generation",
                      "Accuracy")

    # run test/validation set
    run_test_data(test_data, network_model)

    return


if __name__ == '__main__':
    main()
