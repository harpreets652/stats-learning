import numpy as np
from random import shuffle

import classifier.gradient_ascent as ga


class LogisticRegClassifier:
    """
    Logistic regression classifier using:
        * sigmoid prediction function - p(x; b) = e^{bTx} / 1 + e^{bTx}
        * gradient ascent optimization - loop {b = b + alpha*p(x; b)}
    """

    # todo: try other sigmoid(might need gradient descent for this), variable learning rates
    def __init__(self, class_a_data, class_b_data, learning_rate, num_ga_iterations):
        self._class_1_label = class_a_data[0]
        self._class_0_label = class_b_data[0]

        data_1 = self.load_training_data_with_label(1, class_a_data[1])
        data_0 = self.load_training_data_with_label(0, class_b_data[1])
        training_data = data_1 + data_0
        shuffle(training_data)

        labels = []
        train_data = []
        for d in training_data:
            labels.append(d[0])
            train_data.append(d[1])

        y_vec = np.array(labels)
        x_mat = np.array(train_data)

        x_mat = np.insert(x_mat, 0, 1, axis=1)
        self.beta = np.zeros((x_mat.shape[1], 1))

        self.beta = ga.gradient_ascent(x_mat, y_vec, self.beta, learning_rate, num_ga_iterations)

        return

    @staticmethod
    def load_training_data_with_label(class_label, class_file_name):
        training_data = []
        with open(class_file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                training_data.append((class_label, np.array(data)))

        return training_data
