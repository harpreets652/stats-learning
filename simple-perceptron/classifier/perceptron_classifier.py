import numpy as np
import random


class PerceptronClassifier:
    def __init__(self, features, labels):
        """
        :param features: n-dimensional feature vector matrix
        :param labels: y = {0, 1}
        """
        if not features or labels:
            raise RuntimeError("features and labels inputs may not be empty ")

        # adjust the shape of labels, if needed

        x = np.insert(features, 0, 1, axis=1)

        self._weights = PerceptronClassifier.perceptron_learning_algorithm(x,
                                                                           labels,
                                                                           5000,
                                                                           int(x.shape[0] * 0.75))

        return

    @staticmethod
    def perceptron_learning_algorithm(x_mat, y_vec, max_iterations, max_iter_without_adjustment):
        """
        find optimal weight values

        :param max_iter_without_adjustment: terminal condition
        :param max_iterations: terminal condition
        :param x_mat: extended feature vectors
        :param y_vec: labeled outputs
        :return: weights
        """

        weights = np.zeros((x_mat.shape[1], 1))

        num_iter_without_adjustments = 0
        num_iterations = 0
        while True:
            if num_iterations >= max_iterations or \
                    num_iter_without_adjustments >= max_iter_without_adjustment:
                break

            rand_x_index = random.randint(0, x_mat.shape[0])
            x_i = x_mat[rand_x_index]
            y_i = y_vec[rand_x_index]

            prediction = PerceptronClassifier.sign_function(x_i, weights)

            if y_i and not prediction:
                weights += x_i
                num_iter_without_adjustments = 0
            elif not y_i and prediction:
                weights -= x_i
                num_iter_without_adjustments = 0
            else:
                num_iter_without_adjustments += 1

            num_iterations += 1

        return weights

    @staticmethod
    def sign_function(x, weights):
        x_dot_w = np.dot(x, weights)

        return 1 if x_dot_w >= 0 else 0
