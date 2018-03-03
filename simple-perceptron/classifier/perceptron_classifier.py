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

        y_vec = np.reshape(labels, (labels.shape[0], 1))
        x_mat = np.insert(features, 0, 1, axis=1)

        self._weights = PerceptronClassifier.perceptron_learning_algorithm(x_mat,
                                                                           y_vec,
                                                                           5000,
                                                                           int(x_mat.shape[0] * 0.75))

        return

    def get_weights(self):
        return self._weights

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

            prediction = PerceptronClassifier.sign_function(x_i, weights, 0)

            if y_i != prediction:
                num_iter_without_adjustments = 0
                weights += x_i if y_i else -x_i
            else:
                num_iter_without_adjustments += 1

            num_iterations += 1

        return weights

    @staticmethod
    def sign_function(x, weights, threshold):
        """
        :param threshold: sign function threshold value
        :param x: feature vector
        :param weights: weight vector
        :return: true if x*w >= 0, false otherwise
        """

        x_dot_w = np.dot(x, weights)

        return 1 if x_dot_w >= threshold else 0
