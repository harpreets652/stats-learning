import numpy as np
import random


class PerceptronPLAClassifier:
    def __init__(self, features, labels):
        """
        :param features: n-dimensional feature vector matrix
        :param labels: y = {0, 1}
        """
        if not features.size or not labels.size:
            raise RuntimeError("features and labels inputs may not be empty ")

        y_vec = np.reshape(labels, (labels.shape[0], 1))
        x_mat = np.insert(features, 0, 1, axis=1)

        self._weights = PerceptronPLAClassifier.perceptron_learning_algorithm(x_mat,
                                                                              y_vec,
                                                                              5000,
                                                                              int(x_mat.shape[0] * 1.00))

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
        pos_data, neg_data = [], []
        for i in range(x_mat.shape[0]):
            if y_vec[i]:
                pos_data.append(x_mat[i])
            else:
                neg_data.append(x_mat[i])

        weights_column_vec = np.array(pos_data).mean(axis=0) - np.array(neg_data).mean(axis=0)
        weights = np.reshape(weights_column_vec, (weights_column_vec.shape[0], 1))
        weights = np.zeros((x_mat.shape[1], 1))

        num_iter_without_adjustments = 0
        num_iterations = 0
        while True:
            if num_iterations >= max_iterations or \
                    num_iter_without_adjustments >= max_iter_without_adjustment:
                print(f"number of iterations: {num_iterations}, "
                      f"num_iterations_without_adjustments: {num_iter_without_adjustments}")
                break

            rand_x_index = random.randint(0, x_mat.shape[0] - 1)
            x_i_row = x_mat[rand_x_index]
            x_i = np.reshape(x_i_row, (x_i_row.shape[0], 1))

            y_i = y_vec[rand_x_index]

            prediction = PerceptronPLAClassifier.perceptron_forward_pass(x_i, weights, 0)

            if y_i != prediction:
                num_iter_without_adjustments = 0
                weights = np.add(weights, (x_i if y_i else -1 * x_i))
            else:
                num_iter_without_adjustments += 1

            num_iterations += 1

        return weights

    @staticmethod
    def perceptron_forward_pass(x, weights, threshold):
        """
        :param threshold: sign function threshold value
        :param x: feature vector
        :param weights: weight vector
        :return: true if x*w >= 0, false otherwise
        """

        x_dot_w = np.dot(x.T, weights)

        return 1 if x_dot_w >= threshold else 0
