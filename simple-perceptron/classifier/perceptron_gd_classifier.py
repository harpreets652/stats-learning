import numpy as np
import random


class PerceptronGDClassifier:

    def __init__(self, features, labels):
        """
        :param features: n-dimensional feature vector matrix
        :param labels: y = {0, 1}
        """
        if not features.size or not labels.size:
            raise RuntimeError("features and labels inputs may not be empty ")

        y_vec = np.reshape(labels, (labels.shape[0], 1))
        x_mat = np.insert(features, 0, 1, axis=1)

        self._weights = PerceptronGDClassifier.gradient_descent(x_mat, y_vec, 0.1, 500)

        return

    def get_weights(self):
        return self._weights

    @staticmethod
    def gradient_descent(x_mat, y_vec, learning_rate, num_iterations):
        """
        :param x_mat: extended feature vectors [mxn+1]
        :param y_vec: corresponding labels [mx1]
        :param learning_rate: GD learning rate
        :param num_iterations: GD number of iterations
        :return: trained weights
        """

        weights = np.zeros((x_mat.shape[1], 1))

        for i in range(num_iterations):
            # find prediction vector [mx1]
            output = PerceptronGDClassifier.perceptron_forward_pass(x_mat, weights)

            # compute Jacobian matrix
            jacobian = PerceptronGDClassifier.compute_jacobian(y_vec, output, x_mat)

            # compute gradient - sum of columns of Jacobian
            gradient_row_vec = np.sum(jacobian, axis=0)
            gradient = np.reshape(gradient_row_vec, (gradient_row_vec.shape[0], 1))

            # todo: weights are never updated b/c they're initialized to 0 and are included in the gradient calc.
            # adjust weights: w = w - alpha*gradient
            weights = weights - learning_rate * gradient

        return weights

    @staticmethod
    def perceptron_forward_pass(x_mat, weights_vec):
        """
        :param x_mat: raining matrix/vector [MxN]
        :param weights_vec: weights vector [Nx1]
        :return: ouptut vector/scalar [Mx1]
        """

        z = np.dot(x_mat, weights_vec)

        z_exp = np.exp(-1 * z)

        return 1 / (1 + z_exp)

    @staticmethod
    def compute_jacobian(y_vec, perceptron_output_vec, x_mat):
        """
        A = Perceptron output vector for all x_i
        X = matrix of all training examples
        Y = Corresponding target outputs

        Jacobian = (Y - A)*(A)*(1 -  A) * X

        All multiplications are element-wise

        :param y_vec: labeled classes [Mx1]
        :param perceptron_output_vec: forward pass output [Mx1]
        :param weights_vec: perceptron weights [N+1x1]
        :return: MxN+1 Jacobian matrix: partial derivatives of the Error w.r.t. weights
        """

        first_term = y_vec - perceptron_output_vec
        third_term = 1 - perceptron_output_vec

        jacobian = (first_term * perceptron_output_vec * third_term) * x_mat

        return jacobian
