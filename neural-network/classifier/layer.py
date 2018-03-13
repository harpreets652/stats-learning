import numpy as np


def sigmoid_forward_pass(x, w_mat, b_vec):
    """
    single forward pass through a layer; using sigmoid as activation function

    :param x: numpy array, matrix or vector of inputs (each row represents one input)
    :param w_mat: numpy array, matrix of weights (each row represents a unit)
    :param b_vec: numpy array, vector of biases (column vector)
    :return: numpy array, result
    """

    z = np.dot(x, w_mat) + b_vec

    z_exp = np.exp(-1 * z)

    return 1 / (1 + z_exp)


def compute_sigmoid_derivative(output_vec):
    derivative = np.zeros((output_vec.shape[0], output_vec.shape[0]))

    for i, val in enumerate(output_vec):
        derivative[i][i] = val * (1 - val)

    return derivative
