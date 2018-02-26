import numpy as np


def gradient_ascent(x, y, beta_init, alpha, num_iterations, calculate_error_history):
    """
    :param calculate_error_history: keep error history over iterations
    :param x: NxM+1
    :param y: Nx1
    :param beta_init: M+1
    :param alpha: scalar
    :param num_iterations: scalar
    :return: beta[M+1], error_history[iteration][RSS]
    """

    beta = beta_init
    error_history = np.zeros((num_iterations, 2))
    for i in range(0, num_iterations):
        # p = [Nx1]
        predict_vec = predict_sigmoid(x, beta)

        # (y[Nx1] - p[Nx1]) * x[NxM+1] = grad_sum[NxM+1]
        intermediate_grad_mat = np.multiply((y - predict_vec), x)

        # sum over rows: grad[1xM+1]
        grad_vec = np.sum(intermediate_grad_mat, axis=0)

        beta = beta + alpha * grad_vec

        if calculate_error_history:
            error_history[i] = compute_error(x, y, beta)

    return beta, error_history


def compute_error(x, y, beta):
    predicted_outputs = predict_sigmoid(x, beta)

    return np.sum(np.square(y - predicted_outputs))


def predict_sigmoid(x, beta):
    """
    :param x: training matrix [N x M+1]
    :param beta: parameter vector [M+1 x 1]
    :return: prediction vector [N x 1]
    """
    z = np.dot(x, beta)
    z_exp = np.exp(z)

    return z_exp / (1 + z_exp)