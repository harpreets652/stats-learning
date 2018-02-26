import numpy as np


def gradient_ascent(x, y, beta_init, alpha, num_iterations, calculate_error_history=False):
    """
    :param calculate_error_history: keep error history over iterations
    :param x: NxM+1
    :param y: Nx1
    :param beta_init: M+1
    :param alpha: scalar
    :param num_iterations: scalar
    :return: _beta[M+1], _error_history[iteration][RSS]
    """
    # convert y to a column vector
    y_vec = np.reshape(y, (y.shape[0], 1))
    beta = beta_init
    error_history = np.zeros((num_iterations, 2))

    for i in range(0, num_iterations):
        print("iteration: ", i)

        # p = [Nx1]
        predict_vec = predict_sigmoid(x, beta)

        # (y[Nx1] - p[Nx1]) * x[NxM+1] = grad_sum[NxM+1]
        intermediate_grad_mat = np.multiply((y_vec - predict_vec), x)

        # sum over rows(reshaped to column vector): grad[M+1x1]
        grad_sum = np.sum(intermediate_grad_mat, axis=0)
        grad_vec = 1 / x.shape[0] * np.reshape(grad_sum, (grad_sum.shape[0], 1))

        beta = beta + alpha * grad_vec

        if calculate_error_history:
            error_history[i][0] = i
            error_history[i][1] = compute_error(x, y_vec, beta)

    return beta, error_history


def compute_error(x, y, beta):
    predicted_outputs = predict_sigmoid(x, beta)

    return 1 / x.shape[0] * (np.sum(np.multiply(y, np.log(predicted_outputs)) +
                                    np.multiply((1 - y), np.log(1 - predicted_outputs))))


def predict_sigmoid(x, beta):
    """
    :param x: training matrix/vector [N x M+1]
    :param beta: parameter vector [M+1 x 1]
    :return: prediction vector/scalar [N x 1]
    """
    z = np.dot(x, beta)
    z_exp = np.exp(z)

    return z_exp / (1 + z_exp)
