import numpy as np


def sigmoid_forward(x, w_mat, b_vec):
    """
    single forward pass through a layer; using sigmoid as activation function

    :param x: numpy array, matrix or vector of inputs (each row represents one input)
    :param w_mat: numpy array, matrix of weights (each column represents a unit)
    :param b_vec: numpy array, vector of biases (column vector)
    :return: numpy array, result
    """

    z = np.dot(x, w_mat) + b_vec

    z_exp = np.exp(-1 * z)

    return 1 / (1 + z_exp)


def sigmoid_derivative(output_vec):
    """
    derivative of sigmoid
    """
    derivative = np.zeros((output_vec.shape[0], output_vec.shape[0]))

    for i, val in enumerate(output_vec):
        derivative[i][i] = val * (1 - val)

    return derivative


def sigmoid_loss_derivative(prediction_vec, target_label):
    target_vec = np.zeros((prediction_vec.shape[0]))
    target_vec[target_label] = 1

    error_vec = prediction_vec - target_vec
    loss = 0.5 * np.sum(np.square(error_vec))

    return error_vec, loss


def sigmoid_prediction(forward_prop_vec):
    """
    :param forward_prop_vec:
    :return: pass-through function
    """
    return forward_prop_vec


def relu_forward(x, w_mat, b_vec):
    """
    rectified linear unit forward pass

    :param x: numpy array, matrix or vector of inputs (each row represents one input)
    :param w_mat: numpy array, matrix of weights (each column represents a unit)
    :param b_vec: numpy array, vector of biases (column vector)
    :return: numpy array, result
    """

    z = np.dot(x, w_mat) + b_vec

    # (z >= 0) conditional is either 0 or 1
    out = z * (z >= 0)

    return out


def relu_derivative(output_vec):
    """
    relu derivative
    """
    derivative = np.zeros((output_vec.shape[0], output_vec.shape[0]))

    for i, val in enumerate(output_vec):
        derivative[i][i] = (val >= 0)

    return derivative


def relu_loss_derivative(prediction_vec, target_label):
    probabilities = softmax(prediction_vec)
    # first compute the loss
    loss = - np.log(probabilities[target_label])

    # then compute the derivative vector
    probabilities[target_label] -= 1

    return probabilities, loss


def relu_prediction(forward_prop_vec):
    """
    use softmax to compute probabilities
    :param forward_prop_vec:
    :return: probabilities vector
    """
    return softmax(forward_prop_vec)


def softmax(score_vec):
    """
    implements softmax
    """
    exp_output = np.exp(score_vec)
    probs = exp_output / np.sum(exp_output)

    return probs
