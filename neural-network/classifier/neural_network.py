import numpy as np
import classifier.layer as layer


class FullyConnectedNetwork(object):
    """
    Fully connected neural network with a configurable number of hidden dimensions
    """

    def __init__(self, input_dimensions, hidden_dimensions, num_classes):
        """
        Initialization

        :param input_dimensions: Integer, size of the input layer
        :param hidden_dimensions: Array[Integer], number of units per hidden layer
        :param num_classes: Integer, number of classes in the inputs
        """

        self.num_layers = len(hidden_dimensions) + 1
        self.network = {}

        # Weights: matrix representing weights of each unt as a column
        # Biases: array representing a bias to each unit in the layer
        # W0 = first hidden layer, W1 = second hidden layer, etc (same for biases, B)
        # Note: input vec extended with 1, resulting in just add bias in the forward prop

        input_layer_dim = input_dimensions
        for i, dim in enumerate(hidden_dimensions):
            weights_key = "W" + str(i)
            biases_key = "B" + str(i)

            self.network[weights_key] = np.random.randn(input_layer_dim, dim)
            self.network[biases_key] = np.zeros(dim)
            input_layer_dim = dim

        # output layer
        weights_key = "W" + str(self.num_layers - 1)
        biases_key = "B" + str(self.num_layers - 1)
        self.network[weights_key] = np.random.randn(input_layer_dim, num_classes)
        self.network[biases_key] = np.zeros(num_classes)

        return

    def compute_gradient(self, x_mat, y_vec):
        gradients = {}
        accumulated_loss = 0

        for i in range(x_mat.shape[0]):
            x_i = x_mat[i]
            gradients_x_i, loss_x_i = self._grad_step(x_i, y_vec[i])

            accumulated_loss += loss_x_i
            for key, grad in gradients_x_i.items():
                if key in gradients:
                    gradients[key] += grad
                else:
                    gradients[key] = grad

        total_loss = (1 / x_mat.shape[0]) * accumulated_loss
        return gradients, total_loss

    def _grad_step(self, x, y):
        """
        :param x: numpy array, vector
        :param y: scalar labeled output
        :return: gradient at each layer
        """

        # forward prop
        output_cache, prediction_vec = self._network_forward(x)

        # back propagation
        gradients = {}
        delta = {}
        der_out = {}

        # derivative w.r.t. output
        for lay, out in output_cache.items():
            der_out[lay] = layer.compute_sigmoid_derivative(out)

        output_layer_derivative = layer.compute_sigmoid_derivative(prediction_vec)

        # delta for each layer
        output_layer_error, loss = FullyConnectedNetwork._compute_error(prediction_vec, y)

        output_layer_error_col = np.reshape(output_layer_error, (output_layer_error.shape[0], 1))
        delta[self.num_layers - 1] = output_layer_derivative.dot(output_layer_error_col)
        for i in range(self.num_layers - 2, -1, -1):
            delta[i] = der_out[i].dot(self.network["W" + str(i + 1)]).dot(delta[i + 1])

        # compute gradient
        for i in range(self.num_layers):
            lay_input = x if i == 0 else output_cache[i - 1]
            lay_input_vec = np.reshape(lay_input, (1, lay_input.shape[0]))

            gradients["W" + str(i)] = (delta[i].dot(lay_input_vec)).T

            delta_row = np.reshape(delta[i], -1)
            gradients["B" + str(i)] = delta_row

        return gradients, loss

    def _network_forward(self, x):
        layer_input = x
        output_cache = {}

        # forward pass through hidden units
        for l in range(self.num_layers - 1):
            output = layer.sigmoid_forward_pass(layer_input,
                                                self.network["W" + str(l)],
                                                self.network["B" + str(l)])
            output_cache[l] = output
            layer_input = output

        prediction = layer.sigmoid_forward_pass(layer_input,
                                                self.network["W" + str(self.num_layers - 1)],
                                                self.network["B" + str(self.num_layers - 1)])

        return output_cache, prediction

    @staticmethod
    def _compute_error(prediction_vec, target_label):
        target_vec = np.zeros((prediction_vec.shape[0]))
        target_vec[target_label] = 1

        error_vec = prediction_vec - target_vec
        loss = 0.5 * np.sum(np.square(error_vec))

        return error_vec, loss

    def classify(self, x):
        cache, prediction = self._network_forward(x)
        return np.argmax(prediction)
