class Solver(object):
    def __init__(self, net_model, train_data, **kwargs):
        """
        :param net_model: neural network model
        :param train_data: a dictionary with the training vectors and labels; shuffled
        :param kwargs:
            - learn_rate: learning rate for gradient descent
            - num_gen: number of generations to train the network
        """
        self._model = net_model
        self._X_train = train_data["x_train"]
        self._Y_train = train_data["y_train"]
        self._learning_rate = kwargs.pop("learn_rate", 0.01)
        self._num_generations = kwargs.pop("num_gen", 10)
        self._loss_history = []

        # TODO: implement offline and online gradient update

        return

    def train(self):
        """
        Train the neural network
        """
        for i in range(self._num_generations):
            self._step_offline(i)
            print("generation ", i)
            # TODO: COMPUTE ACCURACY OF TRAINING SET

        return

    def _step_offline(self, gen):
        """
        Compute offline gradient and update the model

        Note:
        offline updates: accumulating gradient for each x_i and then updating weights
        online updates: update weights after each x_i
        """
        gradients, loss = self._model.compute_gradient(self._X_train, self._Y_train)
        self._loss_history.append((gen, loss.real))

        for key, model in self._model.network.items():
            update_grad = gradients[key]

            self._model.network[key] = self._model.network[key] - self._learning_rate * update_grad

        return

    def get_loss_history(self):
        return self._loss_history
