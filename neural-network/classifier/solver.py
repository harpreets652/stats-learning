class Solver(object):
    def __init__(self, net_model, train_data, **kwargs):
        """
        :param net_model: neural network model
        :param train_data: a dictionary with the training vectors and labels; shuffled
        :param kwargs:
            - learn_rate: learning rate for gradient descent
            - num_gen: number of generations to train the network
            - gradient_update_type: online(1) vs offline(0)
        """
        self._model = net_model
        self._X_train = train_data["x_train"]
        self._Y_train = train_data["y_train"]
        self._learning_rate = kwargs.pop("learn_rate", 0.01)
        self._num_generations = kwargs.pop("num_gen", 10)
        self._gradient_update_online = kwargs.pop("gradient_update_type", 1)
        self._loss_history = []

        return

    def train(self):
        """
        Train the neural network
        """
        for i in range(self._num_generations):
            if self._gradient_update_online:
                self._step_online(i)
            else:
                self._step_offline(i)

            print("generation ", i)

            # TODO: COMPUTE ACCURACY OF TRAINING SET

        return

    def _step_online(self, gen):
        """
        Compute online gradient update
        """

        total_loss = 0
        for i in range(self._X_train.shape[0]):
            x_i = self._X_train[i]
            y_i = self._Y_train[i]

            gradients_x_i, loss_x_i = self._model.compute_gradient(x_i, y_i)
            total_loss += loss_x_i

            self._update_model(gradients_x_i)

        self._loss_history.append((gen, total_loss))

        return

    def _step_offline(self, gen):
        """
        Compute offline gradient update
        """
        gradients, loss = self._model.compute_gradient_batch(self._X_train, self._Y_train)

        self._loss_history.append((gen, loss))

        self._update_model(gradients)

        return

    def _update_model(self, w_gradients):
        for key, model in self._model.network.items():
            update_grad = w_gradients[key]

            self._model.network[key] = model - self._learning_rate * update_grad

        return

    def get_loss_history(self):
        return self._loss_history
