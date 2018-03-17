import numpy as np
from enum import Enum


class Solver(object):
    class LogLevel(Enum):
        NONE = 0
        INFO = 1
        VERBOSE = 2

    def __init__(self, net_model, train_data, **kwargs):
        """
        :param net_model: neural network model
        :param train_data: a dictionary with the training vectors and labels; shuffled
        :param kwargs:
            - optimization: dict - {"type": "sgd | sgd_m", "learn_rate": float, "momentum": float (if sgd_m)}
            - num_gen: number of generations to train the network
            - gradient_update_online: online(1) vs offline(0)
            - log_level: :class:'Solver.LogLevel'
        """
        self._model = net_model
        self._X_train = train_data["x_train"]
        self._Y_train = train_data["y_train"]
        self._X_test = train_data["x_test"] if "x_test" in train_data else np.array([])
        self._Y_test = train_data["y_test"] if "y_test" in train_data else np.array([])

        optimization = kwargs.pop("optimization", {"type": "sgd_m", "learn_rate": 0.01, "momentum": 0.5})

        self._optimization_type = optimization.get("type", "sgd")
        self._learning_rate = optimization.get("learning_rate", 0.001)

        if self._optimization_type == "sgd_m":
            self._momentum = optimization.get("momentum", 0.9)
            self._momentum_cache = {}

        self._num_generations = kwargs.pop("num_gen", 10)
        self._gradient_update_online = kwargs.pop("gradient_update_online", 1)
        self._log_level = kwargs.pop("log_level", Solver.LogLevel.NONE)
        self._loss_history = []
        self._test_history = []

        return

    def train(self):
        """
        Train the neural network
        """

        if self._log_level.value >= Solver.LogLevel.INFO.value:
            print("number of generations: ", self._num_generations)
            print("optimization type: ", "Online" if self._gradient_update_online else "Offline")
            print("Learning Rate: ", self._learning_rate)

        for i in range(self._num_generations):
            if self._log_level.value >= Solver.LogLevel.INFO.value:
                print("generation ", i)

            if self._gradient_update_online:
                self._step_online(i)
            else:
                self._step_offline(i)

            if self._X_test.size:
                if self._log_level.value >= Solver.LogLevel.VERBOSE.value:
                    print("---->running test set")
                self._run_test(i)

        return self._model

    def get_loss_history(self):
        return self._loss_history

    def get_test_accuracy_history(self):
        return self._test_history

    def _run_test(self, gen):
        confusion_matrix = np.zeros((10, 10))

        for i in range(self._X_test.shape[0]):
            x_i = self._X_test[i]
            y_i = self._Y_test[i]

            predicted_class = self._model.classify(x_i)
            confusion_matrix[y_i][predicted_class] += 1

        total_test_data = np.sum(confusion_matrix)
        accuracy = np.round(np.trace(confusion_matrix) / total_test_data, 5)
        self._test_history.append((gen, accuracy))

        if self._log_level.value >= Solver.LogLevel.INFO.value:
            print("accuracy: \n", accuracy)
        if self._log_level.value >= Solver.LogLevel.VERBOSE.value:
            print("confusion matrix: \n", confusion_matrix)
            print("---->test set complete")

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

            if self._optimization_type == "sgd":
                self._model.network[key] = model - self._learning_rate * update_grad
            elif self._optimization_type == "sgd_m":
                vel = self._momentum_cache.get(key, np.zeros_like(model))

                vel = self._momentum * vel - self._learning_rate * update_grad
                self._momentum_cache[key] = vel
                self._model.network[key] = model + vel

        return
