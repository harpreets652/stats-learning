import numpy as np
import random
import math


class SMO:
    """
    class that optimizes SVM model
    - simplified version of SMO
    http://cs229.stanford.edu/materials/smo.pdf
    """

    MAX_ITERATIONS = 10000

    def __init__(self, x, y, kernel, kernel_config, c, tol, max_iter):
        """
        smo algorithm to find lagrange multipliers, w, and b

        :param x: training data
        :param y: training data labels
        :param kernel: function
        :param kernel_config: dictionary parameters to the kernel function
        :param c: regularization parameter
        :param tol: tolerance value
        :param max_iter: max number of iterations over multipliers without changing
        """

        self._x = x
        self._y = y
        self._kernel = kernel
        self._kernel_config = kernel_config
        self._c = c
        self._tol = tol
        self._max_iter = max_iter

        self._alpha_vec = np.zeros(x.shape[0])
        self._b = 0
        self._w_vec = np.array([])

        return

    def execute(self):
        """
        smo algorithm to find lagrange multipliers, w, and b
        """

        num_iterations = 0
        iter_counter = 0
        while num_iterations < self._max_iter and iter_counter < SMO.MAX_ITERATIONS:
            num_changed_alphas = 0
            for i in range(self._x.shape[0]):
                if self._evaluate_x(i):
                    num_changed_alphas += 1

            iter_counter += 1
            if num_changed_alphas == 0:
                num_iterations += 1
            else:
                num_iterations = 0
            print(f"iteration: {iter_counter}, num_iterations: {num_iterations}")

        return self

    def _evaluate_x(self, x_idx):
        target_y = self._y[x_idx]
        alpha_i = self._alpha_vec[x_idx]
        error_i = self._svm_compute(x_idx) - target_y

        r_2 = error_i * target_y
        if (r_2 < -self._tol and alpha_i < self._c) or (r_2 > self._tol and alpha_i > 0):
            x_idx_2 = self._get_rand_index_excluding(x_idx)
            return self._optimize_step(x_idx, x_idx_2)

        return False

    def _get_rand_index_excluding(self, exclude_idx):
        while True:
            rand_idx = random.randint(0, self._x.shape[0] - 1)
            if rand_idx != exclude_idx:
                return rand_idx

    def _optimize_step(self, x_i, x_j):
        error_i = self._svm_compute(x_i) - self._y[x_i]
        error_j = self._svm_compute(x_j) - self._y[x_j]

        alpha_i_old = self._alpha_vec[x_i]
        alpha_j_old = self._alpha_vec[x_j]

        if self._y[x_i] == self._y[x_j]:
            lower = max(0, alpha_i_old + alpha_j_old - self._c)
            upper = min(self._c, alpha_i_old + alpha_j_old)
        else:
            lower = max(0, alpha_j_old - alpha_i_old)
            upper = min(self._c, self._c + alpha_j_old - alpha_i_old)

        if lower == upper:
            return False

        kij = self._kernel(self._x[x_i], self._x[x_j], self._kernel_config)
        kii = self._kernel(self._x[x_i], self._x[x_i], self._kernel_config)
        kjj = self._kernel(self._x[x_j], self._x[x_j], self._kernel_config)
        eta = 2 * kij - kii - kjj

        if eta >= 0:
            return False

        alpha_j_temp = alpha_j_old - ((self._y[x_j] * (error_i - error_j)) / eta)
        if alpha_j_temp > upper:
            self._alpha_vec[x_j] = upper
        elif alpha_j_temp < lower:
            self._alpha_vec[x_j] = lower
        else:
            self._alpha_vec[x_j] = alpha_j_temp

        if math.fabs(self._alpha_vec[x_j] - alpha_j_old) < 10e-5:
            return False

        self._alpha_vec[x_i] = alpha_i_old + self._y[x_i] * self._y[x_j] * (alpha_j_old - self._alpha_vec[x_j])

        b_1 = self._b - error_i - self._y[x_i] * (self._alpha_vec[x_i] - alpha_i_old) * kii - self._y[x_j] * \
              (self._alpha_vec[x_j] - alpha_j_old) * kij

        b_2 = self._b - error_j - self._y[x_i] * (self._alpha_vec[x_i] - alpha_i_old) * kij - self._y[x_j] * \
              (self._alpha_vec[x_j] - alpha_j_old) * kjj

        if 0 < self._alpha_vec[x_i] < self._c:
            self._b = b_1
        elif 0 < self._alpha_vec[x_j] < self._c:
            self._b = b_2
        else:
            self._b = (b_1 + b_2) / 2

        return True

    def _svm_compute(self, x_idx):
        svm_sum = 0
        for k in range(self._x.shape[0]):
            svm_sum += self._alpha_vec[k] * self._y[k] * self._kernel(self._x[k], self._x[x_idx], self._kernel_config)

        return svm_sum + self._b

    def get_model(self):
        return self._alpha_vec, self._b
