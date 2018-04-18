import numpy as np
import random
import math


class SMO:
    """
    class that optimizes SVM model
    - simplified version of SMO
    http://cs229.stanford.edu/materials/smo.pdf
    """

    def __init__(self, x, y, kernel, kernel_config, c, tol, eps):
        """
        smo algorithm to find lagrange multipliers, w, and b

        :param x: training data
        :param y: training data labels
        :param kernel: function
        :param kernel_config: dictionary parameters to the kernel function
        :param c: regularization parameter
        :param tol: error tolerance value
        :param eps: alpha tolerance value
        """

        self._x = x
        self._y = y
        self._kernel = kernel
        self._kernel_config = kernel_config
        self._c = c
        self._tol = tol
        self._eps = eps

        self._alpha_vec = np.zeros(x.shape[0])
        self._b = 0

        return

    def execute(self):
        """
        smo algorithm to find lagrange multipliers, w, and b
        """

        num_changed = 0
        examine_all = True

        iter_counter = 0
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self._x.shape[0]):
                    if self._evaluate_step(i):
                        num_changed += 1
            else:
                for i, alpha in enumerate(self._alpha_vec):
                    if alpha != 0 and alpha != self._c:
                        if self._evaluate_step(i):
                            num_changed += 1

            iter_counter += 1
            print(f"Iteration: {iter_counter}, examine all: {examine_all}, number changed: {num_changed}")
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return self

    def _evaluate_step(self, x_idx_2):
        target_y_2 = self._y[x_idx_2]
        alpha_2 = self._alpha_vec[x_idx_2]
        error_2 = self._svm_compute(x_idx_2) - target_y_2

        r_2 = error_2 * target_y_2
        if (r_2 < -self._tol and alpha_2 < self._c) or (r_2 > self._tol and alpha_2 > 0):
            # note~ omitting first conditional in evaluation

            # loop over all non-zero and non-c alpha starting from random point
            non_z_c_idx = np.array([i for i, val in enumerate(self._alpha_vec) if val != 0 and val != self._c])
            if non_z_c_idx.size:
                for x_idx_1 in np.roll(non_z_c_idx, random.randint(0, non_z_c_idx.shape[0] - 1)):
                    if self._optimize_step(x_idx_1, x_idx_2):
                        return True

            # loop over all alphas starting from random point
            for x_idx_1 in np.roll(np.arange(0, self._alpha_vec.shape[0]),
                                   random.randint(0, self._alpha_vec.shape[0]) - 1):
                if self._optimize_step(x_idx_1, x_idx_2):
                    return True

        return False

    def _optimize_step(self, x_i, x_j):
        if x_i == x_j:
            return False

        alpha_i_old = self._alpha_vec[x_i]
        alpha_j_old = self._alpha_vec[x_j]

        error_i = self._svm_compute(x_i) - self._y[x_i]
        error_j = self._svm_compute(x_j) - self._y[x_j]

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

        if eta < 0:
            alpha_j_temp = alpha_j_old - ((self._y[x_j] * (error_i - error_j)) / eta)
            if alpha_j_temp > upper:
                self._alpha_vec[x_j] = upper
            elif alpha_j_temp < lower:
                self._alpha_vec[x_j] = lower
            else:
                self._alpha_vec[x_j] = alpha_j_temp
        else:
            alpha_vec_temp = self._alpha_vec.copy()

            alpha_vec_temp[x_j] = lower
            low_obj = self._svm_compute(x_j)
            alpha_vec_temp[x_j] = upper
            hi_obj = self._svm_compute(x_j)

            if low_obj > (hi_obj + self._eps):
                self._alpha_vec[x_j] = lower
            elif low_obj < (hi_obj - self._eps):
                self._alpha_vec[x_j] = upper
            else:
                self._alpha_vec[x_j] = alpha_j_old

        if self._alpha_vec[x_j] < 1e-8:
            self._alpha_vec[x_j] = 0
        elif self._alpha_vec[x_j] > (self._c - 1e-8):
            self._alpha_vec[x_j] = self._c

        if math.fabs(self._alpha_vec[x_j] - alpha_j_old) < self._eps * (self._alpha_vec[x_j] + alpha_j_old + self._eps):
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
