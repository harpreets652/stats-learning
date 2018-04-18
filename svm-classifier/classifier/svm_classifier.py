import numpy as np
import classifier.optimizer as opt
import classifier.kernels as kernels


class SVMClassifier(object):
    """
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/smo-book.pdf
    http://cs229.stanford.edu/notes/cs229-notes3.pdf
    http://cs229.stanford.edu/materials/smo.pdf
    """

    def __init__(self, x_train, y_label, **kwargs):
        """
        initialize SVM Classifier, training data using SMO optimization algorithm

        :param x_train: 2D np array
        :param y_label: 1D corresponding labels
        :param kwargs: arguments to SMO
            - kernel: 'rbf' | 'polynomial'
            - kernel_config: 'gamma' (for rbf) | 'd' (for poly)
            - c: regularization param, slack variable
            - tol: error tolerance
            - eps: alpha tolerance
        """

        kernel_str = kwargs.pop("kernel", "rbf")
        if not hasattr(kernels, kernel_str):
            raise RuntimeError(f"{kernel_str} not found")

        self._kernel = getattr(kernels, kernel_str)
        self._kernel_config = kwargs.pop("kernel_config", {"gamma": 0.5})

        c = kwargs.pop("c", 0.01)
        tol = kwargs.pop("tol", 0.01)
        eps = kwargs.pop("eps", 0.01)
        smo = opt.SMO(x_train, y_label, self._kernel, self._kernel_config, c, tol, eps)

        alphas, bias = smo.execute().get_model()

        self._support_vectors = {}
        for i in range(alphas.shape[0]):
            if alphas[i] > 0:
                self._support_vectors[i] = {"alpha": alphas[i], "y": y_label[i], "x": x_train[i]}

        self._b = bias

        return

    def get_support_vectors(self):
        return self._support_vectors

    def classify(self, x):
        f_x = self.decision_function(x)
        return 1 if f_x >= 0 else -1

    def decision_function(self, x):
        svm_sum = 0
        for key, sv in self._support_vectors.items():
            svm_sum += sv["alpha"] * sv["y"] * self._kernel(sv["x"], x, self._kernel_config)

        return svm_sum + self._b
