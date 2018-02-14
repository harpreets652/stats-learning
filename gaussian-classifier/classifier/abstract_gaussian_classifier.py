import numpy as np


class AbstractGaussianClassifier:
    """
    Abstract Gaussian classifier providing an interface and common methods
    """

    def __init__(self, reg_param):
        self.regularization_param = reg_param
        self.class_models = {}
        return

    def add_class(self, label, data_file):
        raise NotImplementedError("Method not implemented.")
        pass

    def classify(self, new_data_point):
        raise NotImplementedError("Method not implemented.")
        pass

    def _load_training_data(self, file_name):
        features = []

        with open(file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                features.append(data)

        return np.array(features)

    def _calc_probability(self, covariance, centered_point):
        """

        :rtype:
        """
        denominator_1 = ((2 * np.pi) ** (centered_point.shape[0] / 2))
        denominator_2 = np.sqrt(np.linalg.det(covariance))
        denominator = denominator_1 * denominator_2

        exp_term = centered_point.dot(np.linalg.inv(covariance)).dot(centered_point) * (-0.5)
        exp_val = np.exp(exp_term)

        return (1 / denominator) * exp_val
