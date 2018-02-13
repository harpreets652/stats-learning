import numpy as np
import operator


class GaussianClassifier:
    """
    Multi-class classifier that fits a Gaussian distribution to multidimensional data
    Note: assumes Gaussian distribution
    """

    def __init__(self, num_principle_components, reg_param):
        self.num_dimensions = num_principle_components
        self.regularization_param = reg_param
        self.class_models = {}
        return

    def add_class(self, label, data_file):
        # load training data
        training_data_mat = self.__load_training_data(data_file)

        # find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = training_data_mat.mean(axis=0)

        # find the covariance; row var=False because each column represents a variable, with rows as observations
        regularization_term = np.multiply(self.regularization_param, np.eye(training_data_mat.shape[1]))
        cov_mat = np.cov(training_data_mat, rowvar=False) + regularization_term

        # add to a dictionary
        self.class_models[label] = {'mean': mean_vec, 'cov': cov_mat}

        return self

    @staticmethod
    def __load_training_data(file_name):
        features = []

        with open(file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                features.append(data)

        return np.array(features)

    def classify(self, new_data_point):
        class_probabilities = {}

        for label, model in self.class_models:
            denominator_1 = ((2 * np.pi) ** (new_data_point.shape[0] / 2))
            denominator_2 = np.sqrt(np.linalg.det(model['cov']))
            denominator = denominator_1 * denominator_2

            centered_point = new_data_point - model['mean']
            exp_term = centered_point.dot(np.linalg.inv(model['cov'])).dot(centered_point) * (-0.5)
            exp_val = np.exp(exp_term)

            p_x = (1 / denominator) * exp_val
            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]
