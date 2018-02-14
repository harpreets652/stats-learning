import numpy as np
import operator

import classifier.abstract_gaussian_classifier


class DefaultGaussianClassifier(classifier.abstract_gaussian_classifier.AbstractGaussianClassifier):
    """
    Multi-class classifier that fits a Gaussian distribution to multidimensional data
    Note: assumes Gaussian distribution
    """

    def add_class(self, label, data_file):
        # load training data
        training_data_mat = super()._load_training_data(data_file)

        # find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = training_data_mat.mean(axis=0)

        # find the covariance; row var=False because each column represents a variable, with rows as observations
        regularization_term = np.multiply(self.regularization_param, np.eye(training_data_mat.shape[1]))
        cov_mat = np.cov(training_data_mat, rowvar=False) + regularization_term

        # add to a dictionary
        self.class_models[label] = {'mean': mean_vec, 'cov': cov_mat}

        return self

    def classify(self, new_data_point):
        if not self.class_models:
            raise RuntimeError("no class models found")

        class_probabilities = {}

        for label, model in self.class_models.items():
            p_x = super()._calc_probability(model['mean'], model['cov'], new_data_point - model['mean'])
            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]
