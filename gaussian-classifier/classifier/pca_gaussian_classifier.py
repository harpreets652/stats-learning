import numpy as np
import operator

import classifier.abstract_gaussian_classifier


class PcaGaussianClassifier(classifier.abstract_gaussian_classifier.AbstractGaussianClassifier):
    """
    Multi-class classifier that fits a Gaussian distribution to multidimensional data
    with pca dimensionality reduction
    Note: assumes Gaussian distribution
    """

    def __init__(self, reg_param, num_principle_components):
        super().__init__(reg_param)
        self.num_components = num_principle_components
        return

    def add_class(self, label, data_file):
        # load training data
        training_data_mat = super()._load_training_data(data_file)

        # find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = training_data_mat.mean(axis=0)

        # find the covariance; row var=False because each column represents a variable, with rows as observations
        cov_mat = np.cov(training_data_mat, rowvar=False)

        # find the eigen vectors/values
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen = list(zip(eigen_vals, eigen_vecs))
        n_principle_components = sorted(eigen, key=operator.itemgetter(0), reverse=True)[:self.num_components]

        # num_components x 256
        eigen_vec_mat = np.array([vec[1] for vec in n_principle_components])

        # n x 256
        training_data_centered = training_data_mat - mean_vec

        # [n x 256] x [256 x num_components] = [n x num_components]
        projected_data = training_data_centered.dot(eigen_vec_mat.T)

        regularization_term = np.multiply(self._regularization_param, np.eye(self.num_components))
        projected_cov = np.cov(projected_data, rowvar=False) + regularization_term

        self.class_models[label] = {'mean': mean_vec, 'cov': projected_cov, 'eigen': eigen_vec_mat}

        return self

    def classify(self, new_data_point):
        class_probabilities = {}

        for label, model in self.class_models.items():
            centered_point = new_data_point - model['mean']
            projected_point = centered_point.dot(model['eigen'].T)
            p_x = super()._calc_probability(model['cov'], projected_point)
            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]
