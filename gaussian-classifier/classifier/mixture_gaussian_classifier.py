import classifier.abstract_gaussian_classifier
import classifier.expectation_maximization as em
import operator


class MixtureOfGaussianClassifier(classifier.abstract_gaussian_classifier.AbstractGaussianClassifier):
    """
    Multi-class Mixture of Gaussians classifier
    """

    def __init__(self, reg_param=0.1, num_of_gaussian=2, em_algorithm_iterations=500):
        super().__init__(reg_param)
        self._num_gaussian = num_of_gaussian
        self._em_num_iterations = em_algorithm_iterations
        return

    def add_class(self, label, data_file):
        # load training data
        training_data_mat = super()._load_training_data(data_file)

        # call em algorithm and store the mean, covariance, and gamma
        gaussian_models = em.execute_expectation_maximization(training_data_mat,
                                                              self._regularization_param,
                                                              self._num_gaussian,
                                                              self._em_num_iterations)

        self._class_models[label] = gaussian_models

        return self

    def classify(self, new_data_point):
        if not self._class_models:
            raise RuntimeError("no class models found")

        class_probabilities = {}
        for label, class_gaussian_models in self._class_models.items():
            p_x = 0
            for model in class_gaussian_models:
                p_x += super()._calc_probability(model['cov'], new_data_point - model['mean']) * model["gamma"]

            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]
