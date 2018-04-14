import numpy as np
import operator
import math


class CommitteeClassifier(object):
    """
    AdaBoost classifier

    http://www.inf.fu-berlin.de/inst/ag-ki/rojas_home/documents/tutorials/adaboost4.pdf
    """

    def __init__(self, x_train, y_label, classifiers_pool, committee_size=5):
        """
        Constructs a committee of classifiers from the pool

        :param x_train: 2D np array (x1, x2)
        :param y_label: corresponding labels, {-1, 1}
        :param classifiers_pool: list of classifiers, in this case, 2D lines (coefficient_1, bias)
        :param committee_size: size of the committee to construct
        """
        assert len(classifiers_pool) >= committee_size

        s_mat = CommitteeClassifier._get_scouting_matrix(classifiers_pool, x_train, y_label)
        self._committee = {}
        x_weights = np.ones((x_train.shape[0], 1))

        for i in range(committee_size):
            # compute We for each classifier and select classifier with smallest value
            w = np.sum(x_weights)
            w_e = np.dot(x_weights.T, s_mat).T

            w_e_dict = {}
            for k in range(w_e.shape[0]):
                if k not in self._committee.keys():
                    w_e_dict[k] = w_e[k]

            best_classifier_idx = min(w_e_dict.items(), key=operator.itemgetter(1))[0]

            e_m = w_e[best_classifier_idx] / w
            alpha = 0.5 * math.log((1 - e_m) / e_m)

            self._committee[best_classifier_idx] = {"model": classifiers_pool[best_classifier_idx], "alpha": alpha}

            for j in range(s_mat.shape[0]):
                if s_mat[j][best_classifier_idx]:
                    x_weights[j] = x_weights[j] * math.sqrt((1 - e_m) / e_m)
                else:
                    x_weights[j] = x_weights[j] * math.sqrt(e_m / (1 - e_m))

        return

    @staticmethod
    def _get_scouting_matrix(classifier_pool, x_train, y_label):
        """
        return an array of size n x m where n = number of training examples and m = number of classifiers
        The resulting array is in descending order of success

        :param classifier_pool: list of classifiers
        :param x_train: training data
        :param y_label: labels for each training example
        :return: ndarray
        """
        s_mat = []
        x = np.insert(x_train, 0, 1, axis=1)

        for classifier in classifier_pool:
            product = np.dot(x[:, :2], classifier[:, np.newaxis])
            hit_miss = []
            for i in range(x.shape[0]):
                predicted_classification = 1 if x[i][2] > product[i] else -1

                # a correct classification results in 0 (no cost incurred) and 1 otherwise
                score = 0 if predicted_classification == y_label[i] else 1
                hit_miss.append(score)

            s_mat.append(np.array(hit_miss))

        return np.array(s_mat).T

    def get_committee(self):
        return self._committee

    def classify(self, new_point):
        """
        predicts classification of new point

        :param new_point: 2D array (x1, x2)
        :return: +1 or -1
        """

        committee_sum = 0
        x_extended = np.insert(new_point, 0, 1, axis=0)
        for key, info in self._committee.items():
            model = info["model"]
            alpha = info["alpha"]

            product = np.dot(x_extended[:2], model)
            predicted_classification = 1 if x_extended[2] > product else -1
            committee_sum += alpha * predicted_classification

        return 1 if committee_sum >= 0 else -1
