import numpy as np


class LinearRegClassifier:
    """
    Linear regression classifier using the normal equation: inverse(X'X + lambda*I) * X'y = beta
    """

    def __init__(self, pos_class_data, neg_class_data, regularization_param):
        self.pos_class = pos_class_data[0]
        self.neg_class = neg_class_data[0]

        # load data X
        label_pos, features_pos = self.__load_training_data(1, pos_class_data[1])
        label_neg, features_neg = self.__load_training_data(-1, neg_class_data[1])

        y = np.concatenate([label_pos, label_neg])
        x = np.concatenate([features_pos, features_neg])

        # add additional column of ones to X at index 0
        x = np.insert(x, 0, 1, axis=1)

        # initialize beta
        self.beta = np.zeros((x.shape[1], 1))

        # normal equation
        # [n+1 x n] * [n x n+1] = [n+1 x n+1]
        xTx = np.matmul(x.T, x)

        # regularization = [n+1 x n+1]
        regularization_term = np.multiply(regularization_param, np.eye(x.shape[1]))

        # inverse([n+1 x n+1] + [n+1 x n+1]) = [n+1 x n+1]
        first_term = np.linalg.inv(np.add(xTx, regularization_term))

        # [n+1 x n] * [n x 1] = [n+1 x 1]
        xTy = np.matmul(x.T, y)

        # [n+1 x n+1] * [n+1 x 1] = [n+1 x 1]
        self.beta = np.matmul(first_term, xTy)

        return

    @staticmethod
    def __load_training_data(class_label, file_name):
        label = []
        features = []

        with open(file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                label.append(class_label)
                features.append(data)

        return np.array(label), np.array(features)

    def classify(self, new_data_point):
        x = np.insert(new_data_point, 0, 1, axis=0)
        y_hat = np.matmul(x.T, self.beta)

        return self.pos_class if y_hat > 0.5 else self.neg_class
