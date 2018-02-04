import numpy as np
from random import shuffle


class LinearRegClassifier:
    """
    Linear regression classifier using the normal equation: inverse(X'X + lambda*I) * X'y = beta
    """

    def __init__(self, pos_class_data, neg_class_data, regularization_param):
        self.pos_class = pos_class_data[0]
        self.neg_class = neg_class_data[0]

        # load data X and shuffle
        pos_data = self.__load_training_data_with_label(1, pos_class_data[1])
        neg_data = self.__load_training_data_with_label(-1, neg_class_data[1])
        training_data = pos_data + neg_data
        shuffle(training_data)

        labels = []
        train_data = []
        for d in training_data:
            labels.append(d[0])
            train_data.append(d[1])

        y = np.array(labels)
        x = np.array(train_data)

        # add additional column of ones to X at index 0
        x = np.insert(x, 0, 1, axis=1)

        # initialize beta
        self.beta = np.zeros((x.shape[1], 1))

        # normal equation
        # [n+1 x n] * [n x n+1] = [n+1 x n+1]
        xTx = np.matmul(x.T, x)

        # regularization = [n+1 x n+1]
        regularization_term = np.multiply(regularization_param, np.eye(x.shape[1]))

        # if det(a) = 0, then it does not have an inverse
        # print("determinant of first term: ", np.linalg.det(np.add(xTx, regularization_term)))

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

    @staticmethod
    def __load_training_data_with_label(pos_class_label, pos_class_file_name):
        training_data = []
        with open(pos_class_file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                training_data.append((pos_class_label, np.array(data)))

        return training_data

    def classify(self, new_data_point):
        x = np.insert(new_data_point, 0, 1, axis=0)
        y_hat = np.matmul(x.T, self.beta)

        return self.pos_class if y_hat > 0.0 else self.neg_class
