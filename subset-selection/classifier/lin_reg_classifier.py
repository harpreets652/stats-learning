import numpy as np


class LinearRegClassifier:
    """
    Linear regression model using the normal equation: inverse(X'X + lambda*I) * X'Y = beta
    """

    def __init__(self, training_y, training_x, regularization_param):
        y = training_y
        x = training_x

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

    def classify(self, new_data_point):
        """
        :param new_data_point: nxm matrix of features
        :returns: predicted scalar value
        :rtype: float
        """
        x = np.insert(new_data_point, 0, 1, axis=0)

        return np.matmul(x.T, self.beta)

    def classify_batch(self, data_points):
        """
        :param data_points: nxm matrix of features
        :returns: vector of predicted values
        :rtype: np.array
        """
        x = np.insert(data_points, 0, 1, axis=1)
        return np.matmul(x, self.beta)
