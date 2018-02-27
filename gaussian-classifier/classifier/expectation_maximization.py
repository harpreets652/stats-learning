import numpy as np
import random
import operator


def execute_expectation_maximization(x, reg_param, num_gaussian, num_iterations):
    regularization_term = np.multiply(reg_param, np.eye(x.shape[1]))
    sample_cov_mat = np.cov(x, rowvar=False) + regularization_term

    # mean = random sample point, covariance = sample covariance
    gaussian_models = []
    for i in range(num_gaussian):
        rand_mean = random.randint(0, x.shape[0])
        gaussian_models.append({"mean": x[rand_mean], "cov": sample_cov_mat, "gamma": 0.5})

    # x_responsibilities = [[Xi], [Gaussian_model_index]]
    x_responsibilities = []
    for i in range(x.shape[0]):
        buffer = [x[i], -1]
        x_responsibilities.append(buffer)

    # main loop
    for i in range(num_iterations):
        print("iteration: ", i)
        # expectation
        _expectation(gaussian_models, x_responsibilities)

        # maximization
        _maximization(gaussian_models, x_responsibilities, regularization_term)

    return gaussian_models


def _expectation(gaussian_dict, data):
    """
    Assign a Gaussian to each data point
    :param gaussian_dict: dictionary of Gaussian models
    :param data: input data
    :return: modifies data directly
    """
    for i in range(len(data)):
        x_i = data[i][0]

        mahalanobis_distances = {}
        for j in range(len(gaussian_dict)):
            centered_point = x_i - gaussian_dict[j]["mean"]
            mahalanobis_distance = centered_point.dot(np.linalg.inv(gaussian_dict[j]["cov"])).dot(centered_point)
            mahalanobis_distances[j] = mahalanobis_distance

        data[i][1] = min(mahalanobis_distances.items(), key=operator.itemgetter(1))[0]

    return


def _maximization(gaussian_dict, data, regularization_term):
    size_of_data = len(data)
    for i in range(len(gaussian_dict)):
        x_cluster = np.array([t[0] for t in data if t[1] == i])

        gaussian_dict[i]["mean"] = x_cluster.mean(axis=0)
        gaussian_dict[i]["cov"] = np.cov(x_cluster, rowvar=False) + regularization_term
        gaussian_dict[i]["gamma"] = x_cluster.shape[0] / size_of_data

    return
