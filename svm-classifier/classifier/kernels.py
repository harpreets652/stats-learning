import numpy as np

"""
For reference, check out 
http://scikit-learn.org/stable/modules/metrics.html#pairwise-metrics-affinities-and-kernels
"""


def polynomial(x_1, x_2, config):
    d = config["d"]
    return (np.dot(x_1.T, x_2) + 1) ** d


def linear(x_1, x_2, config):
    return np.dot(x_1.T, x_2)


def rbf(x_1, x_2, config):
    gamma = config["gamma"]
    euclidean_distance = np.linalg.norm(x_1 - x_2) ** 2
    exp_term = -gamma * euclidean_distance
    return np.exp(exp_term)
