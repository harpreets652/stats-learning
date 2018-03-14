from random import shuffle

import numpy as np
import operator

training_data_files = {0: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train0.txt",
                       1: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train1.txt",
                       2: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train2.txt",
                       3: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train3.txt",
                       4: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train4.txt",
                       5: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train5.txt",
                       6: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train6.txt",
                       7: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train7.txt",
                       8: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train8.txt",
                       9: "/Users/harpreetsingh/github/stats-learning/neural-network/data/train9.txt"}

test_data_file = "/Users/harpreetsingh/github/stats-learning/neural-network/data/test.txt"


def load_class_training_data(file_name):
    features = []

    with open(file_name, 'r') as file:
        for row in file:
            data_string = row.strip().split(',')
            data = []
            for i in range(len(data_string)):
                data.append(float(data_string[i]))

            features.append(data)

    return np.array(features)


def run_pca(x_mat, num_principle_components):
    # find the mean; axis 0 means mean of each column which represents observations of the feature
    mean_vec = x_mat.mean(axis=0)

    # find the covariance; row var=False because each column represents a variable, with rows as observations
    cov_mat = np.cov(x_mat, rowvar=False)

    # find the eigen vectors/values
    eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
    eigen = list(zip(eigen_vals, eigen_vecs))
    n_principle_components = sorted(eigen, key=operator.itemgetter(0), reverse=True)[:num_principle_components]

    # num_components x 256
    eigen_vec_mat = np.array([vec[1] for vec in n_principle_components])

    # n x 256
    training_data_centered = x_mat - mean_vec

    # [n x 256] x [256 x num_components] = [n x num_components]
    projected_data = training_data_centered.dot(eigen_vec_mat.T)

    return projected_data, mean_vec, eigen_vec_mat


def get_digits_training_data(num_dimensions=16):
    data_with_labels = []
    pca_transform = {}
    for label, file in training_data_files.items():
        class_data = load_class_training_data(file)

        if num_dimensions < class_data.shape[1]:
            reduced_dim, mean, eigen = run_pca(class_data, num_dimensions)
            pca_transform[label] = {"mean": mean, "eigen": eigen}
            class_data = reduced_dim.real

        for i in range(class_data.shape[0]):
            data_with_labels.append((label, class_data[i]))

    shuffle(data_with_labels)

    labels = []
    train_data = []
    for d in data_with_labels:
        labels.append(d[0])
        train_data.append(d[1])

    return np.array(train_data), np.array(labels), pca_transform


def get_test_data(pca_transform=None):
    test_data_buffer = []
    with open(test_data_file, 'r') as file:
        for row in file:
            data_string = row.strip().split()
            data = []
            class_label = -1
            for i in range(len(data_string)):
                if i == 0:
                    class_label = int(data_string[i])
                    continue
                data.append(float(data_string[i]))

            data_np = np.array(data)
            if pca_transform and pca_transform[class_label]:
                centered_data = data_np - pca_transform[class_label]["mean"]
                projected_point = centered_data.dot(pca_transform[class_label]["eigen"].T)
                data_np = projected_point.real

            test_data_buffer.append((class_label, data_np))

    return test_data_buffer
