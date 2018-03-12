import numpy as np
import operator


def load_test_data(file_name):
    test_data_buffer = []
    with open(file_name, 'r') as file:
        for row in file:
            data_string = row.strip().split()
            data = []
            class_label = -1
            for i in range(len(data_string)):
                if i == 0:
                    class_label = int(data_string[i])
                    continue
                data.append(float(data_string[i]))

            test_data_buffer.append((class_label, np.array(data)))

    return test_data_buffer


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

    return projected_data
