import numpy as np
import pickle

"""
This util file reads CIFAR data set
"""


def read_batch(batch_file_path):
    """
    Read python data file and expand image

    :param batch_file_path: path to python data file
    :return: list [label, 32x32x3 image array]
    """
    with open(batch_file_path, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')

    batch_data = []
    for i in range(data_dict[b'data'].shape[0]):
        image = data_dict[b'data'][i].reshape(3, 32, 32).transpose([1, 2, 0])
        batch_data.append((data_dict[b'labels'][i], image))

    return batch_data
