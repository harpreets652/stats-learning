import os
from feature_providers import bag_of_features as bof
import data_util as du


def read_training_data(training_data_dir):
    """
    returns list of training images and their corresponding labels

    :param training_data_dir:
    :return: list [label, 3D image array]
    """
    all_data = []
    for image_file in os.listdir(training_data_dir):
        if "data_batch" not in image_file:
            continue

        data = du.read_batch(training_data_dir + "/" + image_file)

        all_data += data

    return all_data


def main():
    data_directory = "/Users/harpreetsingh/Downloads/cifar-10-batches-py"
    training_data = read_training_data(data_directory)

    img_feature_gen = bof.BagOfFeaturesTransform(patch_size=16, num_clusters=3)
    training_x = img_feature_gen.initialize(training_data)



    return


if __name__ == '__main__':
    main()
