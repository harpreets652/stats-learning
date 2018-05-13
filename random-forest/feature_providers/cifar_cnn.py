import numpy as np
import keras


def load_model(model_file_path):
    """
    Load Keras model from file

    :param model_file_path: file path to .h5 file
    :return: model with output of first fully connected network
    """
    if not model_file_path:
        raise RuntimeError("model file path empty")

    base_model = keras.models.load_model(model_file_path)
    model = keras.models.Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc_1').output)

    return model


def get_deep_features_set(model, images):
    """
    Get deep features for set of images

    :param model: Keras model
    :param images: ndarray of images
    :return: 2d array of features
    """
    if not model or not images.size:
        raise RuntimeError("model and/or images not set")

    features_list = []
    for i in range(images.shape[0]):
        img_features = get_deep_features(model, images[i])
        features_list.append(img_features)

    return np.vstack(features_list)


def get_deep_features(model, image):
    """
    get deep features from model for single image

    :param model: Keras model
    :param image: ndarray image
    :return: 2D array of features
    """
    cv_image = image.astype('float32')
    cv_image /= 255.0

    return model.predict(cv_image)
