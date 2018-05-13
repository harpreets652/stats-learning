import keras


def load_model(model_file_path):
    base_model = keras.models.load_model(model_file_path)
    model = keras.models.Model(inputs=base_model.inputs, outputs=base_model.get_layer('fc_1').output)

    return model


def get_deep_features(model, image):
    cv_image = image.astype('float32')
    cv_image /= 255.0

    return model.predict(cv_image)
