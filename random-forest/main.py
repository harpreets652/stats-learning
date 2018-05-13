import numpy as np
from keras import datasets
from sklearn.ensemble import RandomForestClassifier

from feature_providers import bag_of_features as bof


def main():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # bag of features
    img_feature_gen = bof.BagOfFeaturesTransform(patch_size=16, num_clusters=10)
    training_x = img_feature_gen.initialize(x_train)

    classifier = RandomForestClassifier(n_estimators=10, criterion="gini", min_samples_split=2, n_jobs=-1)
    classifier.fit(training_x, np.ravel(y_train))
    print("Random Forest built")

    confusion = np.zeros((10, 10))
    for i in range(x_test.shape[0]):
        print(f"Test image {i}")
        x = img_feature_gen.get_image_descriptor(x_test[i])
        predicted = classifier.predict(x)[0]
        confusion[y_test[i][0]][predicted] += 1

    print("confusion: \n", confusion)

    return


if __name__ == '__main__':
    main()
