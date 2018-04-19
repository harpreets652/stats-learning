import numpy as np
import data_util as du
import classifier.svm_classifier as svm


def main():
    num_samples = 200
    x, y = du.generate_linearly_separated_data(num_samples=num_samples,
                                               num_of_features=2,
                                               class_separability=0.8)

    du.visualize_2d_data(x, y)

    classifier = svm.SVMClassifier(x,
                                   y,
                                   c=25.0,
                                   kernel="rbf",
                                   kernel_config={"gamma": 0.9})

    predicted_y = []
    confusion = np.zeros((2, 2))
    confusion_mapping = {-1: 0, 1: 1}
    for i in range(x.shape[0]):
        predicted_class = classifier.classify(x[i])

        confusion[confusion_mapping[y[i]]][confusion_mapping[predicted_class]] += 1
        predicted_y.append(predicted_class)

    print("Confusion Matrix: \n", confusion)
    print(f"Number of data points: {num_samples}\nNumber of support vectors: {len(classifier.get_support_vectors())}")
    du.visualize_2d_data(x, np.array(predicted_y), classifier, True)

    return


if __name__ == '__main__':
    main()
