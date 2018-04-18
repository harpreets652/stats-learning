import numpy as np
import data_util as du
import classifier.svm_classifier as svm


def main():
    x, y = du.generate_linearly_separated_data(num_samples=200,
                                               num_of_features=2,
                                               class_separability=1.0)

    du.visualize_2d_data(x, y)

    classifier = svm.SVMClassifier(x,
                                   y,
                                   c=1000.0,
                                   kernel="linear",
                                   kernel_config={"gamma": 0.5})

    predicted_y = []
    for i in x:
        predicted_class = classifier.classify(i)
        predicted_y.append(predicted_class)

    du.visualize_2d_data(x, np.array(predicted_y), classifier, True)

    return


if __name__ == '__main__':
    main()
