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
                                   c=0.1,
                                   max_iter=400,
                                   kernel="polynomial",
                                   kernel_config={"d": 2})

    predicted_y = []
    for i in x:
        predicted_class = classifier.classify(i)
        predicted_y.append(predicted_class)

    du.visualize_2d_data(x, np.array(predicted_y), classifier, True)

    return


if __name__ == '__main__':
    main()
