import numpy as np
import classifier.default_gaussian_classifier as gauss
import classifier.pca_gaussian_classifier as pca

training_data_files = {0: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train0.txt",
                       1: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train1.txt",
                       2: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train2.txt",
                       3: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train3.txt",
                       4: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train4.txt",
                       5: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train5.txt",
                       6: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train6.txt",
                       7: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train7.txt",
                       8: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train8.txt",
                       9: "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/train9.txt"}

output_file_prefix = "/Users/harpreetsingh/github/stats-learning/linear-regression/results/1_vs_9"


def run_test_data(test_data_set, gauss_classifier):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((10, 10))

    for p in test_data_set:
        predicted_class = gauss_classifier.classify(p[1])
        confusion_matrix[p[0]][predicted_class] += 1
        print("p: ", p[0])

    return confusion_matrix


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


# todo: for this save csv: class,accuracy,missclassification error
def run_default_classifier(test_data_set):
    classifier = gauss.DefaultGaussianClassifier(0.1)

    for label, file in training_data_files.items():
        classifier.add_class(label, file)

    confusion = run_test_data(test_data_set, classifier)

    print("done!! \n", confusion)

    return


# todo: for this, graph num dimensions versus error (of the entire testing set, combining all class totals)
def run_pca_classifier(test_data_set):
    classifier = pca.PcaGaussianClassifier(0.1, 25)

    for label, file in training_data_files.items():
        classifier.add_class(label, file)

    confusion = run_test_data(test_data_set, classifier)

    print("done!! \n", confusion)

    return


# ==================================================================================================================

test_data = load_test_data("/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/test.txt")

# run_default_classifier(test_data)
run_pca_classifier(test_data)
