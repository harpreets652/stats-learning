import numpy as np
import matplotlib.pyplot as plt
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

output_file_directory = "/Users/harpreetsingh/github/stats-learning/gaussian-classifier/results/"


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


def run_test_data(test_data_set, gauss_classifier):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((10, 10))

    counter = 0
    for p in test_data_set:
        predicted_class = gauss_classifier.classify(p[1])
        confusion_matrix[p[0]][predicted_class] += 1
        counter += 1
        # print("count: ", counter)

    return confusion_matrix


def calc_performance_from_confusion(confusion_mat):
    total_test_data = np.sum(confusion_mat)

    # calculate the accuracy and error rate (1 - accuracy)
    accuracy = np.round(np.trace(confusion_mat) / total_test_data, 5)
    error_rate = np.round(1.0 - accuracy, 5)

    # calculate the precision of each class (index of array representing the digit)
    precisions = np.zeros((10, 1))
    for i in range(0, 10):
        precisions[i] = np.round(confusion_mat[i][i] / np.sum(confusion_mat[i]), 5)

    return accuracy, error_rate, precisions


def save_data_to_csv(data, file_name, csv_header, data_format):
    np.savetxt(file_name,
               data,
               delimiter=",",
               header=csv_header,
               fmt=data_format)
    return


def run_default_classifier(test_data_set):
    classifier = gauss.DefaultGaussianClassifier(0.1)

    for label, file in training_data_files.items():
        classifier.add_class(label, file)

    confusion = run_test_data(test_data_set, classifier)
    accuracy, error, precisions = calc_performance_from_confusion(confusion)

    csv_file_name = output_file_directory + "all_features_metrics.csv"
    csv_file_header = "Accuracy,Error"

    for i in range(0, 10):
        csv_file_header += ",Precision of {}".format(i)

    save_data_to_csv(np.insert(precisions, 0, (accuracy, error)),
                     csv_file_name,
                     csv_file_header,
                     '%.5f')

    print("done!! \n", confusion)

    return


def visualize_pca_run(run_results):
    plt.scatter(run_results[:, 0], run_results[:, 2], facecolors='none', linewidths=0.5, edgecolors='b', s=10)
    plt.plot(run_results[:, 0], run_results[:, 2], 'r-', linewidth=0.5)

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))

    plt.title("Classifier Error with Varying Dimensions")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Error Rate")

    plt.show()

    return


def run_pca_classifier(test_data_set):
    results = []
    for num_components in range(5, 155, 5):
        classifier = pca.PcaGaussianClassifier(0.1, num_components)

        for label, file in training_data_files.items():
            classifier.add_class(label, file)

        confusion = run_test_data(test_data_set, classifier)
        accuracy, error, precisions = calc_performance_from_confusion(confusion)
        results.append((num_components, accuracy, error))
        print("num_component completed: ", num_components)

    csv_file_name = output_file_directory + "pca_metrics.csv"
    csv_file_header = "Number of Dimensions,Accuracy,Error"
    save_data_to_csv(results,
                     csv_file_name,
                     csv_file_header,
                     '%.5f')

    visualize_pca_run(np.array(results))

    return


# ==================================================================================================================

test_data = load_test_data("/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/test.txt")

# run_default_classifier(test_data)
run_pca_classifier(test_data)
