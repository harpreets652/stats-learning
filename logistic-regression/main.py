import numpy as np
import matplotlib.pyplot as plt
import classifier.logistic_reg_classifier as lrc

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


def run_test_data(test_data_set, p_classifier, class_a, class_b):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((2, 2))
    class_index = {class_a: 0, class_b: 1}

    counter = 0
    for p in test_data_set:
        if p[0] in (class_a, class_b):
            predicted_class = p_classifier.classify(p[1])
            confusion_matrix[class_index[p[0]]][class_index[predicted_class]] += 1
            counter += 1
            # print("count: ", counter)

    return confusion_matrix


def save_data_to_csv(data, file_name, csv_header, data_format):
    np.savetxt(file_name,
               data,
               delimiter=",",
               header=csv_header,
               fmt=data_format)
    return


def visualize_grad_history(gradient_results):
    plt.plot(gradient_results[:, 0], gradient_results[:, 1], 'r-', linewidth=0.5)

    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, 0, 1))

    plt.title("Gradient Ascent Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood Output")

    plt.show()

    return


def visualize_multi_grad_history(gradient_results):
    styles = ['-', '-.', '--', ':']
    colors = ['r', 'b', 'g', 'k']

    counter = 0
    for results in gradient_results:
        plt.plot(results[1][:, 0],
                 results[1][:, 1],
                 color=colors[counter],
                 linestyle=styles[counter],
                 linewidth=1.5,
                 label="Alpha=" + str(results[0])
                 )
        counter += 1

    # x1, x2, y1, y2 = plt.axis()
    # plt.axis((x1, x2, 0, 1))

    plt.legend(loc='best')
    plt.title("Gradient Ascent Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood Output")

    plt.show()

    return


# ==================================================================================================================
test_data = load_test_data("/Users/harpreetsingh/github/stats-learning/logistic-regression/resources/test_set.txt")

error_histories = []
learning_rates = [1.0, 0.5, 0.1, 0.0001]

for i in learning_rates:
    classifier = lrc.LogisticRegClassifier([0, training_data_files[0]],
                                           [3, training_data_files[3]],
                                           i,
                                           200)
    error_histories.append((i, classifier.get_grad_error_history()))

visualize_multi_grad_history(error_histories)

# visualize_grad_history(classifier.get_grad_error_history())

# c_matrix = run_test_data(test_data, classifier, 1, 9)
# print("confusion matrix: \n", c_matrix)
