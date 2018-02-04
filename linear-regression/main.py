import numpy as np
import classifier.lin_reg_classifier as lrc

class_one = 0
class_one_training = "/Users/harpreetsingh/github/stats-learning/k-nearest-neighbors/resources/group_1/train0.txt"
class_two = 3
class_two_training = "/Users/harpreetsingh/github/stats-learning/k-nearest-neighbors/resources/group_1/train3.txt"
output_file_prefix = "/Users/harpreetsingh/github/stats-learning/linear-regression/results/0_vs_3"


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


def run_test_data(lin_reg_classifier, test_data_set, class_a, class_b):
    prediction_metrics = {'True_A': 0, 'True_B': 0, 'False_B': 0, 'False_A': 0}

    for p in test_data_set:
        if p[0] in (class_a, class_b):
            predicted_class = lin_reg_classifier.classify(p[1])
            if predicted_class == p[0]:
                if p[0] == class_a:
                    prediction_metrics['True_A'] += 1
                else:
                    prediction_metrics['True_B'] += 1
            else:
                if p[0] == class_a:
                    prediction_metrics['False_B'] += 1
                else:
                    prediction_metrics['False_A'] += 1

    return prediction_metrics


def calculate_performance(class_a, class_b, reg_param, test_data_results):
    total_test_data = sum(test_data_results.values())

    # accuracy
    accuracy = (test_data_results['True_A'] + test_data_results['True_B']) / total_test_data

    # misclassification
    misclassification = (test_data_results['False_A'] + test_data_results['False_B']) / total_test_data

    # precision
    total_a_predictions = test_data_results['True_A'] + test_data_results['False_A']
    precision_a = test_data_results['True_A'] / total_a_predictions

    total_b_predictions = test_data_results['True_B'] + test_data_results['False_B']
    precision_b = test_data_results['True_B'] / total_b_predictions

    # recall
    actual_a = test_data_results['True_A'] + test_data_results['False_B']
    recall_a = test_data_results['True_A'] / actual_a

    actual_b = test_data_results['True_B'] + test_data_results['False_A']
    recall_b = test_data_results['True_B'] / actual_b

    # false positive rates: when it's actually B, how often did it predict A(false_pos_A) and vice virsa
    # false_positive_a = test_data_results['False_A'] / actual_b
    # false_positive_b = test_data_results['False_B'] / actual_a

    print("")
    print(
        "Lambda,Accuracy,Misclassification,Precision of {},Precision of {},Recall of {},Recall of {}"
            .format(class_a, class_b, class_a, class_b))
    print("{},{},{},{},{},{},{}".format(reg_param, round(accuracy, 5), round(misclassification, 5),
                                        round(precision_a, 5), round(precision_b, 5), round(recall_a, 5),
                                        round(recall_b, 5)))

    print("")
    print("Number of {},Number of {},lambda,True A,True B,False B,False A".format(class_a, class_b))
    print("{},{},{},{},{},{},{}".format(actual_a,
                                        actual_b,
                                        reg_param,
                                        test_data_results['True_A'],
                                        test_data_results['True_B'],
                                        test_data_results['False_B'],
                                        test_data_results['False_A']))

    metrics_array = np.array([reg_param, round(accuracy, 5),
                              round(misclassification, 5),
                              round(precision_a, 5), round(precision_b, 5),
                              round(recall_a, 5), round(recall_b, 5)])

    confusion_mat = np.array([actual_a, actual_b, reg_param,
                              test_data_results['True_A'], test_data_results['True_B'],
                              test_data_results['False_B'], test_data_results['False_A']])

    return metrics_array, confusion_mat


def save_metrics_to_csv(performance_metrics, confusion_mats, file_name, class_a, class_b):
    performance_metrics_header = "Lambda,Accuracy,Misclassification,Precision of {},Precision of {},Recall of {},Recall of {}" \
        .format(class_a, class_b, class_a, class_b)

    confusion_mat_header = "Number of {},Number of {},lambda,True A,True B,False B,False A".format(class_a, class_b)

    np.savetxt(file_name + "_metrics.csv",
               performance_metrics,
               delimiter=",",
               header=performance_metrics_header,
               fmt=['%.1f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f', '%.5f'])

    np.savetxt(file_name + "_confusion.csv",
               confusion_mats,
               delimiter=",",
               header=confusion_mat_header,
               fmt=['%i', '%i', '%.1f', '%i', '%i', '%i', '%i'])

    return


# ==================================================================================================================

test_data = load_test_data("/Users/harpreetsingh/github/stats-learning/k-nearest-neighbors/resources/test_set.txt")

performance_metrics = []
confusion_matrices = []
for regularization_param in np.arange(0.0, 10.5, 0.5):
    classifier = lrc.LinearRegClassifier([class_one, class_one_training],
                                         [class_two, class_two_training],
                                         regularization_param)

    test_results = run_test_data(classifier, test_data, class_one, class_two)
    metrics, confusion = calculate_performance(class_one, class_two, regularization_param, test_results)
    performance_metrics.append(metrics)
    confusion_matrices.append(confusion)

save_metrics_to_csv(performance_metrics, confusion_matrices, output_file_prefix, class_one, class_two)
