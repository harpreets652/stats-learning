import csv
import numpy as np
import itertools
import classifier.lin_reg_classifier as lrc
import matplotlib.pyplot as plt
from collections import defaultdict

training_data_file = "/Users/harpreetsingh/github/stats-learning/subset-selection/resources/data.txt"
output_file_name = "/Users/harpreetsingh/github/stats-learning/subset-selection/results/results"


def load_training_data(file_name):
    training_y = []
    training_x = []
    testing_y = []
    testing_x = []
    with open(file_name, 'r') as inFile:
        csv_reader = csv.DictReader(inFile)
        for line in csv_reader:
            features = [float(line['lcavol']), float(line['lweight']),
                        float(line['age']), float(line['lbph']),
                        float(line['svi']), float(line['lcp']),
                        float(line['gleason']), float(line['pgg45'])]

            if line['train'] == 'T':
                training_y.append(float(line['lpsa']))
                training_x.append(np.array(features))
            else:
                testing_y.append(float(line['lpsa']))
                testing_x.append(np.array(features))

    return np.array(training_y), np.array(training_x), np.array(testing_y), np.array(testing_x)


def run_test_data(lin_reg_classifier, test_set_outputs, test_set_features, feature_subset):
    predicted_outputs = lin_reg_classifier.classify_batch(test_set_features[:, feature_subset])
    return np.sum(np.square(np.subtract(test_set_outputs, predicted_outputs)))


def save_results_to_csv(outputs, file_name):
    with open(file_name + ".csv", 'w+') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(["feature subset", "RSS"])
        writer.writerows(outputs)

    return


def visualize_results(subset_results):
    plotting_data_format = []
    buckets = defaultdict(list)
    for r in subset_results:
        plotting_data_format.append((len(r[0]), r[1]))
        buckets[len(r[0])].append(r[1])

    min_rss = []
    for bucket in buckets.values():
        min_rss.append((min(bucket)))

    plot_data = np.array(plotting_data_format)
    plt.scatter(plot_data[:, 0], plot_data[:, 1], facecolors='none', linewidths=0.5, edgecolors='b', s=10)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 100))
    plt.plot(np.array(min_rss), 'r-', linewidth=0.5)

    plt.title("All Models of Subsets of Features")
    plt.xlabel("Feature Subset Size")
    plt.ylabel("RSS (Training Data)")

    plt.show()
    return


# ==================================================================================================================

training_output, training_features, testing_output, testing_features = load_training_data(training_data_file)

# [list, RSS]
results = []
# k in range 1 to 9
# n = 8
for k in range(0, 9):
    for combo in itertools.combinations(range(8), k):
        classifier = lrc.LinearRegClassifier(training_output, training_features[:, combo], 0.0)

        rss = run_test_data(classifier,
                            training_output,
                            training_features,
                            combo)

        print(k, ": ", combo, ", RSS: ", rss)
        results.append((combo, rss))

visualize_results(results)
save_results_to_csv(results, output_file_name)
