import csv
import numpy as np
import itertools
import classifier.lin_reg_classifier as lrc

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


# ==================================================================================================================

training_output, training_features, testing_output, testing_features = load_training_data(training_data_file)

# [list, RSS]
results = []
# k in range 1 to 9
# n = 8
for k in range(1, 9):
    for combo in itertools.combinations(range(8), k):
        classifier = lrc.LinearRegClassifier(training_output, training_features[:, combo], 0.0)
        rss = run_test_data(classifier, testing_output, testing_features, combo)
        print(k, ": ", combo, ", RSS: ", rss)
        results.append((combo, rss))

save_results_to_csv(results, output_file_name)
