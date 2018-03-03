import numpy as np
import classifier.perceptron_classifier as pc

training_data = "/Users/harpreetsingh/github/stats-learning/simple-perceptron/resources/data.txt"


def load_training_data(file_name):
    labels = []
    features = []
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

            labels.append(class_label)
            features.append(np.array(data))

    return np.array(labels), np.array(features)


def main():
    x, y = load_training_data(training_data)

    classifier = pc.PerceptronClassifier(y, x)

    return


if __name__ == '__main__':
    main()
