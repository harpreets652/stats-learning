import numpy as np
import classifier.gaussian_classifier as gauss

class_one = 1
class_one_training = "/Users/harpreetsingh/github/stats-learning/k-nearest-neighbors/resources/group_3/train1.txt"
class_two = 9
class_two_training = "/Users/harpreetsingh/github/stats-learning/k-nearest-neighbors/resources/group_3/train9.txt"

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


# ==================================================================================================================

# test_data = load_test_data("/Users/harpreetsingh/github/stats-learning/gaussian-classifier/resources/test.txt")

classifier = gauss.GaussianClassifier(0, 0.001)

for label, file in training_data_files.items():
    classifier.add_class(label, file)

print("done!!")
