import numpy as np


# Binary KNN Classifier
class KnnClassifier:

    def __init__(self, pos_class_data, neg_class_data):

        # [ [label, [data]] ]
        self.train_data = []

        self.pos_class = pos_class_data[0]
        self.neg_class = neg_class_data[0]

        self.__load_training_data(self.pos_class, pos_class_data[1])
        self.__load_training_data(self.neg_class, neg_class_data[1])

        return

    def __load_training_data(self, class_label, file_name):
        with open(file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                self.train_data.append((class_label, np.array(data)))

        return

    def classify(self, k, new_data_point):
        k_nearest = self.__find_k_nearest(k, new_data_point)

        prediction = 0.0
        for p in k_nearest:
            prediction += 1 if p == self.pos_class else 0

        prediction /= k

        return self.pos_class if prediction > 0.5 else self.neg_class

    def __find_k_nearest(self, k, new_data_point):
        distances = []
        for p in self.train_data:
            dist = self.__get_distance(p[1], new_data_point)
            distances.append((p[0], dist))

        sorted_points = sorted(distances, key=lambda pt: pt[1])

        k_nearest = []
        for i in range(k):
            k_nearest.append(sorted_points[i][0])

        return k_nearest

    @staticmethod
    def __get_distance(data_point, input_point):
        return np.sqrt(np.sum(np.square(data_point - input_point)))
