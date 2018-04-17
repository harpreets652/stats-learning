import data_util as du
import matplotlib.pyplot as plt
import numpy as np
import classifier.committee_classifier as adaboost


def build_committee():
    # for 25, use .4, 11, use .35
    x, y = du.generate_data(100, circle_radius=0.42)

    size_of_committee = 1500
    # lines = du.generate_square()
    lines = du.generate_lines(size_of_committee)

    classifier_dict = {}
    for i in range(lines.shape[0]):
        classifier_dict[i] = {"model": lines[i], "sign": 1}

    classifier = adaboost.CommitteeClassifier(x, y, classifier_dict, size_of_committee)

    predicted_y = []
    confusion = np.zeros((2, 2))
    prediction_mapping = {-1: 0, 1: 1}
    for k in range(x.shape[0]):
        predicted_class = classifier.classify(x[k])
        predicted_y.append(predicted_class)
        confusion[prediction_mapping[y[k]]][prediction_mapping[predicted_class]] += 1

    # visualize the results
    committee = classifier.get_committee()
    committee_classifiers = np.array([member["classifier"]["model"] for member in committee.values()])

    # visualize_results(x, y, np.array([]))
    visualize_results(x, predicted_y, np.array([]))
    visualize_results(x, y, np.array([]))

    print("committee: \n", committee)
    print("size of data: ", x.shape[0])

    total = np.sum(confusion)
    sum_class = np.sum(confusion, axis=1)
    correct = np.trace(confusion)
    print(f"total accuracy: {correct/total}")
    print(f"negative accuracy: {confusion[0][0]/sum_class[0]}")
    print(f"positive accuracy: {confusion[1][1]/sum_class[1]}")
    print(f"Confusion Matrix: \n {confusion}")

    return


def visualize_results(x, y, lines):
    colors = {1: 'r', -1: 'b'}
    markers = {1: '+', -1: '*'}

    for i in range(x.shape[0]):
        plt.scatter(x[i][0], x[i][1], color=colors[y[i]], marker=markers[y[i]], s=35)

    if lines.size:
        g = np.linspace(0, 1, 3)
        for line in lines:
            f_x = line[1] * g[:, np.newaxis] + line[0]
            plt.plot(g, np.reshape(f_x, (f_x.shape[0])), linestyle='-', alpha=0.4)

    plt.axis([-.1, 1.1, -.1, 1.1])

    plt.show()

    return


if __name__ == '__main__':
    build_committee()
