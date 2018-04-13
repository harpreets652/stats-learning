import data_util as du
import matplotlib.pyplot as plt
import numpy as np
import classifier.committee_classifier as adaboost


def build_committee():
    x, y = du.generate_data(11, circle_radius=0.35)

    lines = du.generate_lines(50)

    classifier = adaboost.CommitteeClassifier(x, y, lines, 25)

    predicted_y = []
    for i in x:
        predicted_class = classifier.classify(i)
        predicted_y.append(predicted_class)

    # visualize the results
    committee = classifier.get_committee()
    committee_classifiers = np.array([member["model"] for member in committee.values()])

    visualize_results(x, y, np.array([]))
    visualize_results(x, y, lines)
    visualize_results(x, predicted_y, committee_classifiers)

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
            plt.plot(g, np.reshape(f_x, (f_x.shape[0])), linestyle='-')

    plt.axis([-.1, 1.1, -.1, 1.1])

    plt.show()

    return


if __name__ == '__main__':
    build_committee()
