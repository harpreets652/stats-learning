import numpy as np
import matplotlib.pyplot as plt
import data_util as du


def generate_data():
    x, y = du.generate_data(11)

    color = {1: 'r', -1: 'b'}

    for i in range(x.shape[0]):
        plt.scatter(x[i][0], x[i][1], color=color[y[i]], s=10)

    plt.show()

    return


def generate_rand_lines():
    num_of_lines = 50

    lines = []
    for i in range(num_of_lines):
        x1 = np.round(np.random.random_sample(2), 2)
        x2 = np.round(np.random.random_sample(2), 2)
        coefficients = np.polyfit(x1, x2, 1)

        lines.append((coefficients[1], coefficients[0]))

    lines = np.array(lines)

    x = np.linspace(0, 1, 5)
    for line in lines:
        y = line[0] * x[:, np.newaxis] + line[1]

        plt.plot(x, np.reshape(y, (y.shape[0])), linestyle='-')

    plt.axis([0, 1, 0, 1])
    plt.xticks(x)
    plt.yticks(x)
    plt.show()

    return


def generate_all_lines():
    size = 3

    # line equation: y=mx+b, m = slope, b=y-intercept
    m_intermediate = np.concatenate((np.linspace(-1, 0, size), np.linspace(0, 1, size)), 0)
    m_dup = []
    for i in m_intermediate:
        for j in m_intermediate:
            if j == 0:
                m_dup.append(0)
            else:
                m_dup.append(i / j)

    m = np.unique(m_dup)
    b = np.unique(np.concatenate((np.linspace(-2, -1, size),
                                  np.linspace(-1, 0, size),
                                  np.linspace(0, 1, size),
                                  np.linspace(1, 2, size)), 0))

    # list of (slope, y-intercept)
    lines = []
    for i in m:
        for j in b:
            lines.append((i, j))

    lines = np.array(lines)

    x = np.linspace(0, 1, size)
    for line in lines:
        y = line[0] * x[:, np.newaxis] + line[1]

        plt.plot(x, np.reshape(y, (y.shape[0])), linestyle='-')

    plt.axis([0, 1, 0, 1])
    plt.xticks(x)
    plt.yticks(x)
    plt.show()

    return


def other_stuff():
    x = np.array(([7, 2, 3],
                  [4, 5, 6]))

    g = np.insert(x, 0, 1, axis=1)
    print("g: \n", g)

    print("sum: \n", np.sum(x, axis=0))

    return


if __name__ == '__main__':
    other_stuff()
