import numpy as np
import matplotlib.pyplot as plt
import data_util as du


def generate_data():
    x, y = du.generate_data(25)

    color = ['r', 'b']

    for i in range(x.shape[0]):
        plt.scatter(x[i][0], x[i][1], color=color[y[i]], s=10)

    plt.show()

    return


def generate_lines():
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


if __name__ == '__main__':
    generate_lines()
