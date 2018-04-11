import data_util as du
import matplotlib.pyplot as plt

x, y = du.generate_data(25)

color = ['r', 'b']

for i in range(x.shape[0]):
    plt.scatter(x[i][0], x[i][1], color=color[y[i]], s=10)

plt.show()
