import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import classifier.expectation_maximization as em


# test data generation code from: http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
np.random.seed(0)

n_samples = 500

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=2)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# plt.scatter(X[:, 0], X[:, 1], color='b', s=10)

# plt.show()

gaussian_models = em.execute_expectation_maximization(X, 0.1, 2, 500)

print("gaussian models: \n", gaussian_models)

for i in range(len(gaussian_models)):
    model = gaussian_models[i]
    plt.scatter(model["mean"][0], model["mean"][1], color='r', s=50, marker='*')
plt.scatter(X[:, 0], X[:, 1], linewidths=0.5, edgecolors='b', s=10, marker='.')
plt.title("EM Results with Sample 2D Data")
plt.figtext(0.1, 0.009, "The points represent test data and the start the mean of the cluster.")

plt.show()
