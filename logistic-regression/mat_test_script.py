import numpy as np

x = np.array(([1, 8, 8, 8, 8],
              [1, 4, 4, 4, 4],
              [1, 12, 12, 12, 12]))


g = np.array(([2], [3], [4]))

print("np.multiply(g, x): \n", np.multiply(g, x))
print("g * x: \n", g * x)

print("x + 1: \n", x + 1)
print("sum of columns: \n", np.sum(x, axis=0))

y = np.array([1, 1, 1])

p = np.array([0.75, 0.75, 0.75])

t = np.array([0, 1, 0])
print ("sigmoid: ", np.exp(t) / (1 + np.exp(t)))

prod = np.dot((y - p), x)

print("result: \n", prod)


