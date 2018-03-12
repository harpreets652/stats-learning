import numpy as np

w = np.array(([1, 2, 3], [4, 5, 6]))
print("w \n", w)
print("w.T \n", w.T)

x = np.array(([0.5, 0.5, 0.5]))
print("x \n", x)

b = np.array(([10], [10]))
print("b \n", b)


print(x.dot(w.T) + b.T)
