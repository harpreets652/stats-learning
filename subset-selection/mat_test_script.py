import numpy as np

# a = np.array(([1, 1, 1], [1, 1, 1]))
#
# b = np.array([2, 2, 2])
#
# c = np.array(([1, 5, 2], [8, 3, 2]))
#
# print("a: \n", a)
# print("b: \n", b)
# # print(np.concatenate([a, b])) # won't work since a and b aren't same shape
# print("add another row of 3 at the 0th index: \n", np.insert(a, 0, 3, axis=0))
# print("add another column of 3 at the 0th index: \n", np.insert(a, 0, 3, axis=1))
# print("0.005 * a: \n", 0.005 * a)
#
# print("np.multiply(0.001, a): \n", np.multiply(0.001, a))
# print("a transpose: \n", a.T)
# print("np.matmult(a.t, b): \n", np.matmul(b, a.T))
# print("a + c: \n", a + c)
# print("np.add(a, c): \n", np.add(a, c))
#
# print("np.append(a, b): \n", np.append(a, b))

x = np.array(([1, 1], [2, 2]))
b = np.array(([0.5, 0.5]))
print("x: \n", x)
print("b: \n", b)
print("x * b\n", np.matmul(x, b))
print("x.T * b\n", np.matmul(x.T, b))
