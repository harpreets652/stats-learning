import numpy as np

w = np.array(([1, 2, 3],
              [4, 5, 6]))

w_c = np.array(([1, 4],
                [2, 5],
                [3, 6]))
print("w \n", w)
print("w.T \n", w.T)

x = np.array(([0.5, 0.5, 0.5]))
print("x \n", x)

b = np.array(([10, 20]))
print("b \n", b)

print("xw.T + b", x.dot(w.T) + b)
print("xw_c + b", x.dot(w_c) + b)
