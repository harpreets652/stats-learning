import numpy as np

a = np.array(([2, 2, 2], [4, 4, 4]))
print("shape a: \n", a.shape)
print("a: \n", a)
print("a/2: \n", a / 2)
print("np.divide(a, 2)\n", np.divide(a, 2))

b = np.array(([1, 1, 1]))
print("shape b: \n", b.shape)
print("a - b: \n", a - b)
print("a + b: \n", a + b)
print("np.add(a, b): \n", np.add(a, b))

print("a * 2: \n", a * 2)

a_t = a.T
mean = np.sum(a_t, axis=1) / a.shape[0]
mean_no_t = np.sum(a, axis=0) / a.shape[0]
print("mean(with transpose): \n", mean)
print("mean(without transpose): \n", mean_no_t)

center_data = a - mean_no_t
my_cov = np.dot(center_data.T, center_data) / (a.shape[0] - 1)
print("my cov: \n", my_cov)

# verifying from http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
# testing = np.array(([4.0, 4.2, 3.9, 4.3, 4.1],
#                     [2.0, 2.1, 2.0, 2.1, 2.2],
#                     [0.60, 0.59, 0.58, 0.62, 0.63])).T
#
#
# print("Testing array: \n", testing)
# print("numpy.mean: \n", testing.mean(axis=0))
# print("numpy.covariance: \n", np.cov(testing, rowvar=False))
