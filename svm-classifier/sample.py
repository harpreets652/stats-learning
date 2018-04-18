import random
import numpy as np


temp_arr = np.arange(0, 9)

print("temp array: ", temp_arr)

result = np.array(np.where(temp_arr % 2 == 0)[0])

print("result: ", result)
print("rolled: ", np.roll(result, random.randint(0, result.shape[0] - 1)))
