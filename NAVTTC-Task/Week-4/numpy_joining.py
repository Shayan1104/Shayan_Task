import numpy as np

first = np.array([[1, 2], [3, 4]])
second = np.array([[5, 6], [7, 8]])

print("Concatenated by rows:\n", np.concatenate((first, second), axis=0))
print("Concatenated by columns:\n", np.concatenate((first, second), axis=1))
print("Stacked into a new dimension:\n", np.stack((first, second), axis=0))
