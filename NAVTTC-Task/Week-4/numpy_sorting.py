import numpy as np

numbers = np.array([3, 1, 5, 2, 4])
matrix = np.array([[3, 2, 1], [6, 5, 4]])

print("Sorted one-dimensional array:", np.sort(numbers))
print("Each row sorted independently:\n", np.sort(matrix, axis=1))
print("Each column sorted independently:\n", np.sort(matrix, axis=0))
