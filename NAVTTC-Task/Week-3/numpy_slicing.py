import numpy as np

numbers = np.array([1, 2, 3, 4, 5, 6, 7])

print("Elements from index 1 to 4:", numbers[1:5])
print("First four elements:", numbers[:4])
print("Elements from index 4:", numbers[4:])
print("Every second element:", numbers[::2])
print("Last three elements:", numbers[-3:])

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("First two rows, first two columns:\n", matrix[:2, :2])
