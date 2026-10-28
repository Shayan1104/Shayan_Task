import numpy as np

numbers = np.array([1, 2, 3])
print("Iterating over a one-dimensional array:")
for number in numbers:
    print(number)

matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Iterating over rows:")
for row in matrix:
    print(row)

print("Iterating over every element:")
for number in np.nditer(matrix):
    print(number)
