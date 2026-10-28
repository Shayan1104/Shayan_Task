import numpy as np

numbers = np.array([1, 2, 3, 4, 5, 6, 7])
parts = np.array_split(numbers, 3)

print("Split into three approximately equal arrays:")
for part in parts:
    print(part)

matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Split into two column groups:\n", np.hsplit(matrix, 2))
