import numpy as np

numbers = np.array([1, 2, 3, 4, 5, 4, 4])

print("Indices containing 4:", np.where(numbers == 4))
print("Indices containing even numbers:", np.where(numbers % 2 == 0))
print("Insertion index for 4 in a sorted array:", np.searchsorted(np.sort(numbers), 4))
