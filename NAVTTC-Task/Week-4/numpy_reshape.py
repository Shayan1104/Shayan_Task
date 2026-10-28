import numpy as np

numbers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

two_by_six = numbers.reshape(2, 6)
three_by_four = numbers.reshape(3, 4)
flattened = three_by_four.reshape(-1)

print("2 rows and 6 columns:\n", two_by_six)
print("3 rows and 4 columns:\n", three_by_four)
print("Flattened array:", flattened)
