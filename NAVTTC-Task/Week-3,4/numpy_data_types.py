import numpy as np

integers = np.array([1, 2, 3])
decimals = np.array([1.1, 2.2, 3.3])
four_byte_integers = np.array([1, 2, 3], dtype="i4")
converted_integers = decimals.astype("i")

print("Integer array type:", integers.dtype)
print("Decimal array type:", decimals.dtype)
print("Four-byte integer type:", four_byte_integers.dtype)
print("Converted values:", converted_integers)
print("Converted type:", converted_integers.dtype)
