import numpy as np

original = np.array([1, 2, 3, 4])
copied = original.copy()
viewed = original.view()

copied[0] = 10
print("Original after changing its copy:", original)
print("Copy:", copied)

viewed[1] = 20
print("Original after changing its view:", original)
print("View:", viewed)
print("Copy shares memory with original:", np.shares_memory(original, copied))
print("View shares memory with original:", np.shares_memory(original, viewed))
