import numpy as np

random_generator = np.random.default_rng(42)

print("Random integer from 0 to 9:", random_generator.integers(0, 10))
print("Three random decimal values:", random_generator.random(3))
print("Randomly selected values:", random_generator.choice([10, 20, 30, 40], size=3))
