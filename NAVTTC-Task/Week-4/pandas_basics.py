import pandas as pd

data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["Lahore", "Karachi", "Islamabad"],
}
people = pd.DataFrame(data)

print("DataFrame:\n", people)
print("First two rows:\n", people.head(2))
print("Summary of numeric columns:\n", people.describe())
