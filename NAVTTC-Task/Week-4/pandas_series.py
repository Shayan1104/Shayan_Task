import pandas as pd

default_index = pd.Series([10, 20, 30])
named_index = pd.Series(
    [85, 90, 95],
    index=["English", "Math", "Science"],
)
from_dictionary = pd.Series({"Monday": 1, "Tuesday": 2, "Wednesday": 3})

print("Default index:\n", default_index)
print("Named index:\n", named_index)
print("Math score by label:", named_index.loc["Math"])
print("Second score by position:", named_index.iloc[1])
print("Series created from a dictionary:\n", from_dictionary)
