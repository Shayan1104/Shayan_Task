import pandas as pd

raw_data = pd.DataFrame(
    {
        "name": ["Amina", "Bilal", None, "Sara", "Sara"],
        "age": [19, None, 21, 18, 18],
        "city": ["Lahore", "Karachi", "Lahore", None, None],
    }
)

cleaned_data = raw_data.drop_duplicates().copy()
cleaned_data["name"] = cleaned_data["name"].fillna("Unknown")
cleaned_data["age"] = cleaned_data["age"].fillna(cleaned_data["age"].mean())
cleaned_data["city"] = cleaned_data["city"].fillna("Not specified")

print("Original data:\n", raw_data)
print("Cleaned data:\n", cleaned_data)
