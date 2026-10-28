from io import StringIO

import pandas as pd

csv_data = StringIO(
    "name,age,city\n"
    "Amina,19,Lahore\n"
    "Bilal,20,Karachi\n"
    "Sara,18,Islamabad\n"
)
students = pd.read_csv(csv_data)

print("Data loaded from CSV:\n", students)
print("Student names:", students["name"].tolist())
