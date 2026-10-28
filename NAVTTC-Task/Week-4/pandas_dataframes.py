import pandas as pd

students = pd.DataFrame(
    {
        "name": ["Amina", "Bilal", "Sara"],
        "age": [19, 20, 18],
        "grade": ["A", "B", "A"],
    },
    index=["student_1", "student_2", "student_3"],
)

print("Complete DataFrame:\n", students)
print("Grade column:\n", students["grade"])
print("Row selected by label:\n", students.loc["student_2"])
print("Row selected by position:\n", students.iloc[0])
