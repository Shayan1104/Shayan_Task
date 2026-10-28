import pandas as pd

measurements = pd.DataFrame(
    {
        "hours_studied": [1, 2, 3, 4, 5],
        "practice_tests": [1, 1, 2, 3, 4],
        "exam_score": [50, 55, 65, 75, 90],
    }
)

print("Correlation matrix:\n", measurements.corr())
