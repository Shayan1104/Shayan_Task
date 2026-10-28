import pandas as pd

sales = pd.DataFrame(
    {
        "product": ["Book", "Pen", "Notebook", "Pencil"],
        "units_sold": [12, 30, 18, 25],
        "unit_price": [8.5, 1.5, 4.0, 1.0],
    }
)
sales["revenue"] = sales["units_sold"] * sales["unit_price"]

print("First rows:\n", sales.head())
print("Last two rows:\n", sales.tail(2))
print("Table dimensions:", sales.shape)
print("Column names:", sales.columns.tolist())
print("Numeric summary:\n", sales.describe())
print("Total revenue:", sales["revenue"].sum())
