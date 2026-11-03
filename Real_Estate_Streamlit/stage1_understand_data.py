"""
=============================================================================
STAGE 1: UNDERSTAND THE DATA
=============================================================================
Goal: Before doing anything, we look at the raw data to understand its
      structure, spot problems, and plan our cleaning strategy.

Why this matters: Real-world data is messy. If we jump straight to modelling
we'll get garbage results. "Garbage in, garbage out."
=============================================================================
"""

import pandas as pd   # pandas is the main library for tabular data in Python
import numpy as np    # numpy is for numerical operations

# --------------------------------------------------------------------------
# 1.1  Load the CSV file
# --------------------------------------------------------------------------
# read_csv() reads a comma-separated file into a DataFrame (a 2D table).
# The first column in the CSV is an unnamed index, so we use index_col=0
# to tell pandas to treat it as the row index, not a data column.
df = pd.read_csv("zameenData.csv", index_col=0)

# --------------------------------------------------------------------------
# 1.2  Shape: how many rows and columns?
# --------------------------------------------------------------------------
print("=" * 60)
print("SHAPE OF THE DATASET")
print("=" * 60)
# .shape returns (number_of_rows, number_of_columns)
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print()
# INTERPRETATION: Each row is one property listing on Zameen.com.
# Each column is an attribute of that listing (price, area, etc.).

# --------------------------------------------------------------------------
# 1.3  Column names and data types
# --------------------------------------------------------------------------
print("=" * 60)
print("COLUMN NAMES AND DATA TYPES")
print("=" * 60)
# .dtypes shows what type of data pandas thinks each column holds.
# 'object' means text/string. If a number column shows 'object', it likely
# has non-numeric characters mixed in (like "PKR" or "Crore").
print(df.dtypes)
print()

# --------------------------------------------------------------------------
# 1.4  First 10 rows (head)
# --------------------------------------------------------------------------
print("=" * 60)
print("FIRST 10 ROWS")
print("=" * 60)
# .head(n) shows the first n rows. Useful to see actual values.
print(df.head(10).to_string())
print()

# --------------------------------------------------------------------------
# 1.5  Last 5 rows (tail)
# --------------------------------------------------------------------------
print("=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail(5).to_string())
print()

# --------------------------------------------------------------------------
# 1.6  Missing values per column
# --------------------------------------------------------------------------
print("=" * 60)
print("MISSING VALUES PER COLUMN")
print("=" * 60)
# .isnull() returns True/False for each cell; .sum() counts the Trues.
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_df)
print()
# INTERPRETATION: Columns with many missing values may need special handling.
# 'baths' and 'bed' will have '-' for plots (they are not houses), so those
# aren't truly "missing" — they just don't apply.

# --------------------------------------------------------------------------
# 1.7  Unique value counts per column
# --------------------------------------------------------------------------
print("=" * 60)
print("NUMBER OF UNIQUE VALUES PER COLUMN")
print("=" * 60)
# .nunique() counts distinct values in each column.
# Columns with very few unique values are likely categorical.
print(df.nunique())
print()

# --------------------------------------------------------------------------
# 1.8  Value counts of categorical columns
# --------------------------------------------------------------------------
# Let's look at the distribution of key categorical columns.

print("=" * 60)
print("VALUE COUNTS: land_type (property type)")
print("=" * 60)
# This tells us what kinds of properties are in the data: Houses, Plots, etc.
print(df["land_type"].value_counts())
print()

print("=" * 60)
print("VALUE COUNTS: pupose (purpose — note the typo in column name!)")
print("=" * 60)
# 'pupose' is misspelled (should be 'purpose'). This is common in real data.
# This column tells us if a listing is For Sale or For Rent.
print(df["pupose"].value_counts())
print()

print("=" * 60)
print("VALUE COUNTS: baths")
print("=" * 60)
print(df["baths"].value_counts().head(15))
print()

print("=" * 60)
print("VALUE COUNTS: bed")
print("=" * 60)
print(df["bed"].value_counts().head(15))
print()

print("=" * 60)
print("VALUE COUNTS: added_days (sample)")
print("=" * 60)
print(df["added_days"].value_counts().head(15))
print()

# --------------------------------------------------------------------------
# 1.9  Check for fully empty rows
# --------------------------------------------------------------------------
print("=" * 60)
print("FULLY EMPTY ROWS (all columns NaN)")
print("=" * 60)
# A row where every single value is NaN is useless.
fully_empty = df.isnull().all(axis=1).sum()
print(f"Number of fully empty rows: {fully_empty}")
print()

# --------------------------------------------------------------------------
# 1.10  Check for MISALIGNED / SHIFTED rows
# --------------------------------------------------------------------------
print("=" * 60)
print("CHECKING FOR MISALIGNED ROWS")
print("=" * 60)
# Some rows in this dataset have EXTRA columns scraped (e.g., multiple prices
# like "original price, discounted price, installment, duration") which
# PUSH the remaining columns to the right. This means:
#   - The 'location' column might contain a price like "PKR4 Crore"
#   - The 'baths' column might contain a price like "PKR52 Lakh"
#   - The 'area' column might contain a duration like "4 Months"
#
# We can detect these by checking if 'location' contains "PKR" (it shouldn't)
# or if 'pupose' is missing/unexpected.

# Check: does the 'location' column contain "PKR"?
location_has_pkr = df["location"].astype(str).str.contains("PKR", na=False)
print(f"Rows where 'location' contains 'PKR' (misaligned): {location_has_pkr.sum()}")
print()
if location_has_pkr.sum() > 0:
    print("Examples of misaligned rows:")
    print(df[location_has_pkr].head(5).to_string())
    print()

# Check: 'pupose' column should only be "For Sale" or "For Rent"
valid_purposes = ["For Sale", "For Rent"]
bad_purpose = ~df["pupose"].isin(valid_purposes) & df["pupose"].notna()
print(f"Rows with unexpected 'pupose' values: {bad_purpose.sum()}")
if bad_purpose.sum() > 0:
    print("Unexpected purpose values:")
    print(df.loc[bad_purpose, "pupose"].value_counts().head(10))
print()

# --------------------------------------------------------------------------
# 1.11  Price format inspection
# --------------------------------------------------------------------------
print("=" * 60)
print("PRICE FORMAT EXAMPLES")
print("=" * 60)
# Prices are stored as text like "PKR25 Crore", "PKR7.25 Crore", "PKR4.5 Lakh"
# We need to parse the number AND the unit (Thousand, Lakh, Crore, Arab).
print("Sample price values:")
print(df["price"].dropna().unique()[:20])
print()

# What units appear in the price column?
print("Price units found:")
# Extract the last word from each price string (should be the unit)
price_units = df["price"].dropna().astype(str).str.extract(r'(Thousand|Lakh|Crore|Arab)', expand=False)
print(price_units.value_counts())
print()

# --------------------------------------------------------------------------
# 1.12  Area format inspection
# --------------------------------------------------------------------------
print("=" * 60)
print("AREA FORMAT EXAMPLES")
print("=" * 60)
# Areas are like "2 Kanal", "10 Marla", "200 Sq. Yd." — text with units.
print("Sample area values:")
print(df["area"].dropna().unique()[:20])
print()

# What units appear?
print("Area units found:")
area_units = df["area"].dropna().astype(str).str.extract(r'(Kanal|Marla|Sq\. Yd\.)', expand=False)
print(area_units.value_counts())
print()

# --------------------------------------------------------------------------
# 1.13  Location / city inspection
# --------------------------------------------------------------------------
print("=" * 60)
print("LOCATION EXAMPLES AND CITIES")
print("=" * 60)
# Location is like "Garden Town, Lahore, Punjab"
# The format is: Locality, City, Province
# Let's extract city names.
cities = df["location"].dropna().astype(str).str.split(",").str[-2].str.strip()
print("Cities found:")
print(cities.value_counts().head(20))
print()

# --------------------------------------------------------------------------
# 1.14  Summary of findings
# --------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY OF STAGE 1 FINDINGS")
print("=" * 60)
print("""
1. SHAPE: ~20,572 listings with 10 columns.
2. DTYPES: All columns are 'object' (text) — even price and area need parsing.
3. COLUMN NAME TYPO: 'pupose' should be 'purpose'.
4. MIXED PROPERTY TYPES: Houses, Residential Plots, Commercial Plots, Flats,
   Upper Portions, etc. We only want "Residential Plot".
5. MIXED PURPOSE: Both "For Sale" and "For Rent" listings. We only want sales.
6. PRICE AS TEXT: Prices include "PKR" prefix and units (Thousand/Lakh/Crore/Arab).
   Must convert to numeric PKR values.
7. AREA AS TEXT: Areas use Marla, Kanal, and Sq. Yd. units.
   Must convert to a single unit (Marla).
8. MISALIGNED ROWS: Some rows have extra price columns that shift all values
   to the right. These rows have "PKR..." in the location column.
   Decision: DROP these rows (we can't reliably recover the correct values).
9. LOCATION FORMAT: "Locality, City, Province" — we can extract locality and city.
10. BATHS/BED: Many entries show '-' which is normal for plots (no rooms).
""")
