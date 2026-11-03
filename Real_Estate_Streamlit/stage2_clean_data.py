"""
=============================================================================
STAGE 2: CLEAN THE DATA
=============================================================================
Goal: Transform the raw, messy data into a clean DataFrame ready for
      analysis and modelling. Every decision is logged with before/after
      row counts.

Why this matters: ML models need clean, numeric data. Text prices like
"PKR25 Crore" can't be fed into a regression model directly.
=============================================================================
"""

import pandas as pd
import numpy as np
import os

# --------------------------------------------------------------------------
# 2.0  Load raw data
# --------------------------------------------------------------------------
df = pd.read_csv("zameenData.csv", index_col=0)
initial_rows = len(df)
print(f"Starting rows: {initial_rows:,}")
print()

# We'll keep a log of how many rows we remove at each step
cleaning_log = []  # list of (step_name, rows_removed, rows_remaining)

# --------------------------------------------------------------------------
# 2.1  Drop fully empty rows
# --------------------------------------------------------------------------
# Rows where EVERY column is NaN are useless — they contain no information.
before = len(df)
df = df.dropna(how="all")  # how='all' means drop only if ALL values are NaN
after = len(df)
cleaning_log.append(("Drop fully empty rows", before - after, after))
print(f"Step 2.1 — Dropped {before - after} fully empty rows. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.2  Remove MISALIGNED rows
# --------------------------------------------------------------------------
# In Stage 1 we discovered some rows have extra price columns scraped from
# the website that shift all values to the right. We can detect these because
# the 'location' column contains "PKR" (a price value) instead of a location.
#
# DECISION: DROP these rows. Why? Because the columns are shifted — the price,
# location, baths, area, and purpose fields are all in the wrong columns.
# We CANNOT reliably reconstruct the correct values, so dropping is safer
# than guessing.
before = len(df)
misaligned_mask = df["location"].astype(str).str.contains("PKR", na=False)
print(f"\nStep 2.2 — Found {misaligned_mask.sum()} misaligned rows (location contains 'PKR')")
print("Example misaligned row:")
if misaligned_mask.sum() > 0:
    print(df[misaligned_mask].head(1).to_string())
df = df[~misaligned_mask]  # keep only rows where location does NOT contain PKR
after = len(df)
cleaning_log.append(("Drop misaligned rows", before - after, after))
print(f"Dropped {before - after} misaligned rows. Remaining: {after:,}")

# Also check for rows where 'pupose' (purpose) has unexpected values
# Valid values should be "For Sale" or "For Rent" only
valid_purposes = ["For Sale", "For Rent"]
before = len(df)
bad_purpose_mask = ~df["pupose"].isin(valid_purposes) & df["pupose"].notna()
if bad_purpose_mask.sum() > 0:
    print(f"\nFound {bad_purpose_mask.sum()} more rows with invalid 'pupose' values:")
    print(df.loc[bad_purpose_mask, "pupose"].value_counts().head())
    df = df[~bad_purpose_mask]
    after = len(df)
    cleaning_log.append(("Drop bad purpose rows", before - after, after))
    print(f"Dropped {before - after} more rows. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.3  Filter: Keep only Residential Plots For Sale
# --------------------------------------------------------------------------
# The assignment asks us to predict prices of RESIDENTIAL PLOTS only.
# We also only want listings that are "For Sale" (not rentals).
before = len(df)
df = df[(df["land_type"] == "Residential Plot") & (df["pupose"] == "For Sale")]
after = len(df)
cleaning_log.append(("Keep only Residential Plot + For Sale", before - after, after))
print(f"\nStep 2.3 — Kept only Residential Plot + For Sale.")
print(f"Removed {before - after:,} non-matching rows. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.4  Convert PRICE to numeric PKR
# --------------------------------------------------------------------------
# Prices look like "PKR25 Crore", "PKR7.25 Crore", "PKR4.5 Lakh", etc.
# Format: PKR<number> <unit>
# Units and their multipliers:
#   Thousand = 1,000
#   Lakh     = 100,000
#   Crore    = 10,000,000
#   Arab     = 1,000,000,000
#
# Strategy: Extract the number and unit separately, then multiply.

def parse_price(price_str):
    """
    Convert a price string like 'PKR25 Crore' to a numeric value in PKR.
    Returns NaN if the string can't be parsed.
    """
    if pd.isna(price_str):  # handle missing values
        return np.nan
    
    price_str = str(price_str).strip()
    
    # Define the multiplier for each unit
    multipliers = {
        "Thousand": 1_000,
        "Lakh": 100_000,
        "Crore": 10_000_000,
        "Arab": 1_000_000_000
    }
    
    # Try to extract number and unit using string operations
    # Remove "PKR" prefix
    price_str = price_str.replace("PKR", "").strip()
    
    # Find which unit is in the string
    for unit, mult in multipliers.items():
        if unit in price_str:
            # Extract the number part (everything before the unit)
            num_str = price_str.replace(unit, "").strip()
            try:
                return float(num_str) * mult  # multiply number by unit
            except ValueError:
                return np.nan
    
    # If no unit found, try to parse as a plain number
    try:
        return float(price_str)
    except ValueError:
        return np.nan

# Apply the parsing function to every row
# .apply() runs a function on every element in a column
df["price_pkr"] = df["price"].apply(parse_price)

# Check for any prices that couldn't be parsed
failed_price = df["price_pkr"].isna()
print(f"\nStep 2.4 — Price parsing: {failed_price.sum()} values could not be parsed.")
if failed_price.sum() > 0:
    print("Unparseable prices:")
    print(df.loc[failed_price, "price"].value_counts().head(10))

# Drop rows where price couldn't be parsed (can't predict without a target!)
before = len(df)
df = df.dropna(subset=["price_pkr"])
after = len(df)
cleaning_log.append(("Drop unparseable prices", before - after, after))
print(f"Dropped {before - after} rows with unparseable prices. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.5  Convert AREA to numeric Marla
# --------------------------------------------------------------------------
# Areas look like "2 Kanal", "10 Marla", "200 Sq. Yd."
# Conversion rules:
#   1 Kanal = 20 Marla
#   1 Sq. Yd. ≈ 0.03673 Marla  (1 Marla = 272.25 sq ft = 30.25 sq yd)
#
# We standardize everything to Marla because it's the most common unit
# in Lahore real estate.

def parse_area(area_str):
    """
    Convert an area string like '2 Kanal' to numeric Marla.
    Returns NaN if the string can't be parsed.
    """
    if pd.isna(area_str):
        return np.nan
    
    area_str = str(area_str).strip()
    
    # Try each unit
    if "Kanal" in area_str:
        num_str = area_str.replace("Kanal", "").strip()
        try:
            return float(num_str) * 20  # 1 Kanal = 20 Marla
        except ValueError:
            return np.nan
    elif "Marla" in area_str:
        num_str = area_str.replace("Marla", "").strip()
        try:
            return float(num_str)  # already in Marla
        except ValueError:
            return np.nan
    elif "Sq. Yd." in area_str:
        num_str = area_str.replace("Sq. Yd.", "").strip()
        try:
            return float(num_str) / 30.25  # 1 Marla ≈ 30.25 sq yd
        except ValueError:
            return np.nan
    else:
        return np.nan

df["area_marla"] = df["area"].apply(parse_area)

# Check for parsing failures
failed_area = df["area_marla"].isna()
print(f"\nStep 2.5 — Area parsing: {failed_area.sum()} values could not be parsed.")
if failed_area.sum() > 0:
    print("Unparseable areas:")
    print(df.loc[failed_area, "area"].value_counts().head(10))

# Drop rows with unparseable area
before = len(df)
df = df.dropna(subset=["area_marla"])
after = len(df)
cleaning_log.append(("Drop unparseable areas", before - after, after))
print(f"Dropped {before - after} rows with unparseable areas. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.6  Extract LOCALITY and CITY from location
# --------------------------------------------------------------------------
# Location format: "Garden Town, Lahore, Punjab"
# We split by comma and take:
#   - First part: locality (neighborhood)
#   - Second part: city
#   - Third part: province (not needed)

def extract_locality(loc_str):
    """Extract the locality (first part before comma) from location string."""
    if pd.isna(loc_str):
        return np.nan
    parts = str(loc_str).split(",")
    if len(parts) >= 1:
        return parts[0].strip()  # first part is locality
    return np.nan

def extract_city(loc_str):
    """Extract the city (second part) from location string."""
    if pd.isna(loc_str):
        return np.nan
    parts = str(loc_str).split(",")
    if len(parts) >= 2:
        return parts[-2].strip()  # second-to-last is usually city
    return np.nan

df["locality"] = df["location"].apply(extract_locality)
df["city"] = df["location"].apply(extract_city)

print(f"\nStep 2.6 — Extracted locality and city.")
print(f"Top 10 localities:")
print(df["locality"].value_counts().head(10))
print(f"\nCities:")
print(df["city"].value_counts())

# --------------------------------------------------------------------------
# 2.7  Handle duplicates
# --------------------------------------------------------------------------
# Duplicate listings can bias our model — the model would "see" the same
# data point multiple times and think that pattern is more common.
before = len(df)
# Check duplicates on key columns (same property listed multiple times)
df = df.drop_duplicates(subset=["price_pkr", "area_marla", "locality", "city"], keep="first")
after = len(df)
cleaning_log.append(("Drop duplicates", before - after, after))
print(f"\nStep 2.7 — Dropped {before - after} duplicate rows. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.8  Handle missing values in remaining columns
# --------------------------------------------------------------------------
# At this point, the critical columns (price_pkr, area_marla, locality, city)
# should have no NaN values. Let's verify.
print(f"\nStep 2.8 — Missing values in key columns:")
key_cols = ["price_pkr", "area_marla", "locality", "city"]
print(df[key_cols].isnull().sum())

# Drop any remaining rows with missing key values
before = len(df)
df = df.dropna(subset=key_cols)
after = len(df)
cleaning_log.append(("Drop remaining missing values", before - after, after))
print(f"Dropped {before - after} rows. Remaining: {after:,}")

# --------------------------------------------------------------------------
# 2.9  Outlier detection using IQR on log(price_per_marla)
# --------------------------------------------------------------------------
# Why log scale? Because property prices are heavily right-skewed (a few
# very expensive properties pull the mean way up). Working on a log scale
# makes the distribution more symmetric, so the IQR rule works better.
#
# IQR rule: Any value below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.
# Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 - Q1.

# First, compute price per Marla (unit rate)
df["price_per_marla"] = df["price_pkr"] / df["area_marla"]

# Take the log (base 10) of price per Marla
df["log_ppm"] = np.log10(df["price_per_marla"])

# Calculate IQR on the log scale
Q1 = df["log_ppm"].quantile(0.25)  # 25th percentile
Q3 = df["log_ppm"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1                       # interquartile range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nStep 2.9 — Outlier detection (IQR on log10 price-per-Marla):")
print(f"  Q1 = {Q1:.4f}, Q3 = {Q3:.4f}, IQR = {IQR:.4f}")
print(f"  Lower bound = {lower_bound:.4f} (= 10^{lower_bound:.2f} = PKR {10**lower_bound:,.0f} per Marla)")
print(f"  Upper bound = {upper_bound:.4f} (= 10^{upper_bound:.2f} = PKR {10**upper_bound:,.0f} per Marla)")

# Count outliers
outliers = (df["log_ppm"] < lower_bound) | (df["log_ppm"] > upper_bound)
print(f"  Outliers found: {outliers.sum()} out of {len(df)} ({outliers.mean()*100:.1f}%)")

# Remove outliers
before = len(df)
df = df[~outliers]
after = len(df)
cleaning_log.append(("Remove price-per-Marla outliers (IQR)", before - after, after))
print(f"  Removed {before - after} outliers. Remaining: {after:,}")

# Clean up temporary columns
df = df.drop(columns=["log_ppm"])

# --------------------------------------------------------------------------
# 2.10  Group rare localities into "Other"
# --------------------------------------------------------------------------
# If a locality has very few listings (< 10), the model can't learn much
# from it. Grouping them into "Other" reduces noise and prevents overfitting.
locality_counts = df["locality"].value_counts()
rare_localities = locality_counts[locality_counts < 10].index
num_rare = len(rare_localities)
rows_affected = df["locality"].isin(rare_localities).sum()

print(f"\nStep 2.10 — Grouping rare localities:")
print(f"  {num_rare} localities have fewer than 10 listings")
print(f"  {rows_affected} rows will be relabeled as 'Other'")

df.loc[df["locality"].isin(rare_localities), "locality"] = "Other"

print(f"  Remaining unique localities: {df['locality'].nunique()}")

# --------------------------------------------------------------------------
# 2.11  Select final columns and save
# --------------------------------------------------------------------------
# We only keep the columns needed for EDA and modelling.
df_clean = df[["locality", "city", "area_marla", "price_pkr", "price_per_marla"]].copy()

# Add a log_price column (useful for modelling and visualization)
df_clean["log_price"] = np.log10(df_clean["price_pkr"])
df_clean["log_area"] = np.log10(df_clean["area_marla"])

print(f"\nFinal clean dataset shape: {df_clean.shape}")
print(df_clean.head(10).to_string())
print()
print(df_clean.describe().to_string())

# Save to CSV for the next stages
df_clean.to_csv("clean_data.csv", index=False)
print(f"\nSaved clean data to clean_data.csv")

# --------------------------------------------------------------------------
# 2.12  Print cleaning log
# --------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CLEANING LOG — ROW REMOVAL SUMMARY")
print("=" * 60)
print(f"{'Step':<45} {'Removed':>8} {'Remaining':>10}")
print("-" * 65)
for step, removed, remaining in cleaning_log:
    print(f"{step:<45} {removed:>8,} {remaining:>10,}")
print("-" * 65)
print(f"{'TOTAL REMOVED':<45} {initial_rows - len(df_clean):>8,} {len(df_clean):>10,}")
print(f"\nStarted with {initial_rows:,} rows, ended with {len(df_clean):,} rows.")
print(f"Removed {initial_rows - len(df_clean):,} rows ({(initial_rows - len(df_clean))/initial_rows*100:.1f}%).")
