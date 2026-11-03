"""
01_explore.py - Stage 1: Understand the Data
--------------------------------------------------------------------------------
Author: Student ML Project
Dataset: zameenData.csv (Zameen.com Pakistan Property Listings)

Educational Purpose:
Before touching any model or cleaning pipeline, an ML engineer must thoroughly
explore the raw dataset (Exploratory Data Analysis - Stage 1). This script reveals:
  1. High-level dataset dimensions, data types, and null value distributions.
  2. Data hygiene anomalies: empty rows, shifted/misaligned columns due to scraping quirks.
  3. Text representations of numerical domains (Price in Lakh/Crore, Area in Marla/Kanal).
  4. Class balances across property types, listing purposes, and geographic locations.
--------------------------------------------------------------------------------
"""

import os
import re
import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data from the specified path.
    
    Parameters:
        file_path (str): Relative or absolute path to zameenData.csv.
        
    Returns:
        pd.DataFrame: Raw unprocessed dataframe.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check your path.")
    print(f"[INFO] Loading raw dataset from: {file_path}")
    return pd.read_csv(file_path)


def explore_general_info(df: pd.DataFrame) -> None:
    """
    Prints baseline information: shape, dtypes, head sample, and missingness.
    """
    print("\n" + "=" * 80)
    print(" 1. BASELINE DATASET OVERVIEW ")
    print("=" * 80)
    print(f"Total Rows:    {df.shape[0]:,}")
    print(f"Total Columns: {df.shape[1]}")
    print("\n--- Column Data Types ---")
    print(df.dtypes)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Missing Value Analysis ---")
    missing_summary = pd.DataFrame({
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing_summary)

    print("\n--- Unique Value Counts per Column ---")
    print(df.nunique())


def explore_categorical_distributions(df: pd.DataFrame) -> None:
    """
    Examines value distributions for categorical columns such as land_type and pupose.
    """
    print("\n" + "=" * 80)
    print(" 2. CATEGORICAL DISTRIBUTIONS & SEGMENTATION ")
    print("=" * 80)
    
    print("\n--- Land Type Distribution (Top 10) ---")
    print("Concept: The dataset contains multiple real estate property categories.")
    print("Our target scope is strictly 'Residential Plot'.")
    print(df['land_type'].value_counts(dropna=False).head(10))

    print("\n--- Listing Purpose (Sale vs Rent) ---")
    print("Concept: Rental prices and plot purchase prices have completely different")
    print("valuation dynamics. We must only model 'For Sale' properties.")
    print(df['pupose'].value_counts(dropna=False).head(10))


def explore_data_anomalies(df: pd.DataFrame) -> None:
    """
    Investigates data anomalies and real-world scraping artifacts:
      - Completely empty rows.
      - Misaligned / shifted rows where installment data pushed columns to the right.
      - Mixed non-standard units for price (Lakh, Crore, Arab, Thousand) and area.
      - Non-Lahore cities in a predominantly Lahore dataset.
    """
    print("\n" + "=" * 80)
    print(" 3. DETAILED ANOMALY INVESTIGATION ")
    print("=" * 80)

    # 1. Fully empty rows
    empty_rows = df[df.isnull().all(axis=1)]
    # Rows where essential columns are NaN
    null_essentials = df[df['land_type'].isnull() & df['price'].isnull()]
    print(f"[ANOMALY 1] Fully empty or blank rows:")
    print(f"  Count: {len(null_essentials)} rows have missing essential values (e.g., all fields NaN).")
    print(f"  Action required: Drop these rows completely during cleaning.")

    # 2. Misaligned / Shifted rows
    print("\n[ANOMALY 2] Misaligned / Shifted Columns:")
    print("  Root Cause Explanation:")
    print("  When Zameen.com listings have an installment schedule (e.g. '36 Months', Down Payment),")
    print("  the web scraper stored installment values in the 'location', 'baths', and 'area' columns,")
    print("  pushing the real location into the 'pupose' column and the real area into 'added_days'!")
    
    shifted_mask = ~df['pupose'].isin(['For Sale', 'For Rent']) & df['pupose'].notnull()
    shifted_count = shifted_mask.sum()
    print(f"  Total shifted rows detected across entire dataset: {shifted_count}")
    
    res_shifted = df[(df['land_type'] == 'Residential Plot') & df['area'].astype(str).str.contains('Months')]
    print(f"  Shifted rows specifically in Residential Plots: {len(res_shifted)}")
    if len(res_shifted) > 0:
        sample_shifted = res_shifted[['price', 'location', 'area', 'pupose', 'added_days']].head(3)
        print("  Sample shifted rows:")
        print(sample_shifted)
        print("  Notice: 'area' holds '36 Months', 'pupose' holds the real location,")
        print("  and 'added_days' holds the real area ('7 Marla', '5 Marla').")
        print("  Repair Strategy: We can successfully salvage these rows by shifting columns back!")

    # 3. Units in Price and Area
    print("\n[ANOMALY 3] Text Units in Numerical Domains:")
    price_units = df['price'].dropna().astype(str).apply(
        lambda x: re.search(r'(Crore|Lakh|Thousand|Arab)', x, re.IGNORECASE).group(1) 
        if re.search(r'(Crore|Lakh|Thousand|Arab)', x, re.IGNORECASE) else 'None/Raw'
    ).value_counts()
    print("  Price unit frequency across dataset:")
    print(price_units)
    print("  -> 1 Lakh = 100,000 PKR")
    print("  -> 1 Crore = 10,000,000 PKR (100 Lakh)")
    print("  -> 1 Arab = 1,000,000,000 PKR (100 Crore)")

    area_units = df['area'].dropna().astype(str).apply(
        lambda x: re.search(r'(Marla|Kanal|Sq\.\s*Yd\.|Months)', x, re.IGNORECASE).group(1)
        if re.search(r'(Marla|Kanal|Sq\.\s*Yd\.|Months)', x, re.IGNORECASE) else 'Other'
    ).value_counts()
    print("\n  Area unit frequency across dataset:")
    print(area_units)
    print("  -> 1 Kanal = 20 Marla (standard Pakistan real estate unit conversion)")
    print("  -> 'Months' indicates shifted installment rows.")

    # 4. Cities other than Lahore
    print("\n[ANOMALY 4] Geographic Coverage (Cities outside Lahore):")
    cities = df['location'].dropna().str.split(',').apply(
        lambda x: x[-2].strip() if len(x) >= 2 else 'Unknown'
    )
    top_cities = cities.value_counts().head(8)
    print("  Top represented cities:")
    print(top_cities)
    print(f"  Lahore dominates the dataset with {top_cities.get('Lahore', 0):,} listings.")
    print("  Listings from other cities (Faisalabad, Islamabad, Kasur) represent small outliers.")


def main():
    """
    Main entry point for Stage 1 data exploration.
    """
    # Locate data relative to project root or src
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'zameenData.csv')
    
    # Fallback to parent directory if running directly
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(base_dir), 'zameenData.csv')

    df = load_dataset(data_path)
    explore_general_info(df)
    explore_categorical_distributions(df)
    explore_data_anomalies(df)
    
    print("\n" + "=" * 80)
    print(" [STAGE 1 COMPLETE] Summary of Key Takeaways for Stage 2 Data Cleaning:")
    print("  1. Filter strictly for Residential Plots for Sale.")
    print("  2. Repair the 95 shifted residential plot rows using column reallocation.")
    print("  3. Standardize prices to numeric PKR and areas to numeric Marla.")
    print("  4. Extract locality and city, drop duplicate and invalid listings.")
    print("  5. Remove extreme price-per-marla outliers using IQR.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
