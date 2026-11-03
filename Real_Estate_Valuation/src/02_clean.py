"""
02_clean.py - Stage 2: Data Cleaning & Preprocessing
--------------------------------------------------------------------------------
Author: Student ML Project
Dataset: zameenData.csv -> cleaned_plots.csv

Educational Purpose:
Real-world data is messy, inconsistent, and filled with noise. Machine learning
models follow the "Garbage In, Garbage Out" (GIGO) principle.
This script performs principled data cleaning:
  1. Scope Filtering: Subsets strictly to 'Residential Plot' and 'For Sale'.
  2. Row Repair: Explains and implements repairing misaligned installment rows
     rather than throwing them away, preserving valuable real estate records.
  3. Feature Standardization: Converts text with heterogeneous units (Lakh, Crore,
     Arab, Marla, Kanal) to continuous numerical features (PKR and Marla).
  4. Geospatial Text Extraction: Parses 'locality' and 'city' from location strings.
  5. Missing Value & Duplicate Handling: Drops non-physical values and identical duplicates.
  6. Outlier Removal: Applies Tukey's Interquartile Range (IQR) on log(price_per_marla).
  7. High Cardinality Regularization: Groups localities with < 10 occurrences into 'Other'.
  8. Step-by-step logging and output saving.
--------------------------------------------------------------------------------
"""

import os
import re
import numpy as np
import pandas as pd


def repair_misaligned_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies and repairs misaligned rows caused by installment plan attributes.
    
    Why repair instead of drop?
    - Dropping 95 valid residential plot listings loses useful variance in the dataset.
    - Inspection in Stage 1 revealed a deterministic shift:
        location  <- holds down payment
        area      <- holds installment period (e.g., '36 Months')
        pupose    <- holds the real location string (e.g., 'GT Road, Lahore, Punjab')
        added_days<- holds the real land area (e.g., '7 Marla', '1 Kanal')
    - Because installment plots are fundamentally 'For Sale', we can cleanly re-assign
      these columns and salvage all 95 listings without introducing synthetic or guessed values.
    """
    df = df.copy()
    shift_mask = (df['land_type'] == 'Residential Plot') & (df['area'].astype(str).str.contains('Months', case=False, na=False))
    shifted_count = shift_mask.sum()
    print(f"  [REPAIR] Detected {shifted_count} shifted Residential Plot rows.")

    if shifted_count > 0:
        # Transfer the genuine location and area from shifted columns
        df.loc[shift_mask, 'location'] = df.loc[shift_mask, 'pupose']
        df.loc[shift_mask, 'area'] = df.loc[shift_mask, 'added_days']
        df.loc[shift_mask, 'pupose'] = 'For Sale'
        print(f"  [REPAIR] Successfully realigned {shifted_count} rows: restored area and location.")
        
    return df


def parse_price_pkr(price_val) -> float:
    """
    Parses a price string with units into an absolute Pakistani Rupee (PKR) float.
    
    Conversions:
      - 1 Thousand = 1,000 PKR
      - 1 Lakh = 100,000 PKR (1e5)
      - 1 Crore = 10,000,000 PKR (1e7)
      - 1 Arab = 1,000,000,000 PKR (1e9)
    """
    if pd.isna(price_val):
        return np.nan
    s = str(price_val).strip()
    match = re.search(r'PKR\s*([\d\.]+)\s*(Crore|Lakh|Thousand|Arab)?', s, re.IGNORECASE)
    if not match:
        return np.nan
    
    number = float(match.group(1))
    unit = (match.group(2) or '').lower()
    
    if unit == 'crore':
        return number * 1e7
    elif unit == 'lakh':
        return number * 1e5
    elif unit == 'thousand':
        return number * 1e3
    elif unit == 'arab':
        return number * 1e9
    else:
        return number


def parse_area_marla(area_val) -> float:
    """
    Parses land area string with units into numeric Marla.
    
    Conversions:
      - 1 Marla = 1.0 Marla (Standard unit in Punjab, Pakistan: ~225 sq ft)
      - 1 Kanal = 20.0 Marla
      - 1 Sq. Yd. = 1 / 25 Marla (approx 0.04 Marla)
    """
    if pd.isna(area_val):
        return np.nan
    s = str(area_val).strip()
    match = re.search(r'([\d\.]+)\s*(Marla|Kanal|Sq\.\s*Yd\.)?', s, re.IGNORECASE)
    if not match:
        return np.nan
    
    number = float(match.group(1))
    unit = (match.group(2) or '').lower()
    
    if 'kanal' in unit:
        return number * 20.0
    elif 'sq' in unit:
        # 1 Marla is roughly 25 Sq. Yds. in standard LDA measurement
        return number / 25.0
    else:
        return number


def extract_locality_and_city(location_val):
    """
    Parses location text structured like 'Locality, City, Province'
    Example: 'Garden Town, Lahore, Punjab' -> locality: 'Garden Town', city: 'Lahore'
    """
    if pd.isna(location_val):
        return 'Unknown', 'Unknown'
    parts = [p.strip() for p in str(location_val).split(',')]
    if len(parts) >= 2:
        locality = parts[0]
        city = parts[-2]
    elif len(parts) == 1:
        locality = parts[0]
        city = parts[0]
    else:
        locality, city = 'Unknown', 'Unknown'
    return locality, city


def clean_zameen_data(raw_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Executes the comprehensive cleaning workflow with logging at each transformation.
    """
    print("\n" + "=" * 80)
    print(" STAGE 2: DATA CLEANING PIPELINE ")
    print("=" * 80)

    # 1. Load Data
    df_raw = pd.read_csv(raw_csv_path)
    initial_count = len(df_raw)
    print(f"Step 1: Loaded raw dataset. Total rows = {initial_count:,}")

    # 2. Repair misaligned rows
    print("\nStep 2: Addressing misaligned / shifted installment listings...")
    df_repaired = repair_misaligned_rows(df_raw)

    # 3. Filter by Land Type and Purpose
    print("\nStep 3: Subsetting strictly to Residential Plots For Sale...")
    df_filtered = df_repaired[
        (df_repaired['land_type'] == 'Residential Plot') & 
        (df_repaired['pupose'] == 'For Sale')
    ].copy()
    count_after_filter = len(df_filtered)
    print(f"  Rows remaining after filter: {count_after_filter:,} "
          f"({initial_count - count_after_filter:,} dropped: non-residential or rental).")

    # 4. Parse Price and Area
    print("\nStep 4: Converting textual units to standard numeric quantities...")
    df_filtered['price'] = df_filtered['price'].apply(parse_price_pkr)
    df_filtered['area'] = df_filtered['area'].apply(parse_area_marla)

    # 5. Extract Locality and City
    print("\nStep 5: Extracting Locality and City from location strings...")
    loc_and_city = df_filtered['location'].apply(extract_locality_and_city)
    df_filtered['locality'] = [x[0] for x in loc_and_city]
    df_filtered['city'] = [x[1] for x in loc_and_city]

    # 6. Handle Invalid Values (NaN, zero, negative prices or areas)
    print("\nStep 6: Handling missing and invalid values...")
    valid_mask = (
        df_filtered['price'].notnull() & (df_filtered['price'] > 1000) &
        df_filtered['area'].notnull() & (df_filtered['area'] > 0) &
        (df_filtered['locality'] != 'Unknown')
    )
    df_valid = df_filtered[valid_mask].copy()
    count_after_valid = len(df_valid)
    print(f"  Rows remaining after invalid removal: {count_after_valid:,} "
          f"({count_after_filter - count_after_valid:,} dropped: missing/zero/token prices).")

    # 7. Deduplication
    print("\nStep 7: Removing duplicate property listings...")
    df_dedup = df_valid.drop_duplicates(subset=['price', 'area', 'locality', 'city']).copy()
    count_after_dedup = len(df_dedup)
    duplicates_dropped = count_after_valid - count_after_dedup
    print(f"  Rows remaining after deduplication: {count_after_dedup:,} "
          f"({duplicates_dropped:,} duplicate listings dropped).")

    # 8. Outlier Detection using IQR on log(Price per Marla)
    print("\nStep 8: Detecting and filtering price-per-marla outliers using IQR...")
    print("  Educational note: Real estate prices naturally follow a log-normal distribution.")
    print("  Applying Tukey's IQR rule on log(Price per Marla) identifies listings that are")
    print("  either typos (e.g. 5 PKR) or extreme speculative outliers, without discarding")
    print("  genuinely expensive premium localities.")
    
    df_dedup['price_per_marla'] = df_dedup['price'] / df_dedup['area']
    df_dedup['log_ppm'] = np.log(df_dedup['price_per_marla'])

    q1 = df_dedup['log_ppm'].quantile(0.25)
    q3 = df_dedup['log_ppm'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (df_dedup['log_ppm'] >= lower_bound) & (df_dedup['log_ppm'] <= upper_bound)
    df_clean = df_dedup[outlier_mask].copy()
    count_after_outliers = len(df_clean)
    outliers_dropped = count_after_dedup - count_after_outliers
    print(f"  IQR Bounds for log(Price/Marla): [{lower_bound:.3f}, {upper_bound:.3f}]")
    print(f"  Equivalent Price/Marla Range: [{np.exp(lower_bound):,.0f} PKR, {np.exp(upper_bound):,.0f} PKR]")
    print(f"  Outliers dropped: {outliers_dropped:,} | Clean rows remaining: {count_after_outliers:,}")

    # 9. Group Low-Frequency Localities into 'Other'
    print("\nStep 9: Grouping rare localities (< 10 listings) into 'Other'...")
    locality_counts = df_clean['locality'].value_counts()
    frequent_localities = locality_counts[locality_counts >= 10].index
    rare_count = (locality_counts < 10).sum()
    df_clean['locality'] = df_clean['locality'].apply(lambda x: x if x in frequent_localities else 'Other')
    print(f"  Total distinct localities retained as specific categories: {len(frequent_localities)}")
    print(f"  Rare localities ({rare_count}) pooled into 'Other' bucket.")

    # 10. Compute Log Features for Modeling Convenience
    df_clean['log_price'] = np.log(df_clean['price'])
    df_clean['log_area'] = np.log(df_clean['area'])

    # Final feature selection for clean output
    final_cols = ['price', 'area', 'locality', 'city', 'price_per_marla', 'log_price', 'log_area', 'location']
    df_final = df_clean[final_cols].reset_index(drop=True)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Cleaned dataset saved to: {output_csv_path}")
    print(f"Final Cleaned Dataset Dimensions: {df_final.shape[0]:,} rows x {df_final.shape[1]} columns")

    print("\n--- Summary of Cleaning Log ---")
    print(f"Initial raw rows:                    {initial_count:>7,}")
    print(f"Residential Plots (For Sale):        {count_after_filter:>7,}")
    print(f"Valid price & area:                  {count_after_valid:>7,}")
    print(f"Unique listings (post-dedup):        {count_after_dedup:>7,}")
    print(f"Post-IQR Outlier Filter:             {count_after_outliers:>7,}")
    print("=" * 80 + "\n")

    return df_final


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, 'data', 'zameenData.csv')
    out_path = os.path.join(base_dir, 'data', 'cleaned_plots.csv')

    if not os.path.exists(raw_path):
        raw_path = os.path.join(os.path.dirname(base_dir), 'zameenData.csv')

    clean_zameen_data(raw_path, out_path)


if __name__ == '__main__':
    main()
