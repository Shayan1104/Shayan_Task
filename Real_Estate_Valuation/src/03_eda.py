"""
03_eda.py - Stage 3: Exploratory Data Analysis & Visualizations
--------------------------------------------------------------------------------
Author: Student ML Project
Input: data/cleaned_plots.csv
Output: eda_outputs/*.png

Educational Purpose:
Exploratory Data Analysis (EDA) allows us to visually inspect the distributions,
relationships, and mathematical characteristics of the data before modeling.
Key insights produced by this script:
  1. Why Target Transformation is Necessary:
     Real estate prices are heavily right-skewed with long fat tails. Fitting
     linear models directly to raw prices violates homoscedasticity (constant error variance)
     and makes MSE loss overly sensitive to high-end luxury properties.
     A log transform normalizes the distribution, compresses variance, and aligns
     with economic theory where price scales multiplicatively.
  2. Area Clustering:
     Plots in Pakistan are developed in standardized sizes (3 Marla, 5 Marla, 10 Marla, 1 Kanal = 20 Marla).
  3. Locality Premium:
     Location is the single biggest driver of price per unit area (e.g. DHA Defence vs suburban fringes).
--------------------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Configure modern, elegant styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def plot_price_distributions(df: pd.DataFrame, output_dir: str):
    """
    Plots Raw Price vs Log-Transformed Price distributions side by side.
    
    WHAT TO LOOK FOR:
    - Left plot (Raw Price): Severe positive (right) skew. Most listings cluster below 50M PKR,
      but a thin tail stretches to 100M+ PKR. Standard regression struggles here because
      errors on large values dominate the squared-error loss function.
    - Right plot (Log Price): Symmetrical, approximately Gaussian (bell-shaped) curve.
      Mean and median align closely. This proves mathematically why log(price) or log features
      stabilize model training and homoscedasticity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw Price (in Million PKR / Lakhs)
    sns.histplot(df['price'] / 1e6, kde=True, ax=axes[0], color='#2563eb', bins=35)
    axes[0].set_title('Raw Price Distribution (Severe Right Skew)', weight='bold')
    axes[0].set_xlabel('Price (Millions PKR)')
    axes[0].set_ylabel('Number of Listings')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Log Price
    sns.histplot(df['log_price'], kde=True, ax=axes[1], color='#10b981', bins=35)
    axes[1].set_title('Log-Transformed Price (Normal / Gaussian Profile)', weight='bold')
    axes[1].set_xlabel('log(Price in PKR)')
    axes[1].set_ylabel('Number of Listings')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Figure 1: Target Variable Transformation Analysis (Raw vs Log Price)', fontsize=14, weight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, '01_price_distribution_raw_vs_log.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SAVED] {path}")


def plot_area_distribution(df: pd.DataFrame, output_dir: str):
    """
    Plots plot area distribution in Marla.
    
    WHAT TO LOOK FOR:
    - Discrete spikes around standard Pakistan plot sizes:
      * 3 Marla (~675 sq ft) - affordable urban plots
      * 5 Marla (~1,125 sq ft) - standard medium family plot
      * 10 Marla (~2,250 sq ft) - large residential plot
      * 20 Marla (1 Kanal = 4,500 sq ft) - executive residential plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw Area up to 40 Marla (covers 98% of residential listings)
    filtered_area = df[df['area'] <= 40]
    sns.histplot(filtered_area['area'], bins=40, kde=False, ax=axes[0], color='#8b5cf6')
    axes[0].set_title('Area Distribution (Marla <= 40)', weight='bold')
    axes[0].set_xlabel('Area (Marla)')
    axes[0].set_ylabel('Listing Count')
    axes[0].axvline(5, color='#ef4444', linestyle='--', label='5 Marla')
    axes[0].axvline(10, color='#f59e0b', linestyle='--', label='10 Marla')
    axes[0].axvline(20, color='#10b981', linestyle='--', label='1 Kanal (20 Marla)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Log Area
    sns.histplot(df['log_area'], kde=True, ax=axes[1], color='#ec4899', bins=30)
    axes[1].set_title('Log-Transformed Area Distribution', weight='bold')
    axes[1].set_xlabel('log(Area in Marla)')
    axes[1].set_ylabel('Listing Count')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('Figure 2: Land Area Distribution & Standard Plot Dimensions', fontsize=14, weight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, '02_area_distribution.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SAVED] {path}")


def plot_price_vs_area_scatter(df: pd.DataFrame, output_dir: str):
    """
    Plots Price vs Area on Log-Log scales with trendline.
    
    WHAT TO LOOK FOR:
    - Strong positive linear relationship in log-log space: log(Price) = alpha + beta * log(Area).
    - The slope beta represents elasticity of price with respect to land area.
    - Vertical dispersion at fixed areas (e.g. at log(5 Marla)) reflects locality premium:
      a 5 Marla plot in DHA Defence costs 4-5x more than a 5 Marla plot in suburban peripheral schemes.
    """
    plt.figure(figsize=(10, 6))
    
    sns.regplot(
        x='log_area', y='log_price', data=df,
        scatter_kws={'alpha': 0.35, 'color': '#0284c7', 's': 20},
        line_kws={'color': '#dc2626', 'linewidth': 2, 'label': 'Log-Linear Fit'}
    )
    plt.title('Figure 3: Log(Price) vs Log(Area) Scatter with Regression Trendline', weight='bold')
    plt.xlabel('log(Area in Marla)')
    plt.ylabel('log(Price in PKR)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    path = os.path.join(output_dir, '03_price_vs_area_scatter.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SAVED] {path}")


def plot_locality_analyses(df: pd.DataFrame, output_dir: str):
    """
    Generates Locality Median Price bar chart and Locality Boxplots for top areas.
    
    WHAT TO LOOK FOR:
    - Immense variance across different neighborhoods: Established prime sectors (DHA Defence,
      Gulberg, Model Town) have much higher median prices than developing schemes.
    - Boxplots highlight within-locality variance (commercial proximity, corner plot premiums).
    """
    # Exclude 'Other' for cleaner visualization of specific sectors
    df_known = df[df['locality'] != 'Other']
    top_12_localities = df_known['locality'].value_counts().head(12).index
    df_top = df_known[df_known['locality'].isin(top_12_localities)]

    # 1. Median Price by Locality
    median_prices = df_top.groupby('locality')['price'].median().sort_values(ascending=True) / 1e7 # in Crores

    plt.figure(figsize=(11, 6))
    colors = sns.color_palette('viridis', len(median_prices))
    bars = plt.barh(median_prices.index, median_prices.values, color=colors)
    plt.title('Figure 4: Median Plot Price Across Top 12 Lahore Localities (in Crores PKR)', weight='bold')
    plt.xlabel('Median Price (Crore PKR, 1 Crore = 10,000,000 PKR)')
    plt.ylabel('Locality')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, f'{width:.2f} Cr', 
                 va='center', fontsize=9, weight='bold')

    plt.tight_layout()
    path_bar = os.path.join(output_dir, '04_median_price_top_localities.png')
    plt.savefig(path_bar, dpi=300)
    plt.close()
    print(f"  [SAVED] {path_bar}")

    # 2. Boxplot of Price Distribution for Top Localities
    plt.figure(figsize=(12, 7))
    sorted_locs = df_top.groupby('locality')['log_price'].median().sort_values(ascending=False).index
    sns.boxplot(
        x='log_price', y='locality', data=df_top, order=sorted_locs,
        palette='Spectral', showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"}
    )
    plt.title('Figure 5: Price Distribution Boxplots by Locality (Sorted by Median)', weight='bold')
    plt.xlabel('log(Price in PKR)')
    plt.ylabel('Locality')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    path_box = os.path.join(output_dir, '05_price_boxplot_top_localities.png')
    plt.savefig(path_box, dpi=300)
    plt.close()
    print(f"  [SAVED] {path_box}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """
    Computes and plots correlation matrix across numerical variables.
    
    WHAT TO LOOK FOR:
    - Extremely high positive correlation between log_price and log_area (~0.7 to 0.8).
    - Notice that log_price correlates much more strongly with log_area than raw price
      does with raw area, empirically validating the power of log transforms for linear models.
    """
    numeric_cols = ['price', 'area', 'price_per_marla', 'log_price', 'log_area']
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr, annot=True, fmt='.3f', cmap='Blues', vmin=-1, vmax=1,
        linewidths=1, square=True, cbar_kws={'shrink': 0.8}
    )
    plt.title('Figure 6: Pearson Correlation Heatmap of Numeric Features', weight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, '06_correlation_heatmap.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  [SAVED] {path}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv_path = os.path.join(base_dir, 'data', 'cleaned_plots.csv')
    output_dir = os.path.join(base_dir, 'eda_outputs')
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" STAGE 3: EXPLORATORY DATA ANALYSIS (EDA) ")
    print("=" * 80)
    
    if not os.path.exists(clean_csv_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_csv_path}. Run 02_clean.py first!")

    df = pd.read_csv(clean_csv_path)
    print(f"Loaded {len(df):,} cleaned plot listings for EDA visualization.")

    print("\nGenerating charts in eda_outputs/ ...")
    plot_price_distributions(df, output_dir)
    plot_area_distribution(df, output_dir)
    plot_price_vs_area_scatter(df, output_dir)
    plot_locality_analyses(df, output_dir)
    plot_correlation_heatmap(df, output_dir)

    print("\n" + "=" * 80)
    print(" [STAGE 3 COMPLETE] All 6 EDA Charts generated successfully!")
    print(" Check eda_outputs/ to review figures.")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
