"""
=============================================================================
STAGE 3: EXPLORATORY DATA ANALYSIS (EDA) WITH CHARTS
=============================================================================
Goal: Visualize the cleaned data to understand distributions, relationships,
      and patterns BEFORE building models.

Why this matters: Visualization reveals patterns that statistics alone miss.
For example, a scatter plot might show a non-linear relationship that tells
us polynomial regression could work better than linear.
=============================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend (saves to file, no window)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a clean visual style for all plots
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)      # default figure size
plt.rcParams["figure.dpi"] = 120               # resolution

# Create a folder to save chart images
os.makedirs("charts", exist_ok=True)

# --------------------------------------------------------------------------
# Load the cleaned data from Stage 2
# --------------------------------------------------------------------------
df = pd.read_csv("clean_data.csv")
print(f"Loaded {len(df):,} clean rows")
print(f"Columns: {list(df.columns)}")
print()

# ==========================================================================
# CHART 1: Price Distribution (Raw)
# ==========================================================================
# A histogram shows how many listings fall into each price range.
# This helps us see if prices are evenly spread or skewed.
fig, ax = plt.subplots()
ax.hist(df["price_pkr"] / 1e7, bins=50, color="#2196F3", edgecolor="white", alpha=0.85)
ax.set_xlabel("Price (Crore PKR)")
ax.set_ylabel("Number of Listings")
ax.set_title("Chart 1: Price Distribution (Raw Scale)")
fig.tight_layout()
fig.savefig("charts/chart1_price_distribution_raw.png")
plt.close()
print("Chart 1 saved: charts/chart1_price_distribution_raw.png")
print("""
INTERPRETATION (Chart 1):
  The distribution is extremely right-skewed — most listings are cheap, but
  a few very expensive ones stretch the x-axis far to the right. This makes
  it hard to see the detail in the majority of the data. This is typical for
  real estate data and is why we need a log transform.
""")

# ==========================================================================
# CHART 2: Price Distribution (Log Scale)
# ==========================================================================
# Taking log10 of the price compresses the scale and makes the distribution
# more symmetric (closer to a bell curve / normal distribution).
fig, ax = plt.subplots()
ax.hist(df["log_price"], bins=50, color="#4CAF50", edgecolor="white", alpha=0.85)
ax.set_xlabel("log₁₀(Price in PKR)")
ax.set_ylabel("Number of Listings")
ax.set_title("Chart 2: Price Distribution (Log Scale)")
fig.tight_layout()
fig.savefig("charts/chart2_price_distribution_log.png")
plt.close()
print("Chart 2 saved: charts/chart2_price_distribution_log.png")
print("""
INTERPRETATION (Chart 2):
  On the log scale, the distribution looks much more symmetric and closer
  to a bell curve. Most plots are priced between 10^6.5 (~30 Lakh) and
  10^8 (~10 Crore). This symmetric shape is much better for regression
  models, which assume (or work better with) normally distributed errors.
""")

# ==========================================================================
# CHART 3: Area Distribution
# ==========================================================================
fig, ax = plt.subplots()
ax.hist(df["area_marla"], bins=50, color="#FF9800", edgecolor="white", alpha=0.85)
ax.set_xlabel("Area (Marla)")
ax.set_ylabel("Number of Listings")
ax.set_title("Chart 3: Area Distribution")
fig.tight_layout()
fig.savefig("charts/chart3_area_distribution.png")
plt.close()
print("Chart 3 saved: charts/chart3_area_distribution.png")
print("""
INTERPRETATION (Chart 3):
  Most residential plots are between 3-20 Marla, with peaks at common
  standard sizes like 5, 7, 10 Marla and 1 Kanal (20 Marla). There are
  some larger plots (1+ Kanal) but they are relatively rare.
""")

# ==========================================================================
# CHART 4: Price vs Area Scatter (Log Scales)
# ==========================================================================
# This is perhaps the most important chart: it shows whether bigger plots
# cost more, and what the relationship looks like.
fig, ax = plt.subplots()
ax.scatter(df["area_marla"], df["price_pkr"],
           alpha=0.3, s=15, c="#E91E63", edgecolors="none")
ax.set_xscale("log")  # log scale on x-axis (area)
ax.set_yscale("log")  # log scale on y-axis (price)
ax.set_xlabel("Area (Marla) — log scale")
ax.set_ylabel("Price (PKR) — log scale")
ax.set_title("Chart 4: Price vs Area (Log-Log Scatter)")
fig.tight_layout()
fig.savefig("charts/chart4_price_vs_area_scatter.png")
plt.close()
print("Chart 4 saved: charts/chart4_price_vs_area_scatter.png")
print("""
INTERPRETATION (Chart 4):
  There is a clear positive relationship: larger plots tend to cost more.
  On log-log scales this relationship looks roughly linear, which means
  the actual relationship is a power law (price ≈ c × area^k). The
  vertical spread at each area shows that location matters a lot — same-size
  plots in different localities have very different prices.
""")

# ==========================================================================
# CHART 5: Median Price by Top 15 Localities
# ==========================================================================
# Which neighborhoods are the most and least expensive?
top_localities = df.groupby("locality")["price_pkr"].median().sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(12, 6))
top_localities_sorted = top_localities.sort_values(ascending=True)  # for horizontal bar
bars = ax.barh(top_localities_sorted.index, top_localities_sorted.values / 1e7,
               color="#9C27B0", alpha=0.85)
ax.set_xlabel("Median Price (Crore PKR)")
ax.set_title("Chart 5: Median Price by Top 15 Localities")
ax.tick_params(axis='y', labelsize=9)
fig.tight_layout()
fig.savefig("charts/chart5_median_price_by_locality.png")
plt.close()
print("Chart 5 saved: charts/chart5_median_price_by_locality.png")
print("""
INTERPRETATION (Chart 5):
  Localities like DHA, Gulberg, and Model Town tend to have the highest
  median plot prices, reflecting their premium location status in Lahore.
  This confirms that 'locality' is an important feature for predicting
  price — the same-sized plot can cost 5-10x more in an expensive area.
""")

# ==========================================================================
# CHART 6: Correlation Heatmap of Numeric Features
# ==========================================================================
# A heatmap shows how strongly each pair of numeric variables is correlated.
# Values close to +1 or -1 mean strong relationships.
numeric_cols = ["area_marla", "price_pkr", "price_per_marla", "log_price", "log_area"]
corr_matrix = df[numeric_cols].corr()  # compute correlation matrix

fig, ax = plt.subplots(figsize=(8, 6))
# annot=True writes the correlation value inside each cell
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Chart 6: Correlation Heatmap of Numeric Features")
fig.tight_layout()
fig.savefig("charts/chart6_correlation_heatmap.png")
plt.close()
print("Chart 6 saved: charts/chart6_correlation_heatmap.png")
print("""
INTERPRETATION (Chart 6):
  log_area and log_price have a strong positive correlation, confirming that
  bigger plots cost more. area_marla and price_pkr are also positively
  correlated but the correlation may be weaker in raw scale due to the
  skewed distributions. price_per_marla is less correlated with area,
  suggesting that unit rates depend more on location than on plot size.
""")

# ==========================================================================
# CHART 7: Boxplot of Price by Top Localities
# ==========================================================================
# Boxplots show the median, quartiles, and outliers for each group.
# This helps us see the price RANGE in each locality.
top_n = 10
top_loc_names = df["locality"].value_counts().head(top_n).index.tolist()
df_top = df[df["locality"].isin(top_loc_names)]

fig, ax = plt.subplots(figsize=(14, 6))
# order by median price for better readability
order = df_top.groupby("locality")["price_pkr"].median().sort_values(ascending=False).index
sns.boxplot(data=df_top, x="locality", y="price_pkr", order=order,
            palette="Set2", ax=ax, showfliers=True)
ax.set_ylabel("Price (PKR)")
ax.set_title(f"Chart 7: Price Distribution by Top {top_n} Localities (Boxplot)")
ax.tick_params(axis='x', rotation=45)
# Format y-axis in Crore for readability
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e7:.0f} Cr'))
fig.tight_layout()
fig.savefig("charts/chart7_boxplot_price_by_locality.png")
plt.close()
print("Chart 7 saved: charts/chart7_boxplot_price_by_locality.png")
print("""
INTERPRETATION (Chart 7):
  Premium localities like DHA Defence show both higher median prices AND
  wider ranges (more variability). Some localities like Bahria Town have
  many listings with varied prices. The outlier dots show exceptionally
  priced plots that are unusual even for their locality.
""")

# ==========================================================================
# WHY PRICE NEEDS A LOG TRANSFORM FOR MODELLING
# ==========================================================================
print("=" * 60)
print("WHY WE NEED LOG TRANSFORM FOR MODELLING")
print("=" * 60)
print("""
Raw property prices are heavily RIGHT-SKEWED: most plots cost 20 Lakh to
2 Crore, but a few cost 50+ Crore. This causes three problems for ML models:

1. LARGE ERRORS DOMINATE: A model that's off by 5 Crore on a 50 Crore plot
   is penalized the same as being off by 5 Crore on a 50 Lakh plot. But 
   the first is a 10% error while the second is a 1000% error! Log transform
   makes the model focus on PROPORTIONAL accuracy.

2. NON-NORMAL RESIDUALS: Linear regression assumes errors are normally
   distributed. With skewed prices, the errors will also be skewed, violating
   this assumption. Log-transformed prices produce more symmetric errors.

3. NON-LINEAR RELATIONSHIPS: The relationship between area and price is
   multiplicative, not additive. A 10 Marla plot isn't "5 Marla more" than
   a 5 Marla plot in price — it's roughly "2x the price". Logs convert
   multiplication to addition, which linear models can handle.

In summary: log(price) is more normally distributed, makes relationships 
more linear, and ensures proportional errors — all of which help ML models
perform better.
""")

print("\nAll charts saved to the 'charts/' folder!")
