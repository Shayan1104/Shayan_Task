"""
=============================================================================
STAGE 4: TRAIN AND COMPARE MODELS
=============================================================================
Goal: Train 6 different ML models to predict property prices, evaluate them,
      compare their performance, and save each as a .pkl file.

Why Pipelines? A sklearn Pipeline bundles preprocessing (scaling, encoding)
with the model itself. This means:
1. No data leakage: fit_transform only on training data, transform on test.
2. Easy deployment: load one .pkl file and it handles everything.
3. Reproducibility: same preprocessing every time.
=============================================================================
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
import os
import joblib  # for saving and loading models as .pkl files
import json    # for saving results to a JSON file
import warnings
warnings.filterwarnings("ignore")  # suppress convergence warnings during tuning

# sklearn imports — each does one thing:
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.model_selection import GridSearchCV       # tunes hyperparameters
from sklearn.pipeline import Pipeline                  # chains preprocessing + model
from sklearn.compose import ColumnTransformer          # different transforms per column
from sklearn.preprocessing import (
    StandardScaler,       # standardizes numeric features (mean=0, std=1)
    OneHotEncoder,        # converts categorical text to binary columns
    PolynomialFeatures    # creates polynomial terms (x², x×y, etc.)
)
from sklearn.linear_model import LinearRegression      # classic linear regression
from sklearn.linear_model import LogisticRegression    # classifier (not regressor!)
from sklearn.neighbors import KNeighborsRegressor      # K-Nearest Neighbors
from sklearn.svm import SVR                            # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor         # Decision Tree
from sklearn.metrics import (
    r2_score,               # R² — how much variance is explained (1.0 = perfect)
    mean_absolute_error,    # MAE — average absolute error in PKR
    root_mean_squared_error, # RMSE — penalizes large errors more
    accuracy_score,         # classification accuracy
    f1_score,               # harmonic mean of precision and recall
    confusion_matrix        # matrix of predicted vs actual classes
)

# --------------------------------------------------------------------------
# 4.0  Create models directory
# --------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

# --------------------------------------------------------------------------
# 4.1  Load cleaned data
# --------------------------------------------------------------------------
df = pd.read_csv("clean_data.csv")
print(f"Loaded {len(df):,} rows of clean data")
print(f"Columns: {list(df.columns)}")
print()

# --------------------------------------------------------------------------
# 4.2  Define features and target
# --------------------------------------------------------------------------
# FEATURES (X): What we use to predict
#   - area_marla: the size of the plot in Marla (numeric)
#   - log_area: log10 of area — captures the non-linear relationship
#   - locality: the neighborhood (categorical — needs one-hot encoding)
#
# TARGET (y): What we want to predict
#   - price_pkr: the listing price in Pakistani Rupees
#
# We use BOTH area_marla and log_area because:
# - area_marla captures the raw size
# - log_area captures the logarithmic/power-law relationship with price

numeric_features = ["area_marla", "log_area"]   # continuous numeric columns
categorical_features = ["locality"]              # categorical text column

X = df[numeric_features + categorical_features]  # feature matrix
y = df["price_pkr"]                               # target variable

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: PKR {y.min():,.0f} to PKR {y.max():,.0f}")
print()

# --------------------------------------------------------------------------
# 4.3  Train/Test Split (80/20)
# --------------------------------------------------------------------------
# We split the data into a TRAINING set (80%) and a TEST set (20%).
# - Training set: the model learns from this data.
# - Test set: we use this to evaluate how well the model generalizes to
#   data it has NEVER seen before.
#
# random_state=42 ensures the split is the same every time we run this code,
# making our results reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% for testing
    random_state=42        # reproducibility seed
)

print(f"Training set: {len(X_train):,} rows ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set:     {len(X_test):,} rows ({len(X_test)/len(X)*100:.0f}%)")
print()

# --------------------------------------------------------------------------
# 4.4  Build the PREPROCESSOR
# --------------------------------------------------------------------------
# The preprocessor handles two types of features differently:
#   - Numeric features → StandardScaler (subtract mean, divide by std)
#     This puts all numbers on the same scale, which is crucial for
#     distance-based algorithms like KNN and SVM.
#   - Categorical features → OneHotEncoder (create binary dummy columns)
#     E.g., locality="DHA" becomes locality_DHA=1, locality_Other=0, etc.
#
# ColumnTransformer applies different transformations to different columns.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),           # scale numbers
        ("cat", OneHotEncoder(handle_unknown="ignore",         # encode text
                              sparse_output=False), 
         categorical_features)
    ]
)

# --------------------------------------------------------------------------
# 4.5  Define helper function to evaluate regression models
# --------------------------------------------------------------------------
def evaluate_regression(model_name, pipeline, X_train, X_test, y_train, y_test):
    """
    Evaluates a regression model on both training and test data.
    Returns a dictionary of metrics.
    
    Metrics explained:
    - R² (R-squared): Proportion of variance in price that the model explains.
      1.0 = perfect, 0.0 = no better than predicting the mean.
    - MAE: Average absolute error in PKR. Easy to interpret.
    - RMSE: Square root of average squared error. Penalizes big mistakes more.
    """
    # Get predictions on training and test data
    y_train_pred = pipeline.predict(X_train)  # predict on data the model WAS trained on
    y_test_pred = pipeline.predict(X_test)    # predict on data the model NEVER saw
    
    # Calculate metrics for training data
    r2_train = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test data
    r2_test = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  R² (Train): {r2_train:.4f}")
    print(f"  R² (Test):  {r2_test:.4f}")
    print(f"  MAE:        PKR {mae:,.0f}")
    print(f"  RMSE:       PKR {rmse:,.0f}")
    
    # Check for overfitting: if train R² >> test R², the model memorized
    # the training data instead of learning general patterns.
    gap = r2_train - r2_test
    if gap > 0.1:
        print(f"  ⚠️  OVERFITTING WARNING: Train-Test R² gap = {gap:.4f}")
        print(f"     The model fits training data much better than test data.")
    elif gap > 0.05:
        print(f"  ⚡ Mild overfitting: Train-Test R² gap = {gap:.4f}")
    else:
        print(f"  ✅ Good generalization: Train-Test R² gap = {gap:.4f}")
    
    return {
        "model": model_name,
        "r2_train": round(r2_train, 4),
        "r2_test": round(r2_test, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "y_test_pred": y_test_pred.tolist(),
        "y_train_pred": y_train_pred.tolist()
    }

# Store results for all models
results = []

# ==========================================================================
# MODEL 1: LINEAR REGRESSION
# ==========================================================================
# Linear Regression finds the best straight-line (hyperplane) fit:
#   price = b0 + b1*area + b2*log_area + b3*locality_DHA + ...
# It minimizes the sum of squared errors between predicted and actual prices.
# PROS: Simple, interpretable, fast.
# CONS: Assumes a linear relationship; can't capture curves.

print("\n" + "#" * 60)
print("  MODEL 1: LINEAR REGRESSION")
print("#" * 60)

# Pipeline chains preprocessing → model in one object
lr_pipeline = Pipeline([
    ("preprocessor", preprocessor),      # step 1: scale + encode
    ("model", LinearRegression())        # step 2: fit linear model
])

lr_pipeline.fit(X_train, y_train)  # train the model on training data
result = evaluate_regression("Linear Regression", lr_pipeline, X_train, X_test, y_train, y_test)
results.append(result)

# Save the trained pipeline
joblib.dump(lr_pipeline, "models/linear_regression.pkl")  # serialize to disk
print("  Saved: models/linear_regression.pkl")

print("""
INTERPRETATION: Linear Regression provides a baseline. It assumes that price
changes linearly with area and locality. If R² is low, it means the
relationship is more complex than a straight line.
""")

# ==========================================================================
# MODEL 2: POLYNOMIAL REGRESSION (Degree 2)
# ==========================================================================
# Polynomial Regression adds squared and interaction terms:
#   price = b0 + b1*area + b2*area² + b3*log_area + b4*log_area² + ...
# This lets the model capture CURVED relationships.
# Degree 2 = quadratic terms. Higher degrees risk overfitting.

print("\n" + "#" * 60)
print("  MODEL 2: POLYNOMIAL REGRESSION (Degree 2)")
print("#" * 60)

# We need a custom preprocessor that adds polynomial features AFTER encoding
poly_pipeline = Pipeline([
    ("preprocessor", preprocessor),                           # step 1: scale + encode
    ("poly", PolynomialFeatures(degree=2,                     # step 2: add x², x*y terms
                                 include_bias=False,          # don't add a column of 1s
                                 interaction_only=False)),    # include squared terms
    ("model", LinearRegression())                             # step 3: fit linear model
])

poly_pipeline.fit(X_train, y_train)
result = evaluate_regression("Polynomial Regression (d=2)", poly_pipeline, X_train, X_test, y_train, y_test)
results.append(result)

joblib.dump(poly_pipeline, "models/polynomial_regression.pkl")
print("  Saved: models/polynomial_regression.pkl")

print("""
INTERPRETATION: Polynomial Regression can capture curves that Linear can't.
If R² improves over Linear, it confirms the price-area relationship is
non-linear. But watch for overfitting — the many extra features (squared
and interaction terms) might fit noise in the training data.
""")

# ==========================================================================
# MODEL 3: K-NEAREST NEIGHBORS (KNN) with GridSearchCV
# ==========================================================================
# KNN predicts the price of a new plot by finding the K most similar plots
# in the training data and averaging their prices.
# "Similar" is measured by distance in the feature space.
#
# Key hyperparameter: K (n_neighbors)
# - Small K → more flexible, risk of overfitting
# - Large K → smoother predictions, risk of underfitting
#
# GridSearchCV tries multiple values of K and picks the best one using
# cross-validation (splitting training data into sub-folds).

print("\n" + "#" * 60)
print("  MODEL 3: K-NEAREST NEIGHBORS (KNN)")
print("#" * 60)

knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", KNeighborsRegressor())
])

# Define the grid of hyperparameters to try
# We prefix the parameter name with "model__" because it's inside a Pipeline
knn_params = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15, 20],  # how many neighbors to average
    "model__weights": ["uniform", "distance"]           # equal weight vs closer=more weight
}

# GridSearchCV: try all combinations and pick the best via 5-fold cross-validation
knn_grid = GridSearchCV(
    knn_pipeline,
    knn_params,
    cv=5,               # 5-fold cross-validation
    scoring="r2",        # optimize for R² score
    n_jobs=-1,           # use all CPU cores for speed
    verbose=0
)

print("  Tuning KNN hyperparameters...")
knn_grid.fit(X_train, y_train)

print(f"  Best parameters: {knn_grid.best_params_}")
print(f"  Best CV R²: {knn_grid.best_score_:.4f}")

# Use the best model found by GridSearchCV
best_knn = knn_grid.best_estimator_
result = evaluate_regression("KNN", best_knn, X_train, X_test, y_train, y_test)
results.append(result)

joblib.dump(best_knn, "models/knn.pkl")
print("  Saved: models/knn.pkl")

print("""
INTERPRETATION: KNN is a non-parametric method — it doesn't assume any
particular mathematical form for the relationship. It can capture complex
patterns but struggles with high-dimensional data (many one-hot encoded
localities). The 'distance' weighting usually helps: closer neighbors
have more influence on the prediction.
""")

# ==========================================================================
# MODEL 4: SUPPORT VECTOR REGRESSION (SVR) with GridSearchCV
# ==========================================================================
# SVR tries to find a function that stays within an epsilon-tube around the
# data points. Points outside the tube are "support vectors" that define
# the model. The RBF kernel allows non-linear relationships.
#
# Key hyperparameters:
# - C: penalty for errors outside the tube (high C → tighter fit)
# - epsilon: width of the tube (smaller → tighter fit)

print("\n" + "#" * 60)
print("  MODEL 4: SUPPORT VECTOR REGRESSION (SVR)")
print("#" * 60)

svr_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", SVR(kernel="rbf"))  # RBF kernel for non-linear relationships
])

# SVR is computationally expensive, so we use a smaller grid
svr_params = {
    "model__C": [0.1, 1, 10, 100],       # regularization strength
    "model__epsilon": [0.01, 0.1, 0.5],   # tube width
}

svr_grid = GridSearchCV(
    svr_pipeline,
    svr_params,
    cv=3,               # 3-fold (faster than 5-fold for SVR)
    scoring="r2",
    n_jobs=-1,
    verbose=0
)

print("  Tuning SVR hyperparameters (this may take a minute)...")
svr_grid.fit(X_train, y_train)

print(f"  Best parameters: {svr_grid.best_params_}")
print(f"  Best CV R²: {svr_grid.best_score_:.4f}")

best_svr = svr_grid.best_estimator_
result = evaluate_regression("SVR", best_svr, X_train, X_test, y_train, y_test)
results.append(result)

joblib.dump(best_svr, "models/svr.pkl")
print("  Saved: models/svr.pkl")

print("""
INTERPRETATION: SVR with an RBF kernel can capture non-linear relationships
but is computationally expensive, especially with many features (all the
one-hot encoded localities). SVR works best when features are well-scaled
(our StandardScaler helps here). If R² is low, it may be because SVR
struggles with the high dimensionality of our one-hot encoded features.
""")

# ==========================================================================
# MODEL 5: DECISION TREE REGRESSOR with GridSearchCV
# ==========================================================================
# A Decision Tree splits the data into groups using if/else rules:
#   "If area > 10 and locality = DHA → predict 3 Crore"
# It learns these rules automatically from the data.
#
# Key hyperparameters:
# - max_depth: how many levels of splits (deeper → more complex)
# - min_samples_leaf: minimum samples in each leaf node

print("\n" + "#" * 60)
print("  MODEL 5: DECISION TREE REGRESSOR")
print("#" * 60)

dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeRegressor(random_state=42))
])

dt_params = {
    "model__max_depth": [5, 10, 15, 20, None],       # None = unlimited depth
    "model__min_samples_leaf": [1, 5, 10, 20, 50],    # minimum samples per leaf
    "model__min_samples_split": [2, 5, 10]             # minimum samples to split a node
}

dt_grid = GridSearchCV(
    dt_pipeline,
    dt_params,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=0
)

print("  Tuning Decision Tree hyperparameters...")
dt_grid.fit(X_train, y_train)

print(f"  Best parameters: {dt_grid.best_params_}")
print(f"  Best CV R²: {dt_grid.best_score_:.4f}")

best_dt = dt_grid.best_estimator_
result = evaluate_regression("Decision Tree", best_dt, X_train, X_test, y_train, y_test)
results.append(result)

joblib.dump(best_dt, "models/decision_tree.pkl")
print("  Saved: models/decision_tree.pkl")

print("""
INTERPRETATION: Decision Trees are great at capturing complex rules and
interactions (like "DHA plots > 1 Kanal are in a different price league").
However, they tend to OVERFIT — the tree memorizes training data perfectly
(R² train = 1.0 if unrestricted) but generalizes poorly. max_depth and
min_samples_leaf control this trade-off.
""")

# ==========================================================================
# MODEL 6: LOGISTIC REGRESSION (Classification)
# ==========================================================================
# IMPORTANT: Logistic Regression is a CLASSIFIER, not a regressor!
# It predicts CATEGORIES, not continuous values.
#
# WHY IS THIS NECESSARY?
# Logistic Regression cannot predict a number like "PKR 3.5 Crore".
# Instead, it predicts which CLASS a property belongs to. So we must
# convert our continuous price into categories (bins).
#
# We use pd.qcut() to create 3 equal-sized bins:
#   - Low: cheapest ~33% of properties
#   - Medium: middle ~33%
#   - High: most expensive ~33%
#
# This is called "discretization" or "binning".

print("\n" + "#" * 60)
print("  MODEL 6: LOGISTIC REGRESSION (Classification)")
print("#" * 60)

# Create price bins using quantiles (equal-frequency bins)
# pd.qcut splits data so each bin has roughly the same number of samples
price_bins = pd.qcut(y, q=3, labels=["Low", "Medium", "High"])

print("\nPrice bin ranges:")
bin_edges = pd.qcut(y, q=3, retbins=True)[1]
print(f"  Low:    PKR {bin_edges[0]:>15,.0f} to PKR {bin_edges[1]:>15,.0f}")
print(f"  Medium: PKR {bin_edges[1]:>15,.0f} to PKR {bin_edges[2]:>15,.0f}")
print(f"  High:   PKR {bin_edges[2]:>15,.0f} to PKR {bin_edges[3]:>15,.0f}")
print(f"\nBin distribution:")
print(price_bins.value_counts().sort_index())

# Split with the same indices as before (same random_state)
y_train_class = price_bins.iloc[X_train.index] if hasattr(X_train, 'index') else price_bins.loc[X_train.index]
y_test_class = price_bins.iloc[X_test.index] if hasattr(X_test, 'index') else price_bins.loc[X_test.index]

# Build logistic regression pipeline
log_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,           # enough iterations to converge
        solver="lbfgs",          # efficient solver
        random_state=42
    ))
])

log_pipeline.fit(X_train, y_train_class)

# Predictions
y_train_class_pred = log_pipeline.predict(X_train)
y_test_class_pred = log_pipeline.predict(X_test)

# Metrics for classification
train_acc = accuracy_score(y_train_class, y_train_class_pred)
test_acc = accuracy_score(y_test_class, y_test_class_pred)
test_f1 = f1_score(y_test_class, y_test_class_pred, average="macro")
conf_mat = confusion_matrix(y_test_class, y_test_class_pred, labels=["Low", "Medium", "High"])

print(f"\n{'='*50}")
print(f"  Logistic Regression (Classification)")
print(f"{'='*50}")
print(f"  Accuracy (Train): {train_acc:.4f}")
print(f"  Accuracy (Test):  {test_acc:.4f}")
print(f"  Macro F1 (Test):  {test_f1:.4f}")
print(f"\n  Confusion Matrix (rows=actual, columns=predicted):")
print(f"              {'Low':>8} {'Medium':>8} {'High':>8}")
for i, label in enumerate(["Low", "Medium", "High"]):
    print(f"  {label:>10}  {conf_mat[i][0]:>8} {conf_mat[i][1]:>8} {conf_mat[i][2]:>8}")

# Save the model
joblib.dump(log_pipeline, "models/logistic_regression.pkl")
print("\n  Saved: models/logistic_regression.pkl")

# Also save the bin edges so the web app knows the price ranges
joblib.dump(bin_edges, "models/price_bin_edges.pkl")
print("  Saved: models/price_bin_edges.pkl")

print("""
INTERPRETATION: Logistic Regression classifies properties into Low/Medium/High
price bands. The accuracy tells us what fraction it gets right. The confusion
matrix shows WHERE it makes mistakes — e.g., it might confuse Medium with High
more than Low with High, which makes sense since neighboring bins overlap.
Macro F1 weights all three classes equally, even if they have slightly
different sizes.
""")

# ==========================================================================
# 4.6  COMPARISON TABLE
# ==========================================================================
print("\n" + "=" * 60)
print("  MODEL COMPARISON — REGRESSION MODELS")
print("=" * 60)

# Build a comparison DataFrame
comparison = pd.DataFrame(results)[["model", "r2_train", "r2_test", "mae", "rmse"]]
comparison["mae_crore"] = comparison["mae"] / 1e7  # convert MAE to Crore for readability
comparison["rmse_crore"] = comparison["rmse"] / 1e7
comparison["overfit_gap"] = comparison["r2_train"] - comparison["r2_test"]

print(comparison[["model", "r2_train", "r2_test", "mae_crore", "rmse_crore", "overfit_gap"]].to_string(index=False))

# Find the best model (highest test R²)
best_model = comparison.loc[comparison["r2_test"].idxmax()]
print(f"\n🏆 BEST MODEL: {best_model['model']}")
print(f"   Test R² = {best_model['r2_test']:.4f}")
print(f"   MAE = PKR {best_model['mae']:,.0f} ({best_model['mae_crore']:.2f} Crore)")
print(f"   RMSE = PKR {best_model['rmse']:,.0f} ({best_model['rmse_crore']:.2f} Crore)")

print("\nLogistic Regression (Classification):")
print(f"   Test Accuracy = {test_acc:.4f}")
print(f"   Macro F1 = {test_f1:.4f}")

# ==========================================================================
# 4.7  Overfitting Analysis
# ==========================================================================
print("\n" + "=" * 60)
print("OVERFITTING ANALYSIS")
print("=" * 60)
print("""
Overfitting occurs when a model learns the training data TOO WELL — including
its noise and random fluctuations — and fails to generalize to new data.

We detect it by comparing Train R² vs Test R²:
  - If Train R² ≈ Test R² → Good generalization
  - If Train R² >> Test R² → Overfitting (model memorized training data)
  - If both R² are low → Underfitting (model is too simple)
""")

for _, row in comparison.iterrows():
    gap = row["overfit_gap"]
    status = "✅ Good" if gap < 0.05 else ("⚡ Mild overfitting" if gap < 0.1 else "⚠️ OVERFITTING")
    print(f"  {row['model']:<35} Gap: {gap:.4f}  {status}")

# ==========================================================================
# 4.8  Save results for the web app
# ==========================================================================
# Save all metrics so the Streamlit app can display them without retraining

# Save regression results
reg_results_clean = []
for r in results:
    reg_results_clean.append({
        "model": r["model"],
        "r2_train": r["r2_train"],
        "r2_test": r["r2_test"],
        "mae": r["mae"],
        "rmse": r["rmse"]
    })

all_results = {
    "regression": reg_results_clean,
    "logistic": {
        "accuracy_train": round(train_acc, 4),
        "accuracy_test": round(test_acc, 4),
        "f1_macro": round(test_f1, 4),
        "confusion_matrix": conf_mat.tolist(),
        "labels": ["Low", "Medium", "High"],
        "bin_edges": bin_edges.tolist()
    }
}

with open("models/results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: models/results.json")

# Save test predictions for actual-vs-predicted plots
predictions_df = pd.DataFrame({
    "actual": y_test.values,
})
for r in results:
    predictions_df[r["model"]] = r["y_test_pred"]

predictions_df.to_csv("models/test_predictions.csv", index=False)
print("Saved: models/test_predictions.csv")

# Save logistic regression test data
log_pred_df = pd.DataFrame({
    "actual": y_test_class.values,
    "predicted": y_test_class_pred
})
log_pred_df.to_csv("models/logistic_predictions.csv", index=False)
print("Saved: models/logistic_predictions.csv")

# Save the locality list for the web app dropdown
localities = sorted(df["locality"].unique().tolist())
with open("models/localities.json", "w") as f:
    json.dump(localities, f)
print("Saved: models/localities.json")

print("\n✅ Stage 4 complete! All models trained, evaluated, and saved.")
