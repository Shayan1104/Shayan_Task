# Zameen Property Valuation & Machine Learning Assignment

An end-to-end Machine Learning and Full-Stack Web Application project for predicting the price of **Residential Plots** in Lahore, Pakistan using Zameen.com property listings (`zameenData.csv`).

---

## Project Structure

```
zameen-project/
  data/
    zameenData.csv             # Raw Zameen property listings
    cleaned_plots.csv          # Cleaned residential plot listings (2,982 rows)
  src/
    01_explore.py              # Stage 1: Data exploration & anomaly detection
    02_clean.py                # Stage 2: Data cleaning, unit parsing & outlier removal
    03_eda.py                  # Stage 3: Visual EDA generating charts in eda_outputs/
    04_train.py                # Stage 4: Scikit-learn Pipelines, GridSearchCV, model export
  models/
    linear_regression.pkl      # Saved Pipeline (StandardScaler + OneHotEncoder + Model)
    polynomial_regression.pkl  # Saved Pipeline (PolynomialFeatures degree 2 + Model)
    knn.pkl                    # Tuned KNN Regressor Pipeline
    svm.pkl                    # Tuned Support Vector Regressor (LinearSVR) Pipeline
    decision_tree.pkl          # Tuned Decision Tree Regressor Pipeline
    logistic_regression.pkl    # Price Band Classifier Pipeline (Low / Medium / High)
    metrics.json               # Evaluation metrics, best hyperparameters, metadata
    test_predictions.csv       # Test set actuals vs predictions across models
  eda_outputs/                 # PNG charts saved from Stage 3
    01_price_distribution_raw_vs_log.png
    02_area_distribution.png
    03_price_vs_area_scatter.png
    04_median_price_top_localities.png
    05_price_boxplot_top_localities.png
    06_correlation_heatmap.png
  backend/
    main.py                    # FastAPI REST API serving endpoints for models & EDA
  frontend/                    # React.js (Vite) + Recharts + Tailwind CSS web dashboard
  requirements.txt             # Python dependencies
  README.md                    # Project documentation & run guide
```

---

## Setup & Execution Guide

### 1. Python Environment Setup
From the `zameen-project/` directory:
```bash
pip install -r requirements.txt
```

### 2. Run the Machine Learning Pipeline in Order

#### Stage 1: Understand the Data
```bash
python src/01_explore.py
```
*Prints dimensions, missing values, column shifts due to installment scraping, price/area text units, and non-Lahore cities.*

#### Stage 2: Clean the Data
```bash
python src/02_clean.py
```
*Repairs 95 shifted installment listings, filters for `Residential Plot` and `For Sale`, converts prices to PKR and areas to Marla (1 Kanal = 20 Marla), deduplicates listings, filters outliers with Tukey's IQR on log(Price/Marla), and groups rare localities (<10) into "Other". Saves `data/cleaned_plots.csv` (2,982 rows).*

#### Stage 3: Exploratory Data Analysis (EDA)
```bash
python src/03_eda.py
```
*Generates and saves 6 high-resolution charts in `eda_outputs/`, visualizing target right-skew, plot clustering (3, 5, 10, 20 Marla), locality variance, and numeric correlation.*

#### Stage 4: Model Training & Hyperparameter Tuning
```bash
python src/04_train.py
```
*Performs an 80/20 train/test split (`random_state=42`), builds encapsulated scikit-learn Pipelines, tunes KNN, SVR, and Decision Tree with `GridSearchCV`, bins prices into 3 balanced bands for Logistic Regression, and exports `.pkl` pipelines, `metrics.json`, and `test_predictions.csv`.*

---

### 3. Launch the FastAPI Backend
From `zameen-project/`:
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`

Available REST Endpoints:
- `GET  /api/metrics` — Model evaluation metrics and hyperparameters
- `GET  /api/predictions` — Test set actual vs predicted values (sampled $\le$ 500)
- `GET  /api/eda` — Summary stats, quartiles, price distribution bins
- `GET  /api/localities` — Valid locality list and allowed area range
- `POST /api/predict` — Real-time price predictions across all 6 models

---

### 4. Launch the React Web Application
From `zameen-project/frontend/`:
```bash
npm install
npm run dev
```
Open your browser at `http://127.0.0.1:5173/` to view the dashboard.

---

## Evaluated Algorithms & Final Results

| Rank | Algorithm | Model Type | Test $R^2$ | Train $R^2$ | Overfitting Gap | MAE (PKR) | RMSE (PKR) | Avg % Error (MAPE) | Key Hyperparameters |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **#1** | **SVM (SVR)** | Regressor | **59.39%** | 37.77% | -21.6% | PKR 1.07 Cr | **PKR 4.32 Cr** | 52.0% | `C: 2,000,000`, `kernel: linear` |
| **#2** | **Linear Regression** | Regressor | 56.77% | 48.26% | -8.5% | PKR 1.49 Cr | PKR 4.46 Cr | 133.1% | `fit_intercept: True` |
| **#3** | **Decision Tree** | Regressor | 47.92% | 70.33% | +22.4% | PKR 94.86 L | PKR 4.90 Cr | 40.2% | `min_samples_split: 10`, `leaf: 2` |
| **#4** | **KNN** | Regressor | 42.79% | 87.62% | +44.8% | **PKR 92.78 L** | PKR 5.13 Cr | **39.9%** | `n_neighbors: 15`, `weights: distance` |
| **#5** | **Polynomial Reg (d=2)** | Regressor | 33.40% | 60.96% | +27.6% | PKR 1.29 Cr | PKR 5.54 Cr | 74.2% | `degree: 2` (on numeric features) |
| **—** | **Logistic Regression** | Classifier | **80.57%** (Acc) | 78.36% (Acc) | -2.2% | — | — | **0.8096** (F1) | `C: 1.0`, `solver: lbfgs` |

*Note on Logistic Regression: Trained to predict 3 equal-sized market brackets: Low ($\le$ 70 Lakh), Medium (70 Lakh to 1.70 Crore), and High (> 1.70 Crore).*
