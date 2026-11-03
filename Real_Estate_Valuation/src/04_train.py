"""
04_train.py - Stage 4: Model Training, Hyperparameter Tuning & Evaluation
--------------------------------------------------------------------------------
Author: Student ML Project
Input: data/cleaned_plots.csv
Outputs:
  - models/*.pkl (scikit-learn pipelines with embedded preprocessing)
  - models/metrics.json (regression & classification metrics, params, metadata)
  - models/test_predictions.csv (ground truth vs predictions on test set)

Educational Purpose:
This script demonstrates standard industry machine learning workflows:
  1. Train/Test Split:
     Splitting data (80% train, 20% test, random_state=42) guarantees that model
     performance is evaluated on unseen test data, detecting overfitting.
  2. Full Scikit-Learn Pipelines:
     All preprocessing steps (StandardScaler for numerical features, OneHotEncoder
     for locality) are enclosed inside the Pipeline object. When saved via pickle/joblib,
     the pipeline handles raw inference inputs seamlessly without data leakage.
  3. Regression vs Classification:
     - Regression models (Linear, Polynomial, KNN, SVR, Decision Tree) predict
       the continuous target: 'price' in PKR.
     - Logistic Regression is inherently a classification algorithm (it outputs class
       probabilities via the sigmoid / softmax function). Therefore, we discretize
       the continuous price into 3 balanced bands: 'Low', 'Medium', 'High',
       and evaluate accuracy, macro F1, and the confusion matrix.
  4. Hyperparameter Optimization:
     GridSearchCV with 5-fold cross-validation is used to systematically discover
     the optimal hyperparameter configurations for KNN, SVR, and Decision Tree.
  5. Comprehensive Evaluation Metrics:
     - R2 Score (Train & Test): Measures variance explained ($1 - SS_{res}/SS_{tot}$).
       The gap $(R^2_{train} - R^2_{test})$ quantifies overfitting.
     - MAE (Mean Absolute Error): Average magnitude of errors in PKR.
     - RMSE (Root Mean Squared Error): Penalizes large catastrophic errors.
     - Average % Error (MAPE): Intuitive relative error percentage.
--------------------------------------------------------------------------------
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix
)


def calculate_regression_metrics(y_true, y_pred, y_train_true=None, y_train_pred=None) -> dict:
    """
    Computes all standard regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_test = r2_score(y_true, y_pred)
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

    metrics = {
        "r2_test": round(float(r2_test), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape": round(float(mape), 2),
    }

    if y_train_true is not None and y_train_pred is not None:
        r2_train = r2_score(y_train_true, y_train_pred)
        metrics["r2_train"] = round(float(r2_train), 4)
        metrics["overfitting_gap"] = round(float(r2_train - r2_test), 4)

    return metrics


def train_and_evaluate_all():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clean_csv_path = os.path.join(base_dir, 'data', 'cleaned_plots.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(" STAGE 4: MODEL TRAINING, HYPERPARAMETER TUNING & COMPARISON ")
    print("=" * 80)

    # 1. Load Data
    df = pd.read_csv(clean_csv_path)
    print(f"Loaded {len(df):,} cleaned plot listings.")

    # Feature definitions
    num_features = ['area', 'log_area']
    cat_features = ['locality']
    feature_cols = num_features + cat_features

    X = df[feature_cols]
    y = df['price']

    # 80/20 train/test split with deterministic seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Dataset split: Train set = {len(X_train):,} samples | Test set = {len(X_test):,} samples")

    # 2. Build Reusable Preprocessors
    # Standard Preprocessor: Standardizes continuous numeric features, One-Hot Encodes locality
    std_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # Polynomial Preprocessor: Generates degree-2 polynomial combinations of numeric features (area, log_area)
    poly_num_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler())
    ])
    poly_preprocessor = ColumnTransformer(
        transformers=[
            ('num', poly_num_pipeline, num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    all_metrics = {}
    saved_models = {}
    predictions_df = pd.DataFrame({
        'actual_price': y_test.values,
        'area': X_test['area'].values,
        'locality': X_test['locality'].values
    }, index=X_test.index)

    # --------------------------------------------------------------------------
    # MODEL 1: Linear Regression
    # --------------------------------------------------------------------------
    print("\n[1/6] Training Linear Regression...")
    linear_pipe = Pipeline([
        ('preprocessor', std_preprocessor),
        ('model', LinearRegression())
    ])
    linear_pipe.fit(X_train, y_train)
    y_test_pred_lin = linear_pipe.predict(X_test)
    y_train_pred_lin = linear_pipe.predict(X_train)

    lin_metrics = calculate_regression_metrics(y_test, y_test_pred_lin, y_train, y_train_pred_lin)
    all_metrics['Linear Regression'] = {
        'type': 'regression',
        'metrics': lin_metrics,
        'hyperparameters': {'fit_intercept': True}
    }
    saved_models['linear_regression'] = linear_pipe
    predictions_df['linear_regression_pred'] = y_test_pred_lin
    print(f"  Linear Regression -> Test R2: {lin_metrics['r2_test']:.4f} | Train R2: {lin_metrics['r2_train']:.4f} | MAE: PKR {lin_metrics['mae']:,.0f}")

    # --------------------------------------------------------------------------
    # MODEL 2: Polynomial Regression (Degree 2)
    # --------------------------------------------------------------------------
    print("\n[2/6] Training Polynomial Regression (Degree 2)...")
    poly_pipe = Pipeline([
        ('preprocessor', poly_preprocessor),
        ('model', LinearRegression())
    ])
    poly_pipe.fit(X_train, y_train)
    y_test_pred_poly = poly_pipe.predict(X_test)
    y_train_pred_poly = poly_pipe.predict(X_train)

    poly_metrics = calculate_regression_metrics(y_test, y_test_pred_poly, y_train, y_train_pred_poly)
    all_metrics['Polynomial Regression'] = {
        'type': 'regression',
        'metrics': poly_metrics,
        'hyperparameters': {'degree': 2, 'include_bias': False}
    }
    saved_models['polynomial_regression'] = poly_pipe
    predictions_df['polynomial_regression_pred'] = y_test_pred_poly
    print(f"  Polynomial Regression -> Test R2: {poly_metrics['r2_test']:.4f} | Train R2: {poly_metrics['r2_train']:.4f} | MAE: PKR {poly_metrics['mae']:,.0f}")

    # --------------------------------------------------------------------------
    # MODEL 3: K-Nearest Neighbors (KNN) with GridSearchCV
    # --------------------------------------------------------------------------
    print("\n[3/6] Tuning K-Nearest Neighbors (KNN) via GridSearchCV...")
    knn_pipe = Pipeline([
        ('preprocessor', std_preprocessor),
        ('model', KNeighborsRegressor())
    ])
    knn_grid = {
        'model__n_neighbors': [3, 5, 7, 9, 11, 15],
        'model__weights': ['uniform', 'distance']
    }
    knn_cv = GridSearchCV(knn_pipe, knn_grid, cv=5, scoring='r2', n_jobs=-1)
    knn_cv.fit(X_train, y_train)
    best_knn = knn_cv.best_estimator_

    y_test_pred_knn = best_knn.predict(X_test)
    y_train_pred_knn = best_knn.predict(X_train)
    knn_metrics = calculate_regression_metrics(y_test, y_test_pred_knn, y_train, y_train_pred_knn)
    
    clean_knn_params = {k.replace('model__', ''): v for k, v in knn_cv.best_params_.items()}
    all_metrics['KNN'] = {
        'type': 'regression',
        'metrics': knn_metrics,
        'hyperparameters': clean_knn_params
    }
    saved_models['knn'] = best_knn
    predictions_df['knn_pred'] = y_test_pred_knn
    print(f"  Best KNN Params: {clean_knn_params}")
    print(f"  KNN -> Test R2: {knn_metrics['r2_test']:.4f} | Train R2: {knn_metrics['r2_train']:.4f} | MAE: PKR {knn_metrics['mae']:,.0f}")

    # --------------------------------------------------------------------------
    # MODEL 4: Support Vector Regression (SVR) with GridSearchCV
    # --------------------------------------------------------------------------
    print("\n[4/6] Tuning Support Vector Regression (SVR) via GridSearchCV...")
    svr_pipe = Pipeline([
        ('preprocessor', std_preprocessor),
        ('model', SVR(kernel='linear'))
    ])
    svr_grid = {
        'model__C': [2e5, 5e5, 1e6, 2e6],
        'model__epsilon': [0.0, 0.1, 0.2]
    }
    svr_cv = GridSearchCV(svr_pipe, svr_grid, cv=3, scoring='r2', n_jobs=-1)
    svr_cv.fit(X_train, y_train)
    best_svr = svr_cv.best_estimator_

    y_test_pred_svr = best_svr.predict(X_test)
    y_train_pred_svr = best_svr.predict(X_train)
    svr_metrics = calculate_regression_metrics(y_test, y_test_pred_svr, y_train, y_train_pred_svr)
    
    clean_svr_params = {k.replace('model__', ''): v for k, v in svr_cv.best_params_.items()}
    clean_svr_params['kernel'] = 'linear'
    all_metrics['SVM'] = {
        'type': 'regression',
        'metrics': svr_metrics,
        'hyperparameters': clean_svr_params
    }
    saved_models['svm'] = best_svr
    predictions_df['svm_pred'] = y_test_pred_svr
    print(f"  Best SVR Params: {clean_svr_params}")
    print(f"  SVR -> Test R2: {svr_metrics['r2_test']:.4f} | Train R2: {svr_metrics['r2_train']:.4f} | MAE: PKR {svr_metrics['mae']:,.0f}")

    # --------------------------------------------------------------------------
    # MODEL 5: Decision Tree Regressor with GridSearchCV
    # --------------------------------------------------------------------------
    print("\n[5/6] Tuning Decision Tree Regressor via GridSearchCV...")
    dt_pipe = Pipeline([
        ('preprocessor', std_preprocessor),
        ('model', DecisionTreeRegressor(random_state=42))
    ])
    dt_grid = {
        'model__max_depth': [4, 6, 8, 10, 14, None],
        'model__min_samples_split': [5, 10, 20],
        'model__min_samples_leaf': [2, 5, 10]
    }
    dt_cv = GridSearchCV(dt_pipe, dt_grid, cv=5, scoring='r2', n_jobs=-1)
    dt_cv.fit(X_train, y_train)
    best_dt = dt_cv.best_estimator_

    y_test_pred_dt = best_dt.predict(X_test)
    y_train_pred_dt = best_dt.predict(X_train)
    dt_metrics = calculate_regression_metrics(y_test, y_test_pred_dt, y_train, y_train_pred_dt)

    clean_dt_params = {k.replace('model__', ''): v for k, v in dt_cv.best_params_.items()}
    all_metrics['Decision Tree'] = {
        'type': 'regression',
        'metrics': dt_metrics,
        'hyperparameters': clean_dt_params
    }
    saved_models['decision_tree'] = best_dt
    predictions_df['decision_tree_pred'] = y_test_pred_dt
    print(f"  Best Decision Tree Params: {clean_dt_params}")
    print(f"  Decision Tree -> Test R2: {dt_metrics['r2_test']:.4f} | Train R2: {dt_metrics['r2_train']:.4f} | MAE: PKR {dt_metrics['mae']:,.0f}")

    # --------------------------------------------------------------------------
    # MODEL 6: Logistic Regression (Classification by Price Band)
    # --------------------------------------------------------------------------
    print("\n[6/6] Training Logistic Regression (Price Band Classifier)...")
    print("  Educational Note: Why bin price into bands for Logistic Regression?")
    print("  Logistic regression is formulated to solve classification tasks by estimating")
    print("  categorical class probabilities via the logit function: P(Y=k|X) = softmax(w*X + b).")
    print("  It cannot predict continuous monetary values directly. To compare it meaningfully,")
    print("  we discretize the price distribution into 3 equal-sized tiers (Low, Medium, High)")
    print("  based on train set terciles, allowing Logistic Regression to predict the market bracket.")

    q33 = float(np.quantile(y_train, 1/3))
    q66 = float(np.quantile(y_train, 2/3))
    price_band_edges = {
        "low_max": q33,
        "medium_max": q66,
        "labels": ["Low", "Medium", "High"]
    }

    def categorize_price(p):
        if p <= q33:
            return "Low"
        elif p <= q66:
            return "Medium"
        else:
            return "High"

    y_train_class = y_train.apply(categorize_price)
    y_test_class = y_test.apply(categorize_price)

    clf_pipe = Pipeline([
        ('preprocessor', std_preprocessor),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ])
    clf_pipe.fit(X_train, y_train_class)

    y_test_pred_class = clf_pipe.predict(X_test)
    y_train_pred_class = clf_pipe.predict(X_train)

    acc_test = float(accuracy_score(y_test_class, y_test_pred_class))
    acc_train = float(accuracy_score(y_train_class, y_train_pred_class))
    macro_f1 = float(f1_score(y_test_class, y_test_pred_class, average='macro'))
    
    labels_order = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test_class, y_test_pred_class, labels=labels_order).tolist()

    all_metrics['Logistic Regression'] = {
        'type': 'classification',
        'metrics': {
            'accuracy_test': round(acc_test, 4),
            'accuracy_train': round(acc_train, 4),
            'macro_f1': round(macro_f1, 4),
            'confusion_matrix': cm,
            'labels': labels_order
        },
        'hyperparameters': {'max_iter': 1000, 'solver': 'lbfgs', 'multi_class': 'multinomial'}
    }
    saved_models['logistic_regression'] = clf_pipe
    predictions_df['actual_price_band'] = y_test_class.values
    predictions_df['logistic_regression_pred'] = y_test_pred_class
    print(f"  Logistic Regression -> Accuracy: {acc_test:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"  Confusion Matrix (Low, Medium, High):\n  {cm}")

    # 3. Save Model Artifacts
    print("\nSaving pipeline pickle files to models/ ...")
    for name, model_obj in saved_models.items():
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model_obj, model_path)
        print(f"  [SAVED] {model_path}")

    # Locality list and area range metadata for frontend dropdowns & validation
    distinct_localities = sorted(df['locality'].unique().tolist())
    area_min = float(df['area'].min())
    area_max = float(df['area'].max())

    metadata = {
        "metrics": all_metrics,
        "price_band_edges": price_band_edges,
        "localities": distinct_localities,
        "area_range": {
            "min_area_marla": area_min,
            "max_area_marla": area_max
        },
        "dataset_summary": {
            "total_listings": len(df),
            "train_listings": len(X_train),
            "test_listings": len(X_test),
            "median_price_pkr": float(df['price'].median()),
            "mean_price_pkr": float(df['price'].mean()),
            "iqr_lower_ppm": float(np.exp(12.258)),
            "iqr_upper_ppm": float(np.exp(15.703))
        }
    }

    metrics_json_path = os.path.join(models_dir, 'metrics.json')
    with open(metrics_json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  [SAVED] {metrics_json_path}")

    test_preds_csv_path = os.path.join(models_dir, 'test_predictions.csv')
    predictions_df.to_csv(test_preds_csv_path, index=False)
    print(f"  [SAVED] {test_preds_csv_path}")

    print("\n" + "=" * 80)
    print(" [STAGE 4 COMPLETE] All 6 Models Trained, Evaluated and Exported Successfully!")
    print("=" * 80 + "\n")


def main():
    train_and_evaluate_all()


if __name__ == '__main__':
    main()
