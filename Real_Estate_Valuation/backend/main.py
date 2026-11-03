"""
backend/main.py - FastAPI Application
--------------------------------------------------------------------------------
Author: Student ML Project
Endpoints:
  - GET  /api/metrics       : Model metrics, hyperparameters, dataset summary
  - GET  /api/predictions   : Actual vs predicted test-set records (sampled <= 500)
  - GET  /api/eda           : Summary statistics, quartiles, locality prices, histogram bins
  - GET  /api/localities    : Distinct locality choices & valid area range
  - POST /api/predict       : Predict price across all models for user inputs
--------------------------------------------------------------------------------
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Zameen Property Price Prediction API",
    description="Machine Learning REST API for Residential Plot Price Valuation in Pakistan",
    version="1.0.0"
)

# Enable CORS for React Vite frontend (running on localhost:5173 or any port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# In-memory storage for loaded models and metadata
MODELS = {}
METRICS_DATA = {}
PREDICTIONS_CACHE = []
EDA_CACHE = {}


def format_pkr(amount: float) -> str:
    """
    Formats a raw numerical PKR amount into conventional Pakistani denominations:
    Crore (10,000,000 PKR), Lakh (100,000 PKR), or Thousand.
    """
    if amount >= 1e7:
        val = amount / 1e7
        return f"PKR {val:.2f} Crore"
    elif amount >= 1e5:
        val = amount / 1e5
        return f"PKR {val:.2f} Lakh"
    elif amount >= 1e3:
        val = amount / 1e3
        return f"PKR {val:.2f} Thousand"
    else:
        return f"PKR {amount:,.0f}"


@app.on_event("startup")
def load_artifacts():
    """
    Loads saved models, metrics.json, predictions, and computes EDA summaries at startup.
    """
    global MODELS, METRICS_DATA, PREDICTIONS_CACHE, EDA_CACHE

    print("[STARTUP] Loading model pipelines from models/ ...")
    model_files = {
        'linear_regression': 'linear_regression.pkl',
        'polynomial_regression': 'polynomial_regression.pkl',
        'knn': 'knn.pkl',
        'svm': 'svm.pkl',
        'decision_tree': 'decision_tree.pkl',
        'logistic_regression': 'logistic_regression.pkl',
    }

    for key, filename in model_files.items():
        file_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(file_path):
            MODELS[key] = joblib.load(file_path)
            print(f"  Loaded model pipeline: {key}")
        else:
            print(f"  [WARNING] Model file {file_path} not found. Ensure 04_train.py has run.")

    # Load metrics.json
    metrics_path = os.path.join(MODELS_DIR, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            METRICS_DATA = json.load(f)
            print("  Loaded metrics.json")

    # Load and sample test predictions
    preds_csv_path = os.path.join(MODELS_DIR, 'test_predictions.csv')
    if os.path.exists(preds_csv_path):
        preds_df = pd.read_csv(preds_csv_path)
        # Sample <= 500 points deterministically
        sample_size = min(500, len(preds_df))
        sampled_df = preds_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        PREDICTIONS_CACHE = sampled_df.to_dict(orient='records')
        print(f"  Loaded and cached {len(PREDICTIONS_CACHE)} sampled test predictions.")

    # Compute EDA summary stats from cleaned_plots.csv
    clean_csv_path = os.path.join(DATA_DIR, 'cleaned_plots.csv')
    if os.path.exists(clean_csv_path):
        df_clean = pd.read_csv(clean_csv_path)
        
        # Price Quartiles
        p_q1 = float(df_clean['price'].quantile(0.25))
        p_med = float(df_clean['price'].median())
        p_q3 = float(df_clean['price'].quantile(0.75))
        p_min = float(df_clean['price'].min())
        p_max = float(df_clean['price'].max())

        # Area Quartiles
        a_q1 = float(df_clean['area'].quantile(0.25))
        a_med = float(df_clean['area'].median())
        a_q3 = float(df_clean['area'].quantile(0.75))
        a_min = float(df_clean['area'].min())
        a_max = float(df_clean['area'].max())

        # Median price by top 15 localities
        top_localities = df_clean[df_clean['locality'] != 'Other']['locality'].value_counts().head(15).index
        loc_medians = df_clean[df_clean['locality'].isin(top_localities)].groupby('locality')['price'].median().sort_values(ascending=False)
        median_by_locality = [
            {
                "locality": loc,
                "median_price_pkr": float(price),
                "median_price_crore": round(float(price) / 1e7, 3),
                "formatted": format_pkr(price),
                "count": int((df_clean['locality'] == loc).sum())
            }
            for loc, price in loc_medians.items()
        ]

        # Histogram Bins for Price (in Crores)
        counts_price, bin_edges_price = np.histogram(df_clean['price'] / 1e7, bins=25)
        price_histogram = [
            {
                "bin_start": round(float(bin_edges_price[i]), 2),
                "bin_end": round(float(bin_edges_price[i+1]), 2),
                "bin_label": f"{bin_edges_price[i]:.1f}-{bin_edges_price[i+1]:.1f} Cr",
                "count": int(counts_price[i])
            }
            for i in range(len(counts_price))
        ]

        # Histogram Bins for Area (in Marla, <= 40)
        area_sub = df_clean[df_clean['area'] <= 40]['area']
        counts_area, bin_edges_area = np.histogram(area_sub, bins=20)
        area_histogram = [
            {
                "bin_start": round(float(bin_edges_area[i]), 1),
                "bin_end": round(float(bin_edges_area[i+1]), 1),
                "bin_label": f"{bin_edges_area[i]:.0f}-{bin_edges_area[i+1]:.0f} M",
                "count": int(counts_area[i])
            }
            for i in range(len(counts_area))
        ]

        # Best performing regression model
        reg_metrics = {k: v['metrics']['r2_test'] for k, v in METRICS_DATA.get('metrics', {}).items() if v.get('type') == 'regression'}
        best_model_name = max(reg_metrics, key=reg_metrics.get) if reg_metrics else "SVM"

        EDA_CACHE = {
            "total_listings": len(df_clean),
            "price_summary": {
                "min": p_min,
                "q1": p_q1,
                "median": p_med,
                "q3": p_q3,
                "max": p_max,
                "formatted_median": format_pkr(p_med),
                "formatted_q1": format_pkr(p_q1),
                "formatted_q3": format_pkr(p_q3)
            },
            "area_summary": {
                "min": a_min,
                "q1": a_q1,
                "median": a_med,
                "q3": a_q3,
                "max": a_max
            },
            "best_model": {
                "name": best_model_name,
                "test_r2": reg_metrics.get(best_model_name, 0.5939)
            },
            "median_by_locality": median_by_locality,
            "price_histogram": price_histogram,
            "area_histogram": area_histogram
        }
        print("  Computed and cached EDA summary statistics.")


@app.get("/api/metrics")
def get_metrics():
    """
    Returns all model performance metrics, hyperparameter grids, and price bands.
    """
    if not METRICS_DATA:
        raise HTTPException(status_code=500, detail="Metrics metadata not loaded.")
    return METRICS_DATA


@app.get("/api/predictions")
def get_predictions():
    """
    Returns test-set actual vs predicted values sampled to <= 500 records.
    """
    return {
        "count": len(PREDICTIONS_CACHE),
        "predictions": PREDICTIONS_CACHE
    }


@app.get("/api/eda")
def get_eda_summary():
    """
    Returns high-level statistics, quartiles, price distribution bins, and locality medians.
    """
    if not EDA_CACHE:
        raise HTTPException(status_code=500, detail="EDA cache not available.")
    return EDA_CACHE


@app.get("/api/localities")
def get_localities():
    """
    Returns the list of valid localities and the allowable area range.
    """
    localities = METRICS_DATA.get("localities", [])
    area_range = METRICS_DATA.get("area_range", {"min_area_marla": 1.0, "max_area_marla": 100.0})
    return {
        "localities": localities,
        "area_range": area_range,
        "supported_units": ["Marla", "Kanal"]
    }


class PredictRequest(BaseModel):
    locality: str = Field(..., description="Locality name, e.g. 'DHA Defence' or 'Bahria Town'")
    area: float = Field(..., gt=0, description="Plot area value, strictly positive numeric")
    unit: Optional[str] = Field("Marla", description="Unit of measurement: 'Marla' or 'Kanal'")


@app.post("/api/predict")
def predict_price(req: PredictRequest):
    """
    Predicts plot price across all 5 regression models and estimates price band with Logistic Regression.
    """
    if not MODELS:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    unit_clean = req.unit.strip().lower()
    if unit_clean not in ["marla", "kanal"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid area unit '{req.unit}'. Must be either 'Marla' or 'Kanal'."
        )

    # Standardize area to Marla (1 Kanal = 20 Marla)
    area_marla = float(req.area * 20.0) if unit_clean == "kanal" else float(req.area)

    if area_marla <= 0:
        raise HTTPException(status_code=400, detail="Area must be strictly greater than 0.")
    if area_marla > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Area of {area_marla} Marla exceeds reasonable residential plot scope (max 500 Marla)."
        )

    # Validate locality against known localities
    known_localities = METRICS_DATA.get("localities", [])
    locality_input = req.locality.strip()
    if locality_input not in known_localities:
        # Graceful fallback to 'Other' if unrecognized
        locality_used = "Other"
    else:
        locality_used = locality_input

    # Prepare single-row DataFrame matching the training schema
    log_area = float(np.log(area_marla))
    input_df = pd.DataFrame([{
        'area': area_marla,
        'log_area': log_area,
        'locality': locality_used
    }])

    regression_predictions = {}
    reg_model_keys = {
        'Linear Regression': 'linear_regression',
        'Polynomial Regression': 'polynomial_regression',
        'KNN': 'knn',
        'SVM': 'svm',
        'Decision Tree': 'decision_tree'
    }

    for display_name, key in reg_model_keys.items():
        if key in MODELS:
            try:
                raw_pred = float(MODELS[key].predict(input_df)[0])
                # Plot prices cannot physically be negative
                pred_val = max(0.0, raw_pred)
                regression_predictions[display_name] = {
                    "price_pkr": round(pred_val, 2),
                    "price_crore": round(pred_val / 1e7, 3),
                    "price_lakh": round(pred_val / 1e5, 2),
                    "formatted": format_pkr(pred_val)
                }
            except Exception as e:
                regression_predictions[display_name] = {
                    "error": str(e)
                }

    # Logistic Regression Price Band Classification
    classification_result = {}
    if 'logistic_regression' in MODELS:
        try:
            band_pred = str(MODELS['logistic_regression'].predict(input_df)[0])
            probs = MODELS['logistic_regression'].predict_proba(input_df)[0]
            classes = MODELS['logistic_regression'].named_steps['model'].classes_.tolist()
            prob_dict = {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))}
            
            classification_result = {
                "predicted_band": band_pred,
                "probabilities": prob_dict,
                "band_definitions": {
                    "Low": "<= PKR 70 Lakh (7M PKR)",
                    "Medium": "PKR 70 Lakh - PKR 1.70 Crore",
                    "High": "> PKR 1.70 Crore"
                }
            }
        except Exception as e:
            classification_result = {"error": str(e)}

    return {
        "input": {
            "locality": locality_input,
            "locality_mapped": locality_used,
            "original_area": req.area,
            "unit": req.unit,
            "area_marla": area_marla,
            "log_area": round(log_area, 4)
        },
        "regression_predictions": regression_predictions,
        "classification": classification_result
    }


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
