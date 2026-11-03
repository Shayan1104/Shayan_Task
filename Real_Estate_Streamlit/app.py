"""
=============================================================================
STREAMLIT WEB APP — Zameen.com Property Price Valuation Platform
=============================================================================
An executive real estate machine learning dashboard featuring 6 pre-trained
models, dual Light/Dark adaptive themes, and interactive property valuation.

NO RETRAINING happens here — all pre-trained .pkl models and metrics are loaded.
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Zameen Intel™ — Plot Price Valuation Engine",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# LOAD CACHED MODELS AND DATA
# --------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load all trained model pipelines from .pkl files."""
    models = {}
    model_files = {
        "Linear Regression": "models/linear_regression.pkl",
        "Polynomial Regression (d=2)": "models/polynomial_regression.pkl",
        "KNN": "models/knn.pkl",
        "SVR": "models/svr.pkl",
        "Decision Tree": "models/decision_tree.pkl",
        "Logistic Regression": "models/logistic_regression.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_data
def load_results():
    """Load pre-computed metrics from results.json."""
    with open("models/results.json", "r") as f:
        return json.load(f)

@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv("clean_data.csv")

@st.cache_data
def load_predictions():
    """Load test set predictions."""
    return pd.read_csv("models/test_predictions.csv")

@st.cache_data
def load_logistic_predictions():
    """Load logistic regression predictions."""
    return pd.read_csv("models/logistic_predictions.csv")

@st.cache_data
def load_localities():
    """Load the list of unique localities."""
    with open("models/localities.json", "r") as f:
        return json.load(f)

# Load data assets
models = load_models()
results = load_results()
df = load_data()
predictions = load_predictions()
log_preds = load_logistic_predictions()
localities = load_localities()

# --------------------------------------------------------------------------
# SIDEBAR CONTROLS & THEME TOGGLE
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, #4f46e5, #7c3aed); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">🏢</div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em;">ZAMEEN INTEL™</h3>
                <span style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.7;">Analytics & Valuation</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # THEME SELECTOR: Resolves light mode issue with full theme adaptability
    theme_choice = st.radio(
        "🎨 Interface Theme",
        ["☀️ Professional Light", "🌙 Executive Dark"],
        index=0,
        horizontal=True
    )
    is_dark = "Dark" in theme_choice

    st.markdown("---")
    st.markdown("### 📊 Dataset Telemetry")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.caption("Active Listings")
        st.write(f"**{len(df):,}**")
    with col_sb2:
        st.caption("Localities")
        st.write(f"**{len(localities)}**")

    st.caption("Target Asset")
    st.write("**Residential Plots (For Sale)**")

    st.markdown("---")
    st.markdown("### 🤖 Trained Algorithms")
    st.markdown("""
    - **Linear Regression** *(Top Generalizer)*
    - **Polynomial Regression** *(Degree 2)*
    - **Decision Tree** *(Lowest MAE)*
    - **K-Nearest Neighbors** *(k=20)*
    - **Support Vector Regressor** *(RBF)*
    - **Logistic Regression** *(Tier Classifier)*
    """)
    st.markdown("---")
    st.caption("NIAI Task • Machine Learning Assignment • v2.0")

# --------------------------------------------------------------------------
# ADAPTIVE CSS THEME INJECTION
# --------------------------------------------------------------------------
if is_dark:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
        }
        section[data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid #1f2937;
        }
        div[data-testid="stMetric"] {
            background: #151c2e;
            border: 1px solid #23304c;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }
        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #f8fafc !important;
            font-weight: 800 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: #111827;
            padding: 6px;
            border-radius: 10px;
            border: 1px solid #1f2937;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            color: #94a3b8;
            font-weight: 600;
            border: none;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #4f46e5 !important;
            color: #ffffff !important;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.4);
        }
        .executive-card {
            background: #151c2e;
            border: 1px solid #23304c;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        }
        .highlight-badge {
            background: rgba(79, 70, 229, 0.2);
            color: #a5b4fc;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            display: inline-block;
        }
        .stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
            letter-spacing: 0.02em;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.5);
            transform: translateY(-1px);
        }
    </style>
    """
else:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }
        .stApp {
            background-color: #f8fafc;
            color: #0f172a;
        }
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e2e8f0;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetric"] label {
            color: #64748b !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #0f172a !important;
            font-weight: 800 !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: #f1f5f9;
            padding: 6px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 18px;
            color: #64748b;
            font-weight: 600;
            border: none;
            background: transparent;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff !important;
            color: #4f46e5 !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08);
        }
        .executive-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }
        .highlight-badge {
            background: #eef2ff;
            color: #4f46e5;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            display: inline-block;
        }
        .stButton > button {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 700;
            letter-spacing: 0.02em;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #4338ca 0%, #4f46e5 100%);
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35);
            transform: translateY(-1px);
        }
        h1, h2, h3, h4 {
            color: #0f172a !important;
        }
        p, li {
            color: #334155;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# MATPLOTLIB THEME HELPER (Dynamic Light / Dark Adaptation)
# --------------------------------------------------------------------------
def apply_chart_theme(fig, ax, is_dark_mode):
    """Dynamically applies high-contrast colors matching the selected theme."""
    bg_color = "#151c2e" if is_dark_mode else "#ffffff"
    text_color = "#f8fafc" if is_dark_mode else "#0f172a"
    sub_color = "#94a3b8" if is_dark_mode else "#64748b"
    grid_color = "#23304c" if is_dark_mode else "#f1f5f9"
    spine_color = "#334155" if is_dark_mode else "#cbd5e1"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.tick_params(colors=sub_color, labelsize=9)
    ax.xaxis.label.set_color(text_color)
    ax.xaxis.label.set_fontweight("600")
    ax.yaxis.label.set_color(text_color)
    ax.yaxis.label.set_fontweight("600")
    ax.title.set_color(text_color)
    ax.title.set_fontweight("700")
    ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(1.0)

# Helper function to format PKR numbers nicely
def format_pkr(price):
    if price >= 1e9:
        return f"₨ {price/1e9:.2f} Arab"
    elif price >= 1e7:
        return f"₨ {price/1e7:.2f} Crore"
    elif price >= 1e5:
        return f"₨ {price/1e5:.2f} Lakh"
    else:
        return f"₨ {price:,.0f}"

# --------------------------------------------------------------------------
# MAIN EXECUTIVE HEADER
# --------------------------------------------------------------------------
st.markdown("""
    <div style="margin-bottom: 24px;">
        <span class="highlight-badge">PAKISTAN PROPERTY INTELLIGENCE • ML ENGINE</span>
        <h1 style="margin: 8px 0 6px 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em;">
            Zameen Residential Plot Valuation Engine
        </h1>
        <p style="margin: 0; font-size: 1.05rem; opacity: 0.85;">
            Predicting real estate market values across 60 localities using 6 machine learning algorithms with automated feature preprocessing pipelines.
        </p>
    </div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# NAVIGATION TABS
# --------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Intelligence",
    "🏆 Model Benchmarking",
    "🎯 Predictive Precision",
    "📋 Tier Classification (Logistic)",
    "🔮 Instant Valuation Calculator"
])

# ==========================================================================
# TAB 1: MARKET INTELLIGENCE (DATA OVERVIEW)
# ==========================================================================
with tab1:
    st.subheader("Market Intelligence & Cleaned Data Summary")
    st.caption("Analysis of 2,949 residential plot transactions filtered and standardized from 20,571 raw listings.")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Verified Listings", f"{len(df):,}", help="Clean residential plot transactions")
    with kpi2:
        st.metric("Median Plot Price", format_pkr(df["price_pkr"].median()), help="Median price across all plots")
    with kpi3:
        st.metric("Average Plot Size", f"{df['area_marla'].mean():.1f} Marla", help="Average area in Marla units")
    with kpi4:
        st.metric("Localities Modeled", f"{df['locality'].nunique()}", help="Categorical areas encoded")

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("#### 📑 Sample Cleaned Records")
        display_sample = df.head(15)[["locality", "city", "area_marla", "price_pkr", "price_per_marla"]].copy()
        display_sample["price_pkr"] = display_sample["price_pkr"].apply(format_pkr)
        display_sample["price_per_marla"] = display_sample["price_per_marla"].apply(format_pkr)
        display_sample.columns = ["Locality", "City", "Area (Marla)", "Total Price", "Rate / Marla"]
        st.dataframe(display_sample, use_container_width=True, height=360)

    with c_right:
        st.markdown("#### 📍 Top Localities by Transaction Volume")
        loc_df = df["locality"].value_counts().head(10).reset_index()
        loc_df.columns = ["Locality", "Listings"]
        st.dataframe(loc_df, use_container_width=True, height=360)

    st.markdown("---")
    st.markdown("#### 📈 Key Exploratory Data Visualizations")

    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        fig1, ax1 = plt.subplots(figsize=(7, 4.2), dpi=130)
        ax1.hist(df["log_price"], bins=40, color="#4f46e5", edgecolor="white" if not is_dark else "#151c2e", alpha=0.9)
        ax1.set_xlabel("log₁₀(Price in PKR)")
        ax1.set_ylabel("Listing Count")
        ax1.set_title("Price Distribution (Normalized Log₁₀ Scale)")
        apply_chart_theme(fig1, ax1, is_dark)
        fig1.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.caption("💡 **Interpretation:** On the logarithmic scale, prices exhibit a near-Gaussian distribution centered at ~PKR 1.08 Crore, stabilizing error variance during linear model fitting.")

    with eda_col2:
        fig2, ax2 = plt.subplots(figsize=(7, 4.2), dpi=130)
        ax2.scatter(df["area_marla"], df["price_pkr"] / 1e7, alpha=0.4, s=16, c="#0284c7", edgecolors="none")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("Plot Area (Marla) — Log Scale")
        ax2.set_ylabel("Price (Crore PKR) — Log Scale")
        ax2.set_title("Price vs Area Power-Law Scaling")
        apply_chart_theme(fig2, ax2, is_dark)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.caption("💡 **Interpretation:** A strong log-linear relationship confirms that price scales with area by a power-law exponent. The vertical dispersion reflects locality premiums.")

# ==========================================================================
# TAB 2: MODEL BENCHMARKING
# ==========================================================================
with tab2:
    st.subheader("Model Benchmarking & Generalization Analysis")
    st.markdown("Evaluation of all 5 continuous regression models on the 20% holdout test set (590 unseen listings).")

    reg_results = results["regression"]
    bench_df = pd.DataFrame(reg_results)
    bench_df["mae_cr"] = bench_df["mae"] / 1e7
    bench_df["rmse_cr"] = bench_df["rmse"] / 1e7
    bench_df["overfit_gap"] = bench_df["r2_train"] - bench_df["r2_test"]

    display_bench = bench_df[["model", "r2_train", "r2_test", "mae_cr", "rmse_cr", "overfit_gap"]].copy()
    display_bench.columns = ["Algorithm", "Train R²", "Test R²", "MAE (Crore ₨)", "RMSE (Crore ₨)", "Train-Test Gap"]

    st.dataframe(
        display_bench.style.highlight_max(subset=["Test R²"], color="#bbf7d0" if not is_dark else "#065f46")
                           .highlight_min(subset=["MAE (Crore ₨)", "RMSE (Crore ₨)"], color="#bbf7d0" if not is_dark else "#065f46")
                           .format({
                               "Train R²": "{:.4f}",
                               "Test R²": "{:.4f}",
                               "MAE (Crore ₨)": "₨ {:.2f} Cr",
                               "RMSE (Crore ₨)": "₨ {:.2f} Cr",
                               "Train-Test Gap": "{:+.4f}"
                           }),
        use_container_width=True
    )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        fig_r2, ax_r2 = plt.subplots(figsize=(7, 4.4), dpi=130)
        x_idx = np.arange(len(bench_df))
        w = 0.35
        ax_r2.bar(x_idx - w/2, bench_df["r2_train"], w, label="Train R²", color="#4f46e5", alpha=0.9)
        ax_r2.bar(x_idx + w/2, bench_df["r2_test"], w, label="Test R²", color="#10b981", alpha=0.9)
        ax_r2.set_xticks(x_idx)
        ax_r2.set_xticklabels([m.split("(")[0].strip() for m in bench_df["model"]], rotation=25, ha="right", fontsize=9)
        ax_r2.set_ylabel("R² Coefficient of Determination")
        ax_r2.set_title("Generalization Gap: Train R² vs Test R²")
        apply_chart_theme(fig_r2, ax_r2, is_dark)
        ax_r2.legend(facecolor="#151c2e" if is_dark else "#ffffff",
                     edgecolor="#334155" if is_dark else "#cbd5e1",
                     labelcolor="#f8fafc" if is_dark else "#0f172a")
        fig_r2.tight_layout()
        st.pyplot(fig_r2)
        plt.close()

    with b_col2:
        fig_err, ax_err = plt.subplots(figsize=(7, 4.4), dpi=130)
        ax_err.bar(x_idx - w/2, bench_df["mae_cr"], w, label="MAE (Crore)", color="#0ea5e9", alpha=0.9)
        ax_err.bar(x_idx + w/2, bench_df["rmse_cr"], w, label="RMSE (Crore)", color="#f59e0b", alpha=0.9)
        ax_err.set_xticks(x_idx)
        ax_err.set_xticklabels([m.split("(")[0].strip() for m in bench_df["model"]], rotation=25, ha="right", fontsize=9)
        ax_err.set_ylabel("Error in Crore PKR")
        ax_err.set_title("Test Prediction Error Comparison (MAE vs RMSE)")
        apply_chart_theme(fig_err, ax_err, is_dark)
        ax_err.legend(facecolor="#151c2e" if is_dark else "#ffffff",
                      edgecolor="#334155" if is_dark else "#cbd5e1",
                      labelcolor="#f8fafc" if is_dark else "#0f172a")
        fig_err.tight_layout()
        st.pyplot(fig_err)
        plt.close()

    st.markdown("---")
    st.markdown("#### 🧠 Diagnostic Engineering Notes")
    st.info(r"""
    - **Linear Regression (Recommended Winner):** Achieved the highest holdout **Test R² = 0.5750** with zero overfitting gap. Pre-transforming area to $\log_{10}(\text{area})$ successfully linearized the price curve.
    - **Decision Tree:** Yielded the lowest **MAE (₨ 0.89 Cr)** by carving clean locality-specific decision boundaries, but shows moderate overfitting ($\Delta R^2 = 0.16$).
    - **Polynomial Regression & KNN:** Exhibited significant overfitting ($\Delta R^2 > 0.18$), memorizing training noise due to the 60 one-hot encoded locality dimensions.
    - **SVR:** Suffered in high dimensions; the radial basis function could not effectively span sparse one-hot encoded locality spaces.
    """)

# ==========================================================================
# TAB 3: PREDICTIVE PRECISION (ACTUAL VS PREDICTED)
# ==========================================================================
with tab3:
    st.subheader("Holdout Validation: Actual vs Predicted Prices")
    st.caption("Points aligned with the red dashed diagonal line represent 100% accurate price predictions.")

    reg_names = [r["model"] for r in reg_results]
    selected_reg = st.selectbox("Select Model to Inspect:", reg_names, index=0)

    if selected_reg in predictions.columns:
        p_col1, p_col2 = st.columns([3, 2])
        actual_cr = predictions["actual"] / 1e7
        pred_cr = predictions[selected_reg] / 1e7
        residuals = actual_cr - pred_cr

        with p_col1:
            fig_avp, ax_avp = plt.subplots(figsize=(7, 5), dpi=130)
            ax_avp.scatter(actual_cr, pred_cr, alpha=0.35, s=20, c="#4f46e5", edgecolors="none")
            min_lim = min(actual_cr.min(), pred_cr.min())
            max_lim = max(actual_cr.max(), pred_cr.max())
            ax_avp.plot([min_lim, max_lim], [min_lim, max_lim], "r--", linewidth=1.8, label="Ideal Prediction (y = x)")
            ax_avp.set_xlabel("Actual Price (Crore PKR)")
            ax_avp.set_ylabel("Predicted Price (Crore PKR)")
            ax_avp.set_title(f"{selected_reg}: Predicted vs Actual")
            apply_chart_theme(fig_avp, ax_avp, is_dark)
            ax_avp.legend(facecolor="#151c2e" if is_dark else "#ffffff",
                          edgecolor="#334155" if is_dark else "#cbd5e1",
                          labelcolor="#f8fafc" if is_dark else "#0f172a")
            fig_avp.tight_layout()
            st.pyplot(fig_avp)
            plt.close()

        with p_col2:
            st.markdown("#### 📐 Residual Metrics")
            st.metric("Mean Absolute Error", f"₨ {bench_df.loc[bench_df['model']==selected_reg, 'mae_cr'].values[0]:.2f} Crore")
            st.metric("Root Mean Squared Error", f"₨ {bench_df.loc[bench_df['model']==selected_reg, 'rmse_cr'].values[0]:.2f} Crore")
            st.metric("Mean Residual Bias", f"₨ {residuals.mean():+.2f} Crore")

            st.markdown("""
            **Interpretation Guide:**
            - **Tight clustering along the diagonal line** means the model captures market prices reliably.
            - Outliers high above the line represent under-valued predictions, while points below represent over-predictions.
            """)

# ==========================================================================
# TAB 4: TIER CLASSIFICATION (LOGISTIC REGRESSION)
# ==========================================================================
with tab4:
    st.subheader("Price Tier Classification (Logistic Regression)")
    st.markdown("""
    **Methodology:** Logistic Regression is a *classifier*. To apply it to property valuation, listings were segmented into 3 balanced price tiers (**Low**, **Medium**, **High**) using empirical quantiles.
    """)

    log_data = results["logistic"]
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.metric("Training Accuracy", f"{log_data['accuracy_train']:.2%}")
    with tc2:
        st.metric("Test Set Accuracy", f"{log_data['accuracy_test']:.2%}")
    with tc3:
        st.metric("Macro F1 Score", f"{log_data['f1_macro']:.2%}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    c_tier_l, c_tier_r = st.columns([1, 1])

    with c_tier_l:
        st.markdown("#### 🎯 Defined Price Segments")
        edges = log_data["bin_edges"]
        st.markdown(f"""
        | Segment Tier | Price Range (PKR) | Value in Standard Denominations |
        |:---|:---|:---|
        | 🟢 **Low Tier** | ₨ {edges[0]:,.0f} — ₨ {edges[1]:,.0f} | ₨ 6.8 Lakh to ₨ 71.2 Lakh |
        | 🟡 **Medium Tier** | ₨ {edges[1]:,.0f} — ₨ {edges[2]:,.0f} | ₨ 71.2 Lakh to ₨ 1.70 Crore |
        | 🔴 **High Tier** | ₨ {edges[2]:,.0f} — ₨ {edges[3]:,.0f} | ₨ 1.70 Crore to ₨ 146 Crore |
        """)
        st.caption("Each tier contains roughly 33.3% of all listings, guaranteeing no class imbalance.")

    with c_tier_r:
        st.markdown("#### 📊 Classification Confusion Matrix")
        conf_mat = np.array(log_data["confusion_matrix"])
        labels = log_data["labels"]

        fig_cm, ax_cm = plt.subplots(figsize=(5.5, 4.2), dpi=130)
        cmap = "Blues" if not is_dark else "magma"
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap=cmap,
                    xticklabels=labels, yticklabels=labels, ax=ax_cm,
                    linewidths=1.2, linecolor="#151c2e" if is_dark else "#ffffff",
                    cbar=False)
        ax_cm.set_xlabel("Predicted Price Tier")
        ax_cm.set_ylabel("Actual Price Tier")
        ax_cm.set_title("Test Confusion Matrix (590 Samples)")
        apply_chart_theme(fig_cm, ax_cm, is_dark)
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

# ==========================================================================
# TAB 5: INSTANT VALUATION CALCULATOR
# ==========================================================================
with tab5:
    st.subheader("🔮 Instant Plot Valuation Calculator")
    st.markdown("Input property dimensions and location to generate real-time consensus valuations from all 6 models.")

    calc_left, calc_right = st.columns([1, 2])

    with calc_left:
        st.markdown("""
            <div class="executive-card">
                <h4 style="margin-top:0;">Input Property Profile</h4>
            </div>
        """, unsafe_allow_html=True)

        selected_loc = st.selectbox("Select Target Locality:", localities, index=localities.index("DHA Defence") if "DHA Defence" in localities else 0)

        area_input = st.number_input("Plot Size:", min_value=0.5, max_value=1000.0, value=10.0, step=0.5)
        unit_choice = st.radio("Unit of Measurement:", ["Marla", "Kanal"], horizontal=True)

        # Quick size presets
        st.caption("Quick Standard Presets:")
        p_c1, p_c2, p_c3 = st.columns(3)
        with p_c1:
            if st.button("5 Marla"):
                st.session_state["preset_area"] = 5.0
                st.session_state["preset_unit"] = "Marla"
        with p_c2:
            if st.button("10 Marla"):
                st.session_state["preset_area"] = 10.0
                st.session_state["preset_unit"] = "Marla"
        with p_c3:
            if st.button("1 Kanal"):
                st.session_state["preset_area"] = 1.0
                st.session_state["preset_unit"] = "Kanal"

        calc_button = st.button("🚀 Calculate Valuation Dossier", use_container_width=True)

    with calc_right:
        if calc_button:
            # Standardize area to Marla and calculate log_area
            standard_marla = area_input * 20.0 if unit_choice == "Kanal" else float(area_input)
            log_marla = np.log10(standard_marla)

            # Construct inference dataframe for Pipeline
            inference_df = pd.DataFrame({
                "area_marla": [standard_marla],
                "log_area": [log_marla],
                "locality": [selected_loc]
            })

            # Execute predictions across all models
            pred_results = {}
            for m_name, m_pipe in models.items():
                pred_results[m_name] = m_pipe.predict(inference_df)[0]

            primary_val = pred_results.get("Linear Regression", 0.0)
            dt_val = pred_results.get("Decision Tree", 0.0)
            tier_pred = pred_results.get("Logistic Regression", "Unknown")

            # Collect all regression predictions
            reg_preds = [v for k, v in pred_results.items() if k != "Logistic Regression"]
            min_val = min(reg_preds)
            max_val = max(reg_preds)

            # Display Executive Valuation Dossier Card
            st.markdown(f"""
                <div class="executive-card" style="border-left: 5px solid #4f46e5;">
                    <span class="highlight-badge">VALUATION DOSSIER • {selected_loc.upper()}</span>
                    <h2 style="margin: 8px 0 2px 0; font-size: 2.2rem; font-weight: 800; color: #4f46e5;">
                        {format_pkr(primary_val)}
                    </h2>
                    <p style="margin: 0 0 16px 0; font-weight: 600; opacity: 0.85;">
                        Recommended Benchmark Valuation (Linear Regression Pipeline)
                    </p>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px;">
                        <div style="background: {'rgba(255,255,255,0.05)' if is_dark else '#f8fafc'}; padding: 12px; border-radius: 8px; border: 1px solid {'#23304c' if is_dark else '#e2e8f0'};">
                            <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; opacity: 0.7;">Consensus Range</span><br>
                            <strong style="font-size: 0.95rem;">{format_pkr(min_val)} – {format_pkr(max_val)}</strong>
                        </div>
                        <div style="background: {'rgba(255,255,255,0.05)' if is_dark else '#f8fafc'}; padding: 12px; border-radius: 8px; border: 1px solid {'#23304c' if is_dark else '#e2e8f0'};">
                            <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; opacity: 0.7;">Market Segment Tier</span><br>
                            <strong style="font-size: 0.95rem; color: #10b981;">{tier_pred} Segment</strong>
                        </div>
                        <div style="background: {'rgba(255,255,255,0.05)' if is_dark else '#f8fafc'}; padding: 12px; border-radius: 8px; border: 1px solid {'#23304c' if is_dark else '#e2e8f0'};">
                            <span style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; opacity: 0.7;">Unit Rate Estimate</span><br>
                            <strong style="font-size: 0.95rem;">{format_pkr(primary_val / standard_marla)} / Marla</strong>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 🔍 Multi-Algorithm Consensus Comparison")

            c_breakdown = []
            for name, val in pred_results.items():
                if name == "Logistic Regression":
                    c_breakdown.append({"Model": name, "Type": "Classification", "Output": f"{val} Tier", "Delta vs Linear": "—"})
                else:
                    delta = val - primary_val
                    delta_str = f"{delta/1e5:+.1f} Lakh" if abs(delta) < 1e7 else f"{delta/1e7:+.2f} Crore"
                    c_breakdown.append({
                        "Model": name,
                        "Type": "Regression",
                        "Output": format_pkr(val),
                        "Delta vs Linear": delta_str if name != "Linear Regression" else "Baseline (0.0)"
                    })

            st.dataframe(pd.DataFrame(c_breakdown), use_container_width=True)

        else:
            st.info("👈 Select a locality and enter plot dimensions on the left, then click **Calculate Valuation Dossier** to generate pricing insights.")

# --------------------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: {'#94a3b8' if is_dark else '#64748b'}; padding: 16px; font-size: 0.85rem;'>"
    "Zameen Intel™ • Automated Property Valuation System • Pair-Programmed for NIAI Task"
    "</div>",
    unsafe_allow_html=True
)
