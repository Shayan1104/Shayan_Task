import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_blobs

import clustering_engine as ce

# Page Configuration
st.set_page_config(
    page_title="Unlabeled Data Clustering Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .cluster-btn>div>button {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: white !important;
        font-size: 1.25rem !important;
        padding: 0.75rem 2.5rem !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.4) !important;
        width: 100%;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .cluster-btn>div>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(79, 70, 229, 0.5) !important;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F46E5;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Helper to load pre-built sample datasets."""
    file_map = {
        "Customer Segments (200 records)": "sample_data/unlabeled_customer_segments.csv",
        "Iris Measurements (150 records)": "sample_data/unlabeled_iris_measurements.csv",
        "Sensor Telemetry (180 records)": "sample_data/unlabeled_sensor_telemetry.csv",
    }
    path = file_map.get(name)
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# Application Header
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-header">🔮 Unlabeled Data Clustering Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Convert unlabeled datasets into structured, labeled clusters with machine learning algorithms.</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 10px;">
            <span style="background-color: #EEF2FF; color: #4338CA; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                🐳 Dockerized Streamlit
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- SIDEBAR -----------------
st.sidebar.header("📁 1. Data Ingestion")
data_source = st.sidebar.radio(
    "Choose Data Input Mode:",
    [
        "📤 Upload Unlabeled File",
        "📦 Load Sample Unlabeled Dataset",
        "📝 Paste Raw CSV / Text",
        "🎲 Generate Synthetic Data",
    ],
)

raw_df: pd.DataFrame = pd.DataFrame()

if data_source == "📤 Upload Unlabeled File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls", "tsv", "txt"],
        help="Upload tabular data without target/label columns.",
    )
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                raw_df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".tsv"):
                raw_df = pd.read_csv(uploaded_file, sep="\t")
            else:
                raw_df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Loaded '{uploaded_file.name}' successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")

elif data_source == "📦 Load Sample Unlabeled Dataset":
    sample_choice = st.sidebar.selectbox(
        "Select a pre-loaded unlabeled dataset:",
        [
            "Customer Segments (200 records)",
            "Iris Measurements (150 records)",
            "Sensor Telemetry (180 records)",
        ],
    )
    raw_df = load_sample_dataset(sample_choice)
    st.sidebar.info(f"Loaded: {sample_choice}")

elif data_source == "📝 Paste Raw CSV / Text":
    default_text = "Feature1,Feature2,Feature3\n1.2,3.4,5.1\n1.5,3.1,4.9\n8.2,9.1,7.8\n8.5,8.9,8.1\n4.1,1.2,2.3\n4.4,1.5,2.1"
    pasted_text = st.sidebar.text_area(
        "Paste comma-separated or tab-separated text:",
        value=default_text,
        height=150,
    )
    if pasted_text:
        try:
            sep = "\t" if "\t" in pasted_text else ","
            raw_df = pd.read_csv(io.StringIO(pasted_text), sep=sep)
        except Exception as e:
            st.sidebar.error(f"Failed to parse text: {e}")

elif data_source == "🎲 Generate Synthetic Data":
    n_samples = st.sidebar.slider("Number of Samples", 50, 1000, 300, step=50)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 4)
    true_clusters = st.sidebar.slider("Cluster Centers", 2, 8, 3)
    cluster_std = st.sidebar.slider("Cluster Spread (Std)", 0.5, 2.5, 1.0, step=0.1)

    if st.sidebar.button("Regenerate Synthetic Data") or "synth_data" not in st.session_state:
        X, _ = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=true_clusters,
            cluster_std=cluster_std,
            random_state=None,
        )
        col_names = [f"Feature_{i+1}" for i in range(n_features)]
        st.session_state["synth_data"] = pd.DataFrame(X, columns=col_names)
    raw_df = st.session_state.get("synth_data", pd.DataFrame())

# Verify we have data
if raw_df.empty:
    st.info("👈 Please select a data source or upload an unlabeled dataset in the sidebar to begin.")
    st.stop()

# ----------------- PREPROCESSING & FEATURE SELECTION -----------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 2. Features & Preprocessing")

all_columns = raw_df.columns.tolist()
numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = [c for c in all_columns if c not in numeric_cols]

# Exclude obvious ID / string columns by default if numeric cols exist
default_features = numeric_cols if numeric_cols else all_columns

selected_features = st.sidebar.multiselect(
    "Select Features for Clustering:",
    options=all_columns,
    default=default_features,
    help="Select numerical or categorical features that determine the clusters.",
)

if not selected_features:
    st.sidebar.warning("Select at least one feature.")
    st.stop()

with st.sidebar.expander("Advanced Preprocessing Options", expanded=False):
    impute_strategy = st.selectbox(
        "Missing Values Imputation:",
        options=["mean", "median", "most_frequent", "none"],
        index=0,
    )
    scaler_type = st.selectbox(
        "Feature Scaling:",
        options=["StandardScaler", "MinMaxScaler", "RobustScaler", "None"],
        index=0,
        help="StandardScaler normalizes mean=0, std=1. Highly recommended for distance-based clustering.",
    )
    encode_categoricals = st.checkbox(
        "One-Hot Encode Categorical Columns",
        value=True,
        help="Convert string / categorical features into binary indicator variables.",
    )

# ----------------- CLUSTERING ALGORITHM SELECTION -----------------
st.sidebar.markdown("---")
st.sidebar.header("🧠 3. Clustering Algorithm")

algorithm = st.sidebar.selectbox(
    "Algorithm:",
    [
        "K-Means",
        "DBSCAN",
        "Hierarchical (Agglomerative)",
        "Gaussian Mixture (GMM)",
    ],
    index=0,
)

algo_params = {}
if algorithm == "K-Means":
    k_clusters = st.sidebar.slider("Number of Clusters (K):", min_value=2, max_value=12, value=3)
    algo_params["n_clusters"] = k_clusters
    algo_params["init"] = st.sidebar.selectbox("Centroid Initialization:", ["k-means++", "random"])
    algo_params["max_iter"] = st.sidebar.slider("Max Iterations:", 100, 1000, 300, step=50)

elif algorithm == "DBSCAN":
    st.sidebar.caption("Density-Based Clustering: Detects clusters of arbitrary shapes and isolates noise (-1).")
    eps = st.sidebar.slider("Epsilon (Neighborhood Radius):", 0.1, 5.0, 0.6, step=0.05)
    min_samples = st.sidebar.slider("Min Samples per Cluster:", 2, 20, 5)
    algo_params["eps"] = eps
    algo_params["min_samples"] = min_samples

elif algorithm == "Hierarchical (Agglomerative)":
    k_clusters = st.sidebar.slider("Number of Clusters:", min_value=2, max_value=12, value=3)
    linkage = st.sidebar.selectbox("Linkage Criterion:", ["ward", "complete", "average", "single"])
    algo_params["n_clusters"] = k_clusters
    algo_params["linkage"] = linkage

elif algorithm == "Gaussian Mixture (GMM)":
    n_components = st.sidebar.slider("Number of Components (Clusters):", min_value=2, max_value=12, value=3)
    cov_type = st.sidebar.selectbox("Covariance Type:", ["full", "tied", "diag", "spherical"])
    algo_params["n_clusters"] = n_components
    algo_params["covariance_type"] = cov_type


# ----------------- MAIN CONTENT: DATA OVERVIEW -----------------
st.subheader("📋 1. Unlabeled Input Data Preview")
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Records", f"{len(raw_df):,}")
with col_m2:
    st.metric("Total Columns", f"{raw_df.shape[1]}")
with col_m3:
    st.metric("Features for Clustering", f"{len(selected_features)}")
with col_m4:
    missing_count = int(raw_df[selected_features].isnull().sum().sum())
    st.metric("Missing Values", f"{missing_count}")

with st.expander("👁️ View Input Data Table & Summary Statistics", expanded=True):
    tab_data, tab_stats = st.tabs(["Data Table", "Summary Statistics"])
    with tab_data:
        st.dataframe(raw_df, use_container_width=True, height=220)
    with tab_stats:
        st.dataframe(raw_df[selected_features].describe().T, use_container_width=True)

# ----------------- THE "CLUSTER" ACTION -----------------
st.markdown("---")
st.markdown("### ⚡ 2. Transform Unlabeled Data into Labeled Data")
st.write("Click the button below to execute preprocessing, run the algorithm, and generate cluster labels.")

col_btn, col_empty = st.columns([1, 1])
with col_btn:
    st.markdown('<div class="cluster-btn">', unsafe_allow_html=True)
    cluster_clicked = st.button("🚀 CLUSTER", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

# Save clustering state
if cluster_clicked or "clustered_df" in st.session_state:
    if cluster_clicked:
        with st.spinner("Processing data, applying algorithms, and assigning labels..."):
            try:
                # Preprocess
                scaled_data, processed_df, imputer, scaler = ce.preprocess_data(
                    df=raw_df,
                    features=selected_features,
                    impute_strategy=impute_strategy,
                    scaler_type=scaler_type,
                    encode_categoricals=encode_categoricals,
                )

                # Cluster
                labels, model = ce.run_clustering(
                    data=scaled_data,
                    algorithm=algorithm,
                    params=algo_params,
                )

                # Compute Metrics
                metrics = ce.compute_metrics(scaled_data, labels)

                # Create labeled dataframe
                labeled_df = raw_df.copy()
                labeled_df["Cluster"] = labels
                # Friendly label name
                labeled_df["Cluster_Label"] = [
                    f"Cluster_{c}" if c != -1 else "Noise/Outlier" for c in labels
                ]

                # Store in session state
                st.session_state["clustered_df"] = labeled_df
                st.session_state["scaled_data"] = scaled_data
                st.session_state["labels"] = labels
                st.session_state["metrics"] = metrics
                st.session_state["algorithm"] = algorithm
                st.session_state["features_used"] = selected_features
                st.session_state["model"] = model

                st.success("🎉 Clustering completed successfully! Labeled dataset generated below.")
            except Exception as ex:
                st.error(f"Error during clustering execution: {ex}")
                st.stop()

    # Retrieve from session state
    labeled_df = st.session_state["clustered_df"]
    scaled_data = st.session_state["scaled_data"]
    labels = st.session_state["labels"]
    metrics = st.session_state["metrics"]
    algo_used = st.session_state["algorithm"]
    features_used = st.session_state["features_used"]

    # ----------------- RESULTS PRESENTATION -----------------
    st.markdown("### 📊 3. Clustering Results & Evaluation")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Clusters Detected", f"{metrics['n_clusters']}")
    with c2:
        sil = metrics.get("silhouette_score")
        sil_display = f"{sil:.3f}" if sil is not None else "N/A"
        st.metric(
            "Silhouette Score",
            sil_display,
            help="Ranges -1 to 1. Higher values (> 0.5) denote dense, well-separated clusters.",
        )
    with c3:
        db = metrics.get("davies_bouldin")
        db_display = f"{db:.3f}" if db is not None else "N/A"
        st.metric(
            "Davies-Bouldin Index",
            db_display,
            help="Lower values indicate better clustering (minimal cluster overlap).",
        )
    with c4:
        if metrics.get("has_noise"):
            st.metric("Noise Points (Outliers)", f"{metrics['n_noise_points']}")
        else:
            ch = metrics.get("calinski_harabasz")
            ch_display = f"{ch:.1f}" if ch is not None else "N/A"
            st.metric("Calinski-Harabasz", ch_display, help="Higher values indicate tighter clusters.")

    # TABS FOR LABELED DATA, VISUALIZATIONS, PROFILING
    tab_labeled, tab_viz, tab_profiles, tab_elbow = st.tabs(
        [
            "💾 Labeled Data & Export",
            "🌐 Interactive Visualizations",
            "📈 Cluster Profiles & Distributions",
            "📉 Optimal K Advisor",
        ]
    )

    # TAB 1: LABELED DATA & EXPORT
    with tab_labeled:
        st.markdown("#### Labeled Dataset")
        st.caption(
            "Your raw unlabeled data now contains the newly generated **Cluster** and **Cluster_Label** columns."
        )

        # Filter by cluster
        cluster_options = ["All"] + sorted(list(labeled_df["Cluster_Label"].unique()))
        selected_cluster_filter = st.selectbox("Filter preview by Cluster:", cluster_options)

        filtered_df = (
            labeled_df
            if selected_cluster_filter == "All"
            else labeled_df[labeled_df["Cluster_Label"] == selected_cluster_filter]
        )

        st.dataframe(filtered_df, use_container_width=True, height=300)

        # Export Center
        st.markdown("##### 📥 Download Converted Labeled Dataset")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            csv_data = labeled_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Labeled CSV",
                data=csv_data,
                file_name="clustered_labeled_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_dl2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                labeled_df.to_excel(writer, index=False, sheet_name="Clustered_Data")
            st.download_button(
                label="⬇️ Download Labeled Excel (.xlsx)",
                data=buffer.getvalue(),
                file_name="clustered_labeled_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # TAB 2: INTERACTIVE VISUALIZATIONS
    with tab_viz:
        st.markdown("#### Cluster Visualizer")
        viz_mode = st.radio(
            "Projection Mode:",
            ["2D PCA Projection", "3D PCA Projection", "Custom Feature Pair (2D)", "Custom Feature Trio (3D)"],
            horizontal=True,
        )

        if viz_mode == "2D PCA Projection":
            pca_proj = ce.compute_pca_projections(scaled_data, n_components=2)
            viz_df = labeled_df.copy()
            viz_df["PCA_1"] = pca_proj[:, 0]
            viz_df["PCA_2"] = pca_proj[:, 1]

            fig = px.scatter(
                viz_df,
                x="PCA_1",
                y="PCA_2",
                color="Cluster_Label",
                hover_data=features_used[:6],
                title="2D Principal Component Analysis (PCA) Cluster Map",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color="white")))
            fig.update_layout(height=520, legend_title_text="Assigned Cluster")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "3D PCA Projection":
            pca_proj = ce.compute_pca_projections(scaled_data, n_components=3)
            viz_df = labeled_df.copy()
            viz_df["PCA_1"] = pca_proj[:, 0]
            viz_df["PCA_2"] = pca_proj[:, 1]
            viz_df["PCA_3"] = pca_proj[:, 2]

            fig = px.scatter_3d(
                viz_df,
                x="PCA_1",
                y="PCA_2",
                z="PCA_3",
                color="Cluster_Label",
                hover_data=features_used[:5],
                title="3D PCA Interactive Cluster Space",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_traces(marker=dict(size=5, opacity=0.85))
            fig.update_layout(height=600, legend_title_text="Cluster")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Custom Feature Pair (2D)":
            col_x, col_y = st.columns(2)
            with col_x:
                feat_x = st.selectbox("X-Axis Feature:", features_used, index=0)
            with col_y:
                y_idx = 1 if len(features_used) > 1 else 0
                feat_y = st.selectbox("Y-Axis Feature:", features_used, index=y_idx)

            fig = px.scatter(
                labeled_df,
                x=feat_x,
                y=feat_y,
                color="Cluster_Label",
                hover_data=features_used[:6],
                title=f"Cluster Separation: {feat_x} vs {feat_y}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="white")))
            fig.update_layout(height=520, legend_title_text="Cluster")
            st.plotly_chart(fig, use_container_width=True)

        elif viz_mode == "Custom Feature Trio (3D)":
            col_x, col_y, col_z = st.columns(3)
            with col_x:
                feat_x = st.selectbox("X Feature:", features_used, index=0)
            with col_y:
                y_idx = 1 if len(features_used) > 1 else 0
                feat_y = st.selectbox("Y Feature:", features_used, index=y_idx)
            with col_z:
                z_idx = 2 if len(features_used) > 2 else 0
                feat_z = st.selectbox("Z Feature:", features_used, index=z_idx)

            fig = px.scatter_3d(
                labeled_df,
                x=feat_x,
                y=feat_y,
                z=feat_z,
                color="Cluster_Label",
                hover_data=features_used[:5],
                title=f"3D Scatter: {feat_x} vs {feat_y} vs {feat_z}",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Vivid,
            )
            fig.update_traces(marker=dict(size=5, opacity=0.85))
            fig.update_layout(height=600, legend_title_text="Cluster")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: CLUSTER PROFILES & DISTRIBUTIONS
    with tab_profiles:
        st.markdown("#### Cluster Demographics & Feature Means")
        
        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            # Cluster count chart
            counts = labeled_df["Cluster_Label"].value_counts().reset_index()
            counts.columns = ["Cluster_Label", "Count"]
            counts["Percentage"] = (counts["Count"] / len(labeled_df) * 100).round(1)

            fig_counts = px.bar(
                counts,
                x="Cluster_Label",
                y="Count",
                text="Percentage",
                color="Cluster_Label",
                title="Cluster Population Distribution",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_counts.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_counts.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_counts, use_container_width=True)

        with col_p2:
            st.markdown("##### Cluster Size Table")
            st.dataframe(counts, use_container_width=True, height=280)

        # Numeric averages per cluster
        numeric_feature_subset = [
            f for f in features_used if np.issubdtype(labeled_df[f].dtype, np.number)
        ]
        if numeric_feature_subset:
            st.markdown("##### Feature Means by Cluster")
            profile_df = labeled_df.groupby("Cluster_Label")[numeric_feature_subset].mean().round(2)
            st.dataframe(profile_df.style.highlight_max(axis=0, color="#D1FAE5"), use_container_width=True)

            # Feature Boxplot comparison
            selected_box_feat = st.selectbox("Inspect Feature Distribution Across Clusters:", numeric_feature_subset)
            fig_box = px.box(
                labeled_df,
                x="Cluster_Label",
                y=selected_box_feat,
                color="Cluster_Label",
                title=f"Distribution of '{selected_box_feat}' per Cluster",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig_box.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig_box, use_container_width=True)

    # TAB 4: OPTIMAL K ADVISOR
    with tab_elbow:
        st.markdown("#### Optimal Number of Clusters (K) Analysis")
        st.write(
            "Use the Elbow Method and Silhouette Analysis to statistically find the most natural number of clusters for this dataset."
        )

        if st.button("🔍 Compute Elbow & Silhouette Curves"):
            with st.spinner("Evaluating K from 2 to 10..."):
                k_vals, inertias, sil_scores = ce.compute_optimal_k(scaled_data, min_k=2, max_k=10)

                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    fig_elbow = go.Figure()
                    fig_elbow.add_trace(
                        go.Scatter(
                            x=k_vals,
                            y=inertias,
                            mode="lines+markers",
                            line=dict(color="#4F46E5", width=3),
                            marker=dict(size=8),
                        )
                    )
                    fig_elbow.update_layout(
                        title="Elbow Method (Inertia vs K)",
                        xaxis_title="Number of Clusters (K)",
                        yaxis_title="Sum of Squared Errors (Inertia)",
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig_elbow, use_container_width=True)

                with col_k2:
                    fig_sil = go.Figure()
                    fig_sil.add_trace(
                        go.Scatter(
                            x=k_vals,
                            y=sil_scores,
                            mode="lines+markers",
                            line=dict(color="#059669", width=3),
                            marker=dict(size=8),
                        )
                    )
                    fig_sil.update_layout(
                        title="Silhouette Score vs K (Peak = Optimal)",
                        xaxis_title="Number of Clusters (K)",
                        yaxis_title="Mean Silhouette Coefficient",
                        template="plotly_white",
                        height=400,
                    )
                    st.plotly_chart(fig_sil, use_container_width=True)

                if sil_scores:
                    best_k = k_vals[np.argmax(sil_scores)]
                    st.success(
                        f"💡 **Recommended K:** Based on maximum silhouette score, **K = {best_k}** yields the cleanest cluster separation."
                    )

