"""
Clustering Engine Module
Provides clustering algorithms, data preprocessing, dimensionality reduction,
and cluster evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def preprocess_data(
    df: pd.DataFrame,
    features: list,
    impute_strategy: str = "mean",
    scaler_type: str = "StandardScaler",
    encode_categoricals: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, Any, Any]:
    """
    Preprocess selected features from the dataframe:
    - handles missing values
    - encodes categorical variables (if any)
    - applies feature scaling
    Returns (processed_array, processed_df, imputer, scaler)
    """
    sub_df = df[features].copy()

    # Identify numeric and categorical columns
    numeric_cols = sub_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = sub_df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Handle missing values for numeric columns
    imputer = None
    if numeric_cols:
        if impute_strategy != "none":
            imputer = SimpleImputer(strategy=impute_strategy)
            sub_df[numeric_cols] = imputer.fit_transform(sub_df[numeric_cols])

    # Handle categorical columns if present
    if cat_cols:
        if encode_categoricals:
            # Impute categorical with most frequent if needed
            cat_imputer = SimpleImputer(strategy="most_frequent")
            sub_df[cat_cols] = cat_imputer.fit_transform(sub_df[cat_cols])
            # One-hot encode
            sub_df = pd.get_dummies(sub_df, columns=cat_cols, drop_first=False)
        else:
            # Drop categorical columns if not encoding
            sub_df = sub_df.drop(columns=cat_cols)

    # Scaling
    scaler = None
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = None

    if scaler is not None and not sub_df.empty:
        scaled_array = scaler.fit_transform(sub_df)
    else:
        scaled_array = sub_df.values.astype(float)

    processed_df = pd.DataFrame(
        scaled_array, columns=sub_df.columns, index=df.index
    )
    return scaled_array, processed_df, imputer, scaler


def run_clustering(
    data: np.ndarray,
    algorithm: str = "K-Means",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Run the chosen clustering algorithm on preprocessed data.
    Returns (labels, model)
    """
    if params is None:
        params = {}

    if algorithm == "K-Means":
        n_clusters = params.get("n_clusters", 3)
        init = params.get("init", "k-means++")
        max_iter = params.get("max_iter", 300)
        random_state = params.get("random_state", 42)

        model = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            n_init="auto",
        )
        labels = model.fit_predict(data)

    elif algorithm == "DBSCAN":
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)
        metric = params.get("metric", "euclidean")

        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = model.fit_predict(data)

    elif algorithm == "Hierarchical (Agglomerative)":
        n_clusters = params.get("n_clusters", 3)
        linkage = params.get("linkage", "ward")
        metric = params.get("metric", "euclidean")

        # In sklearn >= 1.2, metric='euclidean' when linkage='ward'
        if linkage == "ward":
            model = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage, metric=metric
            )
        labels = model.fit_predict(data)

    elif algorithm == "Gaussian Mixture (GMM)":
        n_components = params.get("n_clusters", 3)
        covariance_type = params.get("covariance_type", "full")
        random_state = params.get("random_state", 42)

        model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state,
        )
        labels = model.fit_predict(data)

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return labels, model


def compute_metrics(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Calculate evaluation metrics for the clustering output.
    Excludes noise points (-1 in DBSCAN) for score calculations when possible.
    """
    unique_labels = np.unique(labels)
    # Check if noise exists (e.g. DBSCAN -1)
    non_noise_mask = labels != -1
    effective_labels = labels[non_noise_mask]
    effective_data = data[non_noise_mask]
    n_effective_clusters = len(np.unique(effective_labels))

    metrics: Dict[str, Any] = {
        "n_clusters": len(unique_labels),
        "unique_labels": unique_labels.tolist(),
        "has_noise": -1 in unique_labels,
        "n_noise_points": int(np.sum(labels == -1)) if -1 in unique_labels else 0,
        "silhouette_score": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    # Silhouette and other metrics need at least 2 clusters and > 1 sample per cluster
    if n_effective_clusters >= 2 and len(effective_labels) > n_effective_clusters:
        try:
            metrics["silhouette_score"] = float(
                silhouette_score(effective_data, effective_labels)
            )
        except Exception:
            metrics["silhouette_score"] = None

        try:
            metrics["davies_bouldin"] = float(
                davies_bouldin_score(effective_data, effective_labels)
            )
        except Exception:
            metrics["davies_bouldin"] = None

        try:
            metrics["calinski_harabasz"] = float(
                calinski_harabasz_score(effective_data, effective_labels)
            )
        except Exception:
            metrics["calinski_harabasz"] = None

    return metrics


def compute_pca_projections(
    data: np.ndarray, n_components: int = 3
) -> np.ndarray:
    """
    Computes PCA projection for 2D or 3D visualization.
    """
    actual_components = min(n_components, data.shape[1], data.shape[0])
    pca = PCA(n_components=actual_components, random_state=42)
    projected = pca.fit_transform(data)

    # Pad with zeros if less than requested components
    if projected.shape[1] < n_components:
        padding = np.zeros((projected.shape[0], n_components - projected.shape[1]))
        projected = np.hstack([projected, padding])

    return projected


def compute_optimal_k(
    data: np.ndarray, min_k: int = 2, max_k: int = 10
) -> Tuple[list, list, list]:
    """
    Computes inertias and silhouette scores for K-Means across k values.
    """
    k_values = []
    inertias = []
    silhouettes = []

    actual_max_k = min(max_k, len(data) - 1)
    if actual_max_k < min_k:
        return [], [], []

    for k in range(min_k, actual_max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        preds = km.fit_predict(data)
        k_values.append(k)
        inertias.append(km.inertia_)
        try:
            score = silhouette_score(data, preds)
            silhouettes.append(score)
        except Exception:
            silhouettes.append(0.0)

    return k_values, inertias, silhouettes

