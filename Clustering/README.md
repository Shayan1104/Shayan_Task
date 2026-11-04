# 🔮 Unlabeled Data Clustering Studio

A full-stack, interactive Machine Learning web application built with **Streamlit** and containerized with **Docker**.

Easily ingest raw, unlabeled datasets (via CSV/Excel upload, pre-loaded industry benchmarks, raw text, or synthetic generation), configure unsupervised algorithms, and transform unlabeled rows into labeled clusters with a single click of the **CLUSTER** button.

---

## 🌟 Key Features

1. **Multiple Unlabeled Data Ingestion Methods**:
   - 📤 **Upload Files**: CSV, Excel (`.xlsx`, `.xls`), TSV, or TXT.
   - 📦 **Pre-loaded Benchmarks**:
     - *Customer Segmentation* (200 records: Age, Income, Spending Score, Store Visits, Cart Value).
     - *Iris Morphology* (150 records: Sepal & Petal measurements without species target).
     - *Industrial Sensor Telemetry* (180 records: Temperature, Vibration, Pressure, Power Draw).
   - 📝 **Paste Raw Tabular Text**: Quickly test snippets of tabular data.
   - 🎲 **Synthetic Generator**: Generate Gaussian blob distributions on the fly.

2. **Automated Preprocessing & Feature Selection**:
   - Select arbitrary subsets of numeric and categorical columns.
   - Missing value imputation (Mean, Median, Most Frequent).
   - Feature scaling (**StandardScaler**, **MinMaxScaler**, **RobustScaler**).
   - Categorical One-Hot Encoding.

3. **Multi-Algorithm Clustering Suite**:
   - **K-Means**: Configurable $K$, centroid initialization (`k-means++`, `random`), max iterations.
   - **DBSCAN**: Density-based clustering with automatic noise point detection (`-1` label).
   - **Hierarchical (Agglomerative)**: Ward, Complete, Average, and Single linkage criteria.
   - **Gaussian Mixture Models (GMM)**: Soft clustering with flexible covariance matrices.

4. **One-Click "CLUSTER" Execution**:
   - Instantly appends `Cluster` and `Cluster_Label` columns to the input data.
   - Calculates statistical evaluation metrics:
     - **Silhouette Score** (density and separation quality).
     - **Davies-Bouldin Index** (cluster compactness).
     - **Calinski-Harabasz Score** (variance ratio criterion).
     - Noise point tracking.

5. **Visualizations & Export**:
   - **2D & 3D PCA Space**: Interactive Plotly projections.
   - **Custom Feature Pairs & Trios**: Scatter plots with full hover tooltips.
   - **Cluster Demographics**: Population distribution and feature mean profiles.
   - **Optimal K Finder**: Interactive Elbow Curve and Silhouette score progression.
   - **Export Options**: Download labeled dataset as **CSV** or **Excel (.xlsx)**.

---

## 🐳 Running with Docker (Recommended)

You can run this application entirely through Docker by simply pulling the image and running the container.

### 1. Pull the Image from Docker Hub
```bash
docker pull <your-dockerhub-username>/clustering-app:latest
```

### 2. Run the Container
```bash
docker run -d -p 8501:8501 --name clustering-app <your-dockerhub-username>/clustering-app:latest
```

### 3. Open in Browser
Visit **`http://localhost:8501`** in any web browser.

---

## 🚀 Building & Pushing to Docker Hub

If you wish to build the image locally and push it to your personal or organization's Docker Hub repository:

### Method A: Using the Automated Script

- **Windows**:
  ```cmd
  push_to_dockerhub.bat <your-dockerhub-username>
  ```
- **Linux / macOS**:
  ```bash
  chmod +x push_to_dockerhub.sh
  ./push_to_dockerhub.sh <your-dockerhub-username>
  ```

### Method B: Manual Docker Commands

1. **Log in to Docker Hub**:
   ```bash
   docker login
   ```

2. **Build the Docker Image**:
   ```bash
   docker build -t <your-dockerhub-username>/clustering-app:latest .
   ```

3. **Push to Docker Hub**:
   ```bash
   docker push <your-dockerhub-username>/clustering-app:latest
   ```

4. **Verify by Pulling and Running on any machine**:
   ```bash
   docker pull <your-dockerhub-username>/clustering-app:latest
   docker run -p 8501:8501 <your-dockerhub-username>/clustering-app:latest
   ```

---

## 💻 Local Development Setup (Without Docker)

If you prefer running directly in Python:

1. **Clone or navigate to the directory**:
   ```bash
   cd "c:\Users\hp\Desktop\Netsol Tasks\Clustering"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

4. The app will launch automatically at **`http://localhost:8501`**.

---

## 📁 Repository Structure

```
Clustering/
├── app.py                      # Main Streamlit web application & UI
├── clustering_engine.py        # Preprocessing, algorithms, metrics & PCA engine
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Production container definition
├── .dockerignore               # Docker build exclusions
├── docker-compose.yml          # Compose service definition
├── push_to_dockerhub.bat       # Windows helper to build & push to Docker Hub
├── push_to_dockerhub.sh        # Unix/Linux helper to build & push to Docker Hub
├── run_docker.bat              # Windows helper to pull & run container
├── run_docker.sh               # Unix/Linux helper to pull & run container
├── sample_data/                # Pre-loaded benchmark datasets
│   ├── unlabeled_customer_segments.csv
│   ├── unlabeled_iris_measurements.csv
│   └── unlabeled_sensor_telemetry.csv
└── README.md                   # Full documentation & usage guide
```

