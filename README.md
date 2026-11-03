# Applied AI & Machine Learning Systems

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-61DAFB.svg?logo=react&logoColor=black)](https://react.dev/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A collection of applied Artificial Intelligence, Machine Learning, and Retrieval-Augmented Generation (RAG) applications developed for enterprise workflows, automated data auditing, and predictive real estate valuation.

---

## 📌 Repository Architecture

```
applied-ai-ml-systems/
├── Data_embedder/             # Neural Document Vectorizer & Grounded RAG Chatbot
├── Real_Estate_Valuation/     # Zameen Real Estate Valuation Platform (React + FastAPI + ML)
├── Real_Estate_Streamlit/     # Streamlit Interactive Prototype & Visual EDA Pipeline
├── Email_spam_pipeline/       # Automated NLP Data Cleaning, Profiling & PDF Audit Pipeline
└── ML_Foundations/            # Foundational Supervised Regression & KNN Modeling Lab
```

---

## 1. 🧠 DataEmbedder AI — Neural Document Vectorizer & RAG System

### Problem
Organizations handle heterogeneous document formats (PDFs, Word documents, Markdown, raw text, and live web pages) with unstructured data scattered across directories. Finding grounded answers with exact source citations without manual page-by-page review is slow and error-prone.

### Solution
A full-stack document intelligence application that ingests whole folders or web URLs, partitions documents into semantic text chunks with exact page tracking, embeds them using local dense vector embeddings, and provides a grounded conversational RAG chatbot with source page citations.

### Key Capabilities
- **Multi-Source Ingestion**: Batch folder ingestion, multi-page PDFs (`pypdf`), Microsoft Word (`python-docx`), TXT, Markdown, CSV, JSON, ZIP archives, and live URL web scraping via `WebBaseLoader`.
- **Dynamic Semantic Chunking**: Recursive character chunker with real-time UI sliders for Chunk Size (200–2500 chars) and Overlap (0–400 chars) with live page-count and chunk-count telemetry.
- **Local Dense Embeddings**: 384-dimensional dense vectors generated via `sentence-transformers/all-MiniLM-L6-v2` — runs 100% locally and free without external API quotas. ChromaDB integration for persistent vector storage.
- **Hybrid RAG Chatbot**: Grounded Q&A with exact source pill citations (filename, page number, similarity percentage). Supports:
  - *Offline Mode*: Zero-API local semantic extractor.
  - *Cloud LLMs*: Seamless integration with **Groq** (Llama 3.3 70B / 3.1 8B), **Google Gemini 1.5 Flash**, and **OpenAI GPT-4o-mini**.
- **User Interface**: Glassmorphic dark-mode web application built with vanilla HTML5, CSS3, and JavaScript.

### Stack
`FastAPI`, `Uvicorn`, `Sentence-Transformers`, `ChromaDB`, `PyPDF`, `Python-docx`, `BeautifulSoup4`, `Groq SDK`, `Google Generative AI`, `OpenAI SDK`, `Pydantic`.

### How to Run
```bash
cd Data_embedder

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional for cloud LLMs)
cp .env.example .env
# Edit .env with your GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY

# 3. Launch application
python run.py
# Application is live at: http://127.0.0.1:8080
```

---

## 2. 🏡 Zameen Real Estate Valuation & ML Dashboard

### Problem
Real estate pricing in developing markets like Pakistan suffers from extreme price variance, non-standard unit systems (*Marla*, *Kanal*, *Sq. Yards*), and lack of transparent pricing benchmarks for buyers and investors.

### Solution
An end-to-end Machine Learning property valuation platform trained on 20,000+ listings from Zameen.com. Features automated unit conversion, outlier detection via IQR, scikit-learn pipelines with hyperparameter tuning, a FastAPI REST service, and a modern React 18 dashboard.

### Key Capabilities
- **End-to-End Pipeline**:
  - `src/01_explore.py`: Distribution audits, missingness checks, and anomaly detection.
  - `src/02_clean.py`: Unit standardization (e.g. 1 Kanal = 20 Marla), feature engineering, log-transformation, and IQR outlier clipping.
  - `src/03_eda.py`: Automated export of visual distributions, correlation heatmaps, and price-by-locality boxplots.
  - `src/04_train.py`: Cross-validated Scikit-learn pipelines with `StandardScaler`, `OneHotEncoder`, and `GridSearchCV`.
- **Benchmarked Models**:
  - Linear Regression (Baseline)
  - Polynomial Regression (Degree 2)
  - K-Nearest Neighbors (KNN Regressor)
  - Support Vector Regression (Linear SVR)
  - Decision Tree Regressor
  - Logistic Regression (Price-tier classifier: Low / Medium / High bands)
- **FastAPI REST API**: Serves real-time predictions, evaluation metrics, and EDA summary data.
- **Interactive React Frontend**: Modern dark-mode UI with KPI cards, multi-model price predictors with PKR Millions/Crores conversion, and interactive Recharts visualizations (Actual vs. Predicted scatter plots, model comparison bar charts).

### Stack
`Python`, `Scikit-Learn`, `Pandas`, `NumPy`, `FastAPI`, `React 18`, `Vite`, `Tailwind CSS`, `Recharts`, `Lucide React`, `Streamlit`.

### How to Run

#### Option A: Full-Stack React + FastAPI Application
```bash
# Terminal 1: FastAPI Backend
cd Real_Estate_Valuation
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
# API documentation available at: http://127.0.0.1:8000/docs

# Terminal 2: React Frontend
cd Real_Estate_Valuation/frontend
npm install
npm run dev
# Dashboard live at: http://localhost:5173
```

#### Option B: Streamlit Prototype
```bash
cd Real_Estate_Streamlit
pip install -r requirements.txt
streamlit run app.py
```

---

## 3. 📧 Automated NLP Data Profiling & Cleaning Pipeline

### Problem
Raw text and NLP datasets frequently arrive with missing values, malformed encodings, redundant columns, and noisy formatting, requiring tedious manual data-wrangling before modeling can begin.

### Solution
A modular, production-ready data cleaning and profiling engine designed for text classification datasets (`spam.csv`). Automatically cleans the data, detects anomalies, and generates publication-grade PDF and JSON audit reports.

### Key Capabilities
- **Automated Profiling**: Analyzes null rates, duplicate records, class distributions, and memory footprints.
- **Robust Cleaning**: Standardizes non-standard missing values (`NA`, `null`, whitespace), normalizes text content, handles outliers, and standardizes labels.
- **Automated Reporting**:
  - Machine-readable audit: `reports/cleaning_report.json`
  - Executive visual audit: `reports/cleaning_report.pdf` generated with `reportlab`.

### Stack
`Python`, `Pandas`, `Scikit-Learn`, `ReportLab`.

### How to Run
```bash
cd Email_spam_pipeline/data_cleaning
python main.py
# Cleaned dataset saved to: cleaned_data/spam_cleaned.csv
# Audit reports generated in: reports/cleaning_report.pdf & .json
```

---

## 4. 📊 Supervised Machine Learning Foundations

### Problem & Solution
Experimental testbed implementing and evaluating core supervised regression and classification models from scratch and via Scikit-Learn:
- **Linear Regression**: Continuous value prediction evaluating Mean Squared Error (MSE) and $R^2$ scores.
- **Polynomial Regression**: Modeling non-linear feature relationships.
- **K-Nearest Neighbors (KNN)**: Non-parametric regression on multidimensional feature sets.

### How to Run
```bash
cd ML_Foundations
python linear.py
python polynomial.py
python KNN.py
```

---

## 🛠️ Repository Standards & Hygiene
- **Strict `.gitignore`**: Excludes local virtual environments (`.venv`), temporary caches (`__pycache__`), build outputs (`node_modules`, `dist`), and runtime vector stores (`chroma_db`).
- **Secret Protection**: API keys are isolated in local `.env` files and never committed to version control.
- **Reproducible Environments**: Every project directory provides explicit `requirements.txt` or `package.json` configurations.
