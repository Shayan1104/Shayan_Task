# DataEmbedder AI — Neural Document Vectorizer & RAG Chatbot

An intelligent, full-stack document processing application designed to ingest whole folders, PDFs, Word documents, text files, and web pages, dynamically partition them into semantic chunks with exact page tracking, embed them into high-dimensional vector representations, and provide a conversational Retrieval-Augmented Generation (RAG) chatbot.

---

## ✨ Features Implemented

1. **Document & Folder Ingestion**:
   - **Whole Folder Ingestion**: Upload an entire folder directory via the UI or enter any local filesystem directory path.
   - **Multi-Format Support**: Native page-by-page parsing for `.pdf` (using `pypdf`), Microsoft Word `.docx` (using `python-docx`), `.txt`, `.md`, `.csv`, `.json`, `.py`, and `.zip` archives.
   - **WebBaseLoader**: Scrape live webpage content, cleaning boilerplate headers, footers, and scripts.

2. **Semantic Chunking & Page Count Metrics**:
   - Recursive character text chunker preserving paragraph and sentence boundaries.
   - **Real-Time UI Metrics**: Displays **Total Pages** across all documents and **Total Chunks** in prominent glowing badges.
   - **Adjustable Parameters**: Live sliders for **Chunk Size** (200–2500 chars) and **Chunk Overlap** (0–400 chars).
   - **Interactive Chunk Inspector**: Document breakdown summary table and preview list showing chunk index, page number, and estimated token counts.

3. **Vector Conversion Engine ("Embed All Data")**:
   - Dedicated action button: **"Convert Chunks into Vectors (Embed All Data)"**.
   - **Default Local Embedding Model**: High-performance `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) — runs **100% locally and free without API keys**.
   - **Vector Database**: Fast normalized cosine similarity vector index with telemetry (vector dimensions, elapsed indexing time, active model).

4. **Neural Document Mini Chatbot (RAG)**:
   - Grounded Q&A against the uploaded and embedded documents.
   - **Source Citations**: Every answer includes clickable source pills displaying the exact document name, page number (e.g. `netsol_handbook.pdf (P.2)`), and similarity score percentage.
   - **Hybrid LLM Support**:
     - *Offline Mode*: Smart local semantic extractor synthesizing direct excerpts with citations.
     - *Cloud LLM Mode*: Seamless connection with **Google Gemini 1.5 Flash**, **OpenAI GPT-4o-mini**, or **Groq Llama-3.3-70b** via the Settings modal.

---

## 🏗️ Architecture

```
Data_embedder/
├── backend/
│   ├── app.py               # FastAPI REST endpoints & static file serving
│   ├── loaders.py           # Multi-format parsers (PDF, DOCX, TXT, WebBaseLoader, Folder)
│   ├── chunker.py           # Recursive chunker with page & chunk statistics
│   ├── vector_store.py      # Vector Store manager (SentenceTransformers & Cosine Index)
│   └── rag_engine.py        # RAG retrieval & answer generation (Gemini, OpenAI, Groq, Local)
├── static/
│   ├── index.html           # Dark-mode glassmorphic single-page application
│   ├── css/
│   │   └── style.css        # Responsive stylesheet with vibrant glowing accents
│   └── js/
│       └── app.js           # Client-side state, event handlers, and REST integrations
├── sample_docs/             # Multi-page test documents (PDF, MD, TXT)
├── run.py                   # One-click application launcher
├── requirements.txt         # Project dependencies
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.11 recommended).

### 2. Activate Virtual Environment & Install Dependencies
```bash
# If using existing virtual environment:
.\.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python run.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8080
```

---

## 📖 How to Use

1. **Step 1: Load Data**
   - Click the **"Whole Folder"** tab and type `sample_docs` in the path input, then click **"Load Folder"** (or drag & drop any PDF/file).
   - Notice the telemetry bar update with loaded document count and total pages.

2. **Step 2: Split into Chunks**
   - Adjust the **Chunk Size** and **Chunk Overlap** sliders if desired.
   - Click **"Split Documents into Chunks"**.
   - Notice the **TOTAL PAGES** and **TOTAL CHUNKS** cards display live statistics, along with the detailed per-document breakdown in the Chunk Inspector.

3. **Step 3: Convert Chunks into Vectors**
   - Click the glowing **"Convert Chunks into Vectors (Embed All Data)"** button.
   - The engine generates 384-dimensional vector embeddings for all chunks.

4. **Step 4: Ask Questions in the Chatbot**
   - Type your question in the chat input or click one of the suggested prompt chips.
   - Example questions:
     - *"What is the hybrid work policy and on-site days?"*
     - *"What were the Q3 revenue highlights and R&D expenditures?"*
     - *"What embedding models and vector dimensions are used?"*
   - View the grounded answer and verify the citations showing exact file names and page numbers!

5. **(Optional) Settings & Cloud LLMs**
   - Click the **"Settings"** button in the top navigation bar.
   - Enter your **Google Gemini**, **OpenAI**, or **Groq** API key to enable conversational generative answers.
