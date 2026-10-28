import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.loaders import LoadedPage, load_file_auto, load_folder, load_web_url, load_zip_file
from backend.chunker import DocumentChunk, split_pages_into_chunks
from backend.vector_store import VectorStoreManager
from backend.rag_engine import answer_question

UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploaded_data"))
STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Data Embedder & Neural RAG Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application Global In-Memory State
class AppState:
    def __init__(self):
        self.loaded_pages: List[LoadedPage] = []
        self.chunks: List[DocumentChunk] = []
        self.vector_store = VectorStoreManager()
        self.last_chunk_stats: Dict[str, Any] = {}

    def clear(self):
        self.loaded_pages = []
        self.chunks = []
        self.vector_store.clear()
        self.last_chunk_stats = {}
        # Clear uploaded directory
        for f in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, f)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error cleaning {file_path}: {e}")

state = AppState()

# Request Models
class FolderPathRequest(BaseModel):
    folder_path: str

class UrlRequest(BaseModel):
    url: str

class SplitRequest(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100

class VectorizeRequest(BaseModel):
    provider: str = "local"
    api_key: Optional[str] = None

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = []
    llm_provider: Optional[str] = "auto"
    api_keys: Optional[Dict[str, str]] = {}
    top_k: Optional[int] = 4


@app.get("/api/status")
async def get_status():
    doc_names = list(set(p.filename for p in state.loaded_pages))
    total_pages = sum(len(set(p.page_number for p in state.loaded_pages if p.filename == name)) for name in doc_names)
    
    return {
        "total_documents": len(doc_names),
        "total_pages": total_pages,
        "total_chunks": len(state.chunks),
        "total_vectors": len(state.vector_store.items),
        "vector_dimension": state.vector_store.dimension,
        "active_provider": state.vector_store.active_provider,
        "last_chunk_stats": state.last_chunk_stats
    }


@app.post("/api/upload-files")
async def upload_files(files: List[UploadFile] = File(...)):
    """Receives multiple files or ZIP archive and parses them."""
    new_pages_count = 0
    added_files = []

    for file in files:
        # Save file to disk
        safe_filename = os.path.basename(file.filename or "upload")
        destination = os.path.join(UPLOAD_DIR, safe_filename)
        
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse file
        pages = load_file_auto(destination, original_filename=safe_filename)
        state.loaded_pages.extend(pages)
        new_pages_count += len(pages)
        added_files.append(safe_filename)

    doc_names = list(set(p.filename for p in state.loaded_pages))
    total_pages = sum(len(set(p.page_number for p in state.loaded_pages if p.filename == name)) for name in doc_names)

    return {
        "success": True,
        "message": f"Successfully loaded {len(added_files)} file(s) with {new_pages_count} pages.",
        "added_files": added_files,
        "total_documents": len(doc_names),
        "total_pages": total_pages
    }


@app.post("/api/load-folder-path")
async def load_folder_path(req: FolderPathRequest):
    """Loads all documents from a local folder directory."""
    path = os.path.expanduser(req.folder_path.strip().strip('"').strip("'"))
    if not os.path.exists(path) or not os.path.isdir(path):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {path}")

    pages = load_folder(path, base_display_name=os.path.basename(path) or "Folder")
    if not pages:
        return {
            "success": False,
            "message": f"No supported documents found in directory: {path}",
            "total_documents": 0,
            "total_pages": 0
        }

    state.loaded_pages.extend(pages)
    doc_names = list(set(p.filename for p in state.loaded_pages))
    total_pages = sum(len(set(p.page_number for p in state.loaded_pages if p.filename == name)) for name in doc_names)

    return {
        "success": True,
        "message": f"Successfully loaded {len(pages)} pages from folder: {path}",
        "total_documents": len(doc_names),
        "total_pages": total_pages
    }


@app.post("/api/load-url")
async def load_url(req: UrlRequest):
    """WebBaseLoader: fetches webpage text and parses into pages."""
    url = req.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    pages = load_web_url(url)
    state.loaded_pages.extend(pages)

    doc_names = list(set(p.filename for p in state.loaded_pages))
    total_pages = sum(len(set(p.page_number for p in state.loaded_pages if p.filename == name)) for name in doc_names)

    return {
        "success": True,
        "message": f"Successfully loaded web page into {len(pages)} page(s).",
        "url": url,
        "total_documents": len(doc_names),
        "total_pages": total_pages
    }


@app.post("/api/split-chunks")
async def split_chunks(req: SplitRequest):
    """Functionality 1: Splits documents into chunks and computes chunk & page statistics."""
    if not state.loaded_pages:
        raise HTTPException(status_code=400, detail="No documents have been loaded yet! Upload a file, folder, or URL first.")

    res = split_pages_into_chunks(
        pages=state.loaded_pages,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap
    )

    state.chunks = res["chunks"]
    state.last_chunk_stats = res["statistics"]

    # Preview first 100 chunks for UI
    preview = [c.to_dict() for c in state.chunks[:100]]

    return {
        "success": True,
        "statistics": res["statistics"],
        "chunks_preview": preview,
        "total_chunks_available": len(state.chunks)
    }


@app.post("/api/vectorize")
async def vectorize_chunks(req: VectorizeRequest):
    """Functionality 2: Button to convert chunks into vectors and embed all data."""
    if not state.chunks:
        raise HTTPException(
            status_code=400,
            detail="No chunks available to embed! Please run 'Split into Chunks' first."
        )

    res = state.vector_store.embed_and_store(
        chunks=state.chunks,
        provider=req.provider,
        api_key=req.api_key
    )

    return res


@app.post("/api/chat")
async def chat_rag(req: ChatRequest):
    """Functionality 4: Mini chatbot area to ask any question about the uploaded/embedded data."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(
        question=question,
        vector_store=state.vector_store,
        history=req.history,
        llm_provider=req.llm_provider or "auto",
        api_keys=req.api_keys or {},
        top_k=req.top_k or 4
    )
    return result


@app.post("/api/clear")
async def clear_all():
    """Resets all data."""
    state.clear()
    return {"success": True, "message": "All uploaded documents, chunks, and vector embeddings cleared."}


# Serve static frontend files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
