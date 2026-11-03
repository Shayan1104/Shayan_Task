import uvicorn
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print("=" * 60)
    print(" * Launching DataEmbedder AI Web Application...")
    print(" * Local URL: http://127.0.0.1:8080")
    print(" * Document Ingestion: Folders, PDFs, TXT, DOCX, Web URLs")
    print(" * Features: Chunk & Page Counter, Vectorizer, RAG Chatbot")
    print("=" * 60)
    
    # Ensure project root is in sys.path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("backend.app:app", host="127.0.0.1", port=port, reload=False)
