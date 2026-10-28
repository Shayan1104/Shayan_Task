# AI Vector Architecture & RAG Pipeline Specification

## 1. Overview
This technical specification outlines the end-to-end vector embedding and retrieval architecture for the NIAI Task Data Embedder.

## 2. Ingestion & Document Parsing
The system accepts raw documents in various formats:
- Adobe Portable Document Format (.pdf) with page-level text extraction
- Plain text (.txt), Markdown (.md), CSV, and JSON data
- Microsoft Word documents (.docx)
- Web URLs parsed via WebBaseLoader / BeautifulSoup
- Recursive folder trees and compressed ZIP archives

## 3. Semantic Chunking Strategy
Documents are partitioned using RecursiveCharacterChunker:
- Target Chunk Size: 800 characters
- Chunk Overlap: 100 characters
- Separator Hierarchy: `["\n\n", "\n", ". ", "? ", "! ", " "]`
- Metadata: Each chunk preserves the source filename, exact page number (1-indexed), chunk index, and estimated token counts.

## 4. Vector Embedding Models
Vector representations are computed using:
- Local CPU Mode: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
- Optional Cloud Mode: OpenAI `text-embedding-3-small` (1536-d) or Google Gemini `text-embedding-004` (768-d).
- Distance Metric: Cosine similarity via normalized dot products.
