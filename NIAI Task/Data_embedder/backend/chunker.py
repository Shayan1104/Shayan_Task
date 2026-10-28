import re
import uuid
from typing import List, Dict, Any, Optional
from backend.loaders import LoadedPage

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        document_name: str,
        source: str,
        page_number: int,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.content = content.strip()
        self.document_name = document_name
        self.source = source
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.char_count = len(self.content)
        # Approximate tokens as 1 token ~= 4 characters
        self.token_estimate = max(1, round(self.char_count / 4))
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "document_name": self.document_name,
            "source": self.source,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata
        }


class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = max(100, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size - 50))
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s.strip():
                continue
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Hard split
                    for idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[idx:idx + self.chunk_size])

        # Combine small pieces respecting overlap
        combined: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_len += piece_len
            else:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    combined.append(chunk_text)
                    
                    # Compute overlap buffer
                    while current_chunk and current_len > self.chunk_overlap:
                        dropped = current_chunk.pop(0)
                        current_len -= (len(dropped) + len(separator))

                current_chunk.append(piece)
                current_len += len(piece)

        if current_chunk:
            combined.append(separator.join(current_chunk))

        return [c.strip() for c in combined if c.strip()]


def split_pages_into_chunks(
    pages: List[LoadedPage],
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> Dict[str, Any]:
    """
    Splits loaded pages into chunks preserving page numbers and source metadata.
    Returns:
      - chunks: list of DocumentChunk
      - statistics: total_pages, total_chunks, documents_summary, etc.
    """
    chunker = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[DocumentChunk] = []

    doc_pages_map: Dict[str, set] = {}
    doc_chunks_count: Dict[str, int] = {}
    total_chars = 0

    global_chunk_idx = 1

    for page in pages:
        doc_name = page.filename
        if doc_name not in doc_pages_map:
            doc_pages_map[doc_name] = set()
            doc_chunks_count[doc_name] = 0

        doc_pages_map[doc_name].add(page.page_number)

        page_content = page.content.strip()
        if not page_content:
            continue

        raw_chunks = chunker._split_text(page_content, chunker.separators)
        for sub_idx, raw_chunk in enumerate(raw_chunks):
            cid = f"chk_{global_chunk_idx}"
            chunk_obj = DocumentChunk(
                chunk_id=cid,
                content=raw_chunk,
                document_name=doc_name,
                source=page.source,
                page_number=page.page_number,
                chunk_index=sub_idx + 1,
                metadata={
                    **page.metadata,
                    "total_pages": page.total_pages
                }
            )
            chunks.append(chunk_obj)
            doc_chunks_count[doc_name] += 1
            total_chars += chunk_obj.char_count
            global_chunk_idx += 1

    # Aggregate statistics
    total_pages_count = sum(len(pages_set) for pages_set in doc_pages_map.values())
    total_chunks_count = len(chunks)
    avg_chunk_size = round(total_chars / total_chunks_count) if total_chunks_count > 0 else 0

    documents_summary = []
    for doc_name, pset in doc_pages_map.items():
        documents_summary.append({
            "name": doc_name,
            "pages": len(pset),
            "chunks": doc_chunks_count.get(doc_name, 0)
        })

    return {
        "chunks": chunks,
        "statistics": {
            "total_documents": len(doc_pages_map),
            "total_pages": total_pages_count,
            "total_chunks": total_chunks_count,
            "avg_chunk_size": avg_chunk_size,
            "chunk_size_param": chunk_size,
            "chunk_overlap_param": chunk_overlap,
            "documents_summary": documents_summary
        }
    }
