import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from backend.chunker import DocumentChunk

class VectorItem:
    def __init__(self, chunk_id: str, embedding: np.ndarray, metadata: Dict[str, Any], text: str):
        self.chunk_id = chunk_id
        self.embedding = embedding  # normalized 1D float vector
        self.metadata = metadata
        self.text = text


class VectorStoreManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.default_model_name = model_name
        self._local_model = None
        self.items: List[VectorItem] = []
        self.dimension: Optional[int] = None
        self.last_indexed_time: Optional[float] = None
        self.active_provider: str = "Local (SentenceTransformers)"

    def get_local_model(self):
        """Lazy load sentence-transformers model to ensure fast app startup."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {self.default_model_name}...")
            self._local_model = SentenceTransformer(self.default_model_name)
        return self._local_model

    def embed_texts_local(self, texts: List[str]) -> np.ndarray:
        model = self.get_local_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def embed_texts_openai(self, texts: List[str], api_key: str) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        vectors = [item.embedding for item in response.data]
        vectors = np.array(vectors, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def embed_texts_gemini(self, texts: List[str], api_key: str) -> np.ndarray:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        vectors = []
        for text in texts:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            vectors.append(result['embedding'])
        vectors = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def embed_and_store(
        self,
        chunks: List[DocumentChunk],
        provider: str = "local",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Embeds all chunks into vector store.
        provider: 'local', 'openai', or 'gemini'
        """
        if not chunks:
            return {
                "success": False,
                "message": "No chunks provided to embed.",
                "total_embedded": 0
            }

        start_time = time.time()
        texts = [c.content for c in chunks]
        
        provider_clean = (provider or "local").lower()
        if provider_clean == "openai" and api_key:
            embeddings = self.embed_texts_openai(texts, api_key)
            self.active_provider = "OpenAI (text-embedding-3-small)"
        elif provider_clean == "gemini" and api_key:
            embeddings = self.embed_texts_gemini(texts, api_key)
            self.active_provider = "Google Gemini (text-embedding-004)"
        else:
            embeddings = self.embed_texts_local(texts)
            self.active_provider = f"Local ({self.default_model_name})"

        # Populate store
        self.items = []
        for idx, chunk in enumerate(chunks):
            vec = embeddings[idx]
            meta = {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "token_estimate": chunk.token_estimate
            }
            self.items.append(VectorItem(
                chunk_id=chunk.chunk_id,
                embedding=vec,
                metadata=meta,
                text=chunk.content
            ))

        self.dimension = int(embeddings.shape[1])
        elapsed = round(time.time() - start_time, 2)
        self.last_indexed_time = elapsed

        return {
            "success": True,
            "message": f"Successfully embedded {len(self.items)} chunks into vectors.",
            "total_embedded": len(self.items),
            "dimension": self.dimension,
            "provider": self.active_provider,
            "time_elapsed_seconds": elapsed
        }

    def search(
        self,
        query: str,
        top_k: int = 4,
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Finds top-k most similar chunks for the query string."""
        if not self.items:
            return []

        # Embed query
        provider_clean = (provider or "").lower()
        if "openai" in self.active_provider.lower() and api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            res = client.embeddings.create(input=[query], model="text-embedding-3-small")
            q_vec = np.array(res.data[0].embedding, dtype=np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
        elif "gemini" in self.active_provider.lower() and api_key:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            res = genai.embed_content(
                model="models/text-embedding-004",
                content=query,
                task_type="retrieval_query"
            )
            q_vec = np.array(res['embedding'], dtype=np.float32)
            q_vec = q_vec / (np.linalg.norm(q_vec) or 1)
        else:
            model = self.get_local_model()
            q_vec = model.encode([query], normalize_embeddings=True)[0]

        # Cosine similarity matrix multiplication
        matrix = np.array([item.embedding for item in self.items])
        scores = np.dot(matrix, q_vec)

        # Top k indices
        k = min(top_k, len(self.items))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            item = self.items[idx]
            score = float(scores[idx])
            results.append({
                "chunk_id": item.chunk_id,
                "content": item.text,
                "document_name": item.metadata["document_name"],
                "page_number": item.metadata["page_number"],
                "chunk_index": item.metadata["chunk_index"],
                "score": round(score, 4),
                "relevance_pct": max(0, min(100, round((score + 1) / 2 * 100, 1))) if score <= 1 else 100,
                "metadata": item.metadata
            })

        return results

    def clear(self):
        """Clears all vectors."""
        self.items = []
        self.dimension = None
        self.last_indexed_time = None
