import os
import sys

# Ensure stdout handles UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.loaders import load_folder, load_pdf
from backend.chunker import split_pages_into_chunks
from backend.vector_store import VectorStoreManager
from backend.rag_engine import answer_question

def test():
    print("--- 1. Testing Loaders ---")
    sample_dir = os.path.abspath("sample_docs")
    pages = load_folder(sample_dir)
    print(f"Loaded {len(pages)} total page segments across documents.")
    for p in pages:
        print(f"  • Doc: {p.filename} | Page: {p.page_number}/{p.total_pages} | Chars: {len(p.content)}")

    print("\n--- 2. Testing Chunker ---")
    res = split_pages_into_chunks(pages, chunk_size=600, chunk_overlap=80)
    chunks = res["chunks"]
    stats = res["statistics"]
    print(f"Statistics:")
    print(f"  Total Documents: {stats['total_documents']}")
    print(f"  Total Pages: {stats['total_pages']}")
    print(f"  Total Chunks: {stats['total_chunks']}")
    print(f"  Avg Chunk Size: {stats['avg_chunk_size']} chars")
    assert stats['total_pages'] >= 5, "Total pages should be at least 5"
    assert stats['total_chunks'] > 0, "Chunks should be created"

    print("\n--- 3. Testing Vector Store (SentenceTransformers) ---")
    vstore = VectorStoreManager()
    embed_res = vstore.embed_and_store(chunks, provider="local")
    print("Embedding result:", embed_res)
    assert embed_res["success"], "Vector embedding failed"
    assert len(vstore.items) == len(chunks), "Mismatch in vector count"

    print("\n--- 4. Testing Semantic Search ---")
    query = "What is the hybrid work policy and on-site days?"
    top_matches = vstore.search(query, top_k=2)
    print(f"Query: '{query}'")
    for m in top_matches:
        print(f"  -> Match: [{m['document_name']} P.{m['page_number']}] Score: {m['score']} ({m['relevance_pct']}%)")
        print(f"     Snippet: {m['content'][:120]}...")

    print("\n--- 5. Testing RAG Answer Engine ---")
    ans_res = answer_question(query, vstore, llm_provider="local")
    print(f"RAG Answer:\n{ans_res['answer']}\n")
    print(f"Citations count: {len(ans_res['citations'])}")

    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test()
