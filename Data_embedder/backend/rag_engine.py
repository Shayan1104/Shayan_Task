import re
from typing import List, Dict, Any, Optional
from backend.vector_store import VectorStoreManager

SYSTEM_PROMPT = """You are an intelligent, precise AI assistant analyzing uploaded documents.
Answer the user's question using ONLY the provided document context.
Follow these guidelines:
1. Always base your response directly on the provided context snippets.
2. Cite the source document name and page number when stating facts (e.g. `[source: filename.pdf, Page 2]`).
3. If the context does not contain enough information to answer the question, clearly state what is known and what is missing.
4. Keep answers organized, using bullet points, bold highlights, or tables where appropriate."""


def answer_question(
    question: str,
    vector_store: VectorStoreManager,
    history: Optional[List[Dict[str, str]]] = None,
    llm_provider: str = "auto",
    api_keys: Optional[Dict[str, str]] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """Retrieves relevant chunks and generates an answer using the chosen or available LLM."""
    history = history or []
    api_keys = api_keys or {}

    # Check if vector store has items
    if len(vector_store.items) == 0:
        return {
            "answer": "⚠️ **No data has been vectorized yet!** Please upload documents, split them into chunks, and click **'Convert Chunks into Vectors'** first.",
            "citations": [],
            "model_used": "None"
        }

    # 1. Retrieve top-k chunks
    retrieved = vector_store.search(query=question, top_k=top_k)
    if not retrieved:
        return {
            "answer": "I couldn't find any relevant chunks matching your question in the uploaded documents.",
            "citations": [],
            "model_used": "None"
        }

    # Format context block
    context_blocks = []
    citations = []
    for idx, item in enumerate(retrieved):
        doc = item["document_name"]
        page = item["page_number"]
        chunk_id = item["chunk_id"]
        score = item["score"]
        pct = item["relevance_pct"]
        content = item["content"]
        
        context_blocks.append(f"--- Context Snippet {idx+1} [Source: {doc} | Page: {page} | Chunk: {chunk_id}] ---\n{content}\n")
        citations.append({
            "citation_id": idx + 1,
            "document_name": doc,
            "page_number": page,
            "chunk_id": chunk_id,
            "score": score,
            "relevance_pct": pct,
            "snippet": content[:220] + ("..." if len(content) > 220 else "")
        })

    full_context = "\n".join(context_blocks)

    # 2. Determine LLM provider
    provider = (llm_provider or "auto").lower()
    gemini_key = api_keys.get("gemini", "").strip()
    openai_key = api_keys.get("openai", "").strip()
    groq_key = api_keys.get("groq", "").strip()

    # Route based on explicit selection or available keys
    if (provider == "gemini" or (provider == "auto" and gemini_key)) and gemini_key:
        return _generate_gemini(question, full_context, history, gemini_key, citations)
    elif (provider == "openai" or (provider == "auto" and openai_key)) and openai_key:
        return _generate_openai(question, full_context, history, openai_key, citations)
    elif (provider == "groq" or (provider == "auto" and groq_key)) and groq_key:
        return _generate_groq(question, full_context, history, groq_key, citations)
    else:
        # Fallback to local offline synthesis
        return _generate_local_synthesis(question, retrieved, citations)


def _generate_gemini(
    question: str,
    context: str,
    history: List[Dict[str, str]],
    api_key: str,
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try gemini-2.5-flash or gemini-1.5-flash
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        
        # Build prompt
        prompt = f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nPlease answer clearly based on the document context."
        response = model.generate_content(prompt)
        answer_text = response.text if response and response.text else "No response generated."
        
        return {
            "answer": answer_text,
            "citations": citations,
            "model_used": "Google Gemini 1.5 Flash"
        }
    except Exception as e:
        print(f"Gemini error: {e}")
        # Graceful fallback to local synthesis with error note
        fallback = _generate_local_synthesis(question, citations, citations)
        fallback["answer"] = f"> ⚠️ *Gemini API call failed ({str(e)}). Falling back to local semantic synthesis:*\n\n" + fallback["answer"]
        return fallback


def _generate_openai(
    question: str,
    context: str,
    history: List[Dict[str, str]],
    api_key: str,
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Add recent history
        for msg in history[-4:]:
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            
        messages.append({
            "role": "user",
            "content": f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"
        })
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "citations": citations,
            "model_used": "OpenAI GPT-4o-mini"
        }
    except Exception as e:
        print(f"OpenAI error: {e}")
        fallback = _generate_local_synthesis(question, citations, citations)
        fallback["answer"] = f"> ⚠️ *OpenAI API call failed ({str(e)}). Falling back to local semantic synthesis:*\n\n" + fallback["answer"]
        return fallback


def _generate_groq(
    question: str,
    context: str,
    history: List[Dict[str, str]],
    api_key: str,
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\nQUESTION:\n{question}"}
        ]
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2
        )
        return {
            "answer": response.choices[0].message.content,
            "citations": citations,
            "model_used": "Groq (Llama-3.3-70b)"
        }
    except Exception as e:
        fallback = _generate_local_synthesis(question, citations, citations)
        fallback["answer"] = f"> ⚠️ *Groq API call failed ({str(e)}). Falling back to local semantic synthesis:*\n\n" + fallback["answer"]
        return fallback


def _generate_local_synthesis(
    question: str,
    retrieved_items: List[Dict[str, Any]],
    citations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Smart offline extraction & synthesis engine when no external API key is configured."""
    q_words = set(re.findall(r'\w+', question.lower()))
    
    top_excerpts = []
    for item in retrieved_items:
        content = item.get("content", item.get("snippet", ""))
        doc = item.get("document_name", "Document")
        page = item.get("page_number", 1)
        rel = item.get("relevance_pct", 85)
        
        # Split into sentences and find highest keyword overlap
        sentences = re.split(r'(?<=[.!?])\s+', content)
        scored_sentences = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            s_words = set(re.findall(r'\w+', s_clean.lower()))
            overlap = len(q_words.intersection(s_words))
            scored_sentences.append((overlap, s_clean))
            
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentences = [s for score, s in scored_sentences[:2] if score > 0]
        
        excerpt = " ".join(best_sentences) if best_sentences else (content[:250] + "...")
        top_excerpts.append((doc, page, rel, excerpt))

    # Construct synthesized answer
    lines = [
        f"### 📄 Findings from Uploaded Documents ({top_excerpts[0][2]}% Top Match)\n",
        f"Based on semantic retrieval from your uploaded data for: **\"{question}\"**:\n"
    ]

    for doc, page, rel, excerpt in top_excerpts:
        lines.append(f"- **From `{doc}` (Page {page})** — *{rel}% Match*:\n  > \"{excerpt}\"\n")

    lines.append(
        "\n*(💡 Tip: Enter a free Google Gemini, OpenAI, or Groq API key in the top **Settings** bar for fully conversational, synthesized prose!)*"
    )

    return {
        "answer": "\n".join(lines),
        "citations": citations,
        "model_used": "Local Semantic Extractor (Offline)"
    }
