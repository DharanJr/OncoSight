"""
Module 6 — RAG chatbot. Retrieves relevant chunks from the knowledge base
built by build_knowledge_base.py, then asks the local LLM (Ollama) to
answer USING ONLY that retrieved context — this is what keeps answers
grounded in WHO/NCI/CDC/ACS sources rather than the LLM's own (unverified,
possibly wrong) medical knowledge.

Anti-hallucination design:
  - If no retrieved chunk clears RAG_RELEVANCE_THRESHOLD, the question is
    treated as out-of-scope and answered with a polite decline — NOT passed
    to the LLM to improvise an answer.
  - The prompt explicitly instructs the LLM to answer only from the
    provided context and say so if the context doesn't cover it.
  - Every answer cites which source document(s) it drew from.

Usage:
    python -m src.rag.chatbot "What are the symptoms of lung cancer?"
    python -m src.rag.chatbot   (interactive mode if no question given)
"""

import sys
import re
import pickle
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    RAG_INDEX_DIR,
    RAG_TOP_K,
    RAG_RELEVANCE_THRESHOLD,
    RAG_TOPIC_KEYWORDS,
    RAG_SOURCE_DISPLAY_NAMES,
)
from src.reports.llm_backend import call_ollama, OllamaUnavailableError

RAG_PROMPT_TEMPLATE = """You are a lung cancer information assistant. Answer the question using ONLY the context below. If the context does not fully answer the question, say what it does cover and note the gap — do not add information from outside the context.

Context:
{context}

Question: {question}

Answer concisely, in plain language:"""


def load_index():
    backend_path = RAG_INDEX_DIR / "backend.txt"
    if not backend_path.exists():
        raise FileNotFoundError(
            f"No index found in {RAG_INDEX_DIR}. Run "
            "`python -m src.rag.build_knowledge_base` first."
        )
    backend = backend_path.read_text().strip()

    with open(RAG_INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    if backend == "embeddings":
        import faiss
        from sentence_transformers import SentenceTransformer
        from src.config import EMBEDDING_MODEL_NAME

        index = faiss.read_index(str(RAG_INDEX_DIR / "faiss.index"))
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return {"backend": "embeddings", "index": index, "model": model, "chunks": chunks}
    else:
        with open(RAG_INDEX_DIR / "tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        with open(RAG_INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
            matrix = pickle.load(f)
        return {"backend": "tfidf", "vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}


def is_on_topic(query: str) -> bool:
    """
    Independent guard alongside the similarity threshold: the query must
    contain at least one lung-cancer-related keyword. Catches cases where
    an off-topic question accidentally scores above the similarity
    threshold on incidental word overlap.
    """
    query_words = set(re.findall(r"[a-z]+", query.lower()))
    return len(query_words & RAG_TOPIC_KEYWORDS) > 0


def resolve_source_name(filename: str) -> str:
    """Maps a raw PDF filename to a proper organization display name for
    citations, e.g. 'who-fact-sheet.pdf' -> 'World Health Organization (WHO)'.
    Falls back to the raw filename if no known organization keyword matches."""
    lower_name = filename.lower()
    for keyword, display_name in RAG_SOURCE_DISPLAY_NAMES.items():
        if keyword in lower_name:
            return display_name
    return filename


def retrieve(query: str, index_data, top_k: int = RAG_TOP_K):
    """Returns list of (chunk_dict, similarity_score), sorted best first."""
    if index_data["backend"] == "embeddings":
        import faiss
        query_emb = index_data["model"].encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_emb)
        scores, indices = index_data["index"].search(query_emb, top_k)
        results = [
            (index_data["chunks"][idx], float(score))
            for idx, score in zip(indices[0], scores[0]) if idx != -1
        ]
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = index_data["vectorizer"].transform([query])
        sims = cosine_similarity(query_vec, index_data["matrix"])[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = [(index_data["chunks"][i], float(sims[i])) for i in top_indices]

    return results


def answer_question(query: str, index_data):
    # Guard 1: keyword topic gate — reject obviously off-topic questions
    # before even running retrieval, no wasted compute either.
    if not is_on_topic(query):
        return (
            "I can only answer questions about lung cancer — symptoms, diagnosis, "
            "staging, treatment, screening, and prevention. That question looks "
            "outside my scope; try rephrasing around one of those topics.",
            [],
        )

    results = retrieve(query, index_data)

    # Guard 2: similarity threshold — even for an on-topic-sounding question,
    # only answer if the knowledge base actually has relevant content.
    relevant = [(chunk, score) for chunk, score in results if score >= RAG_RELEVANCE_THRESHOLD]

    if not relevant:
        return (
            "That's a lung cancer-related question, but I don't have specific "
            "information on it in my current knowledge base. I can speak to "
            "symptoms, diagnosis, staging, treatment, screening, and prevention "
            "based on the sources I've been given — try asking about one of those.",
            [],
        )

    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c, _ in relevant)
    sources = sorted(set(resolve_source_name(c["source"]) for c, _ in relevant))

    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

    try:
        answer = call_ollama(prompt, max_tokens=400)
    except OllamaUnavailableError as e:
        answer = (
            f"[Local LLM unavailable: {e}]\n\n"
            "Here is the most relevant retrieved passage instead:\n\n"
            f"{relevant[0][0]['text']}"
        )

    # Merge citations into the answer text as one clean artifact — easier to
    # display as-is in the dashboard/report later, rather than requiring the
    # caller to remember to also render the separate `sources` list.
    answer_with_citations = answer + "\n\nSources: " + ", ".join(sources)

    return answer_with_citations, sources


def main():
    print("Loading knowledge base index...")
    index_data = load_index()
    print(f"Loaded {len(index_data['chunks'])} chunks (backend: {index_data['backend']})\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        answer, sources = answer_question(query, index_data)
        print(f"Q: {query}\n")
        print(f"A: {answer}\n")
    else:
        print("Interactive mode. Type a question, or 'exit' to quit.\n")
        while True:
            query = input("Q: ").strip()
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue
            answer, sources = answer_question(query, index_data)
            print(f"\nA: {answer}\n")


if __name__ == "__main__":
    main()