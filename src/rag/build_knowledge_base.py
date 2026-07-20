"""
Module 6 — builds the RAG knowledge base from PDFs and/or .txt files in
knowledge_base/.

Pipeline: PDF/.txt -> extracted text -> overlapping chunks -> embeddings -> index.

.txt files are supported alongside .pdf because some official source PDFs
(WHO/NCI/CDC/ACS) are scanned/image-based with no extractable text layer —
saving that page's content as .txt instead avoids needing OCR, while the
source remains the same official organization either way.

Uses sentence-transformers (all-MiniLM-L6-v2, per project spec) + FAISS for
real semantic search when both are available. Both packages include
compiled binaries that CAN hit the same Windows Application Control DLL
blocking issue seen earlier with numba/shap on this machine — if that
happens, this script automatically falls back to a pure-Python TF-IDF +
cosine-similarity retriever (scikit-learn only, zero compiled-binary risk).
Either path produces the same downstream interface, so chatbot.py doesn't
need to know or care which one built the index.

Usage:
    python -m src.rag.build_knowledge_base
"""

import sys
import pickle
from pathlib import Path

import numpy as np
from pypdf import PdfReader

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    KNOWLEDGE_BASE_DIR,
    RAG_INDEX_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
)

# Try the real embedding + FAISS path; fall back to TF-IDF if unavailable
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    BACKEND = "embeddings"
except ImportError as e:
    print(f"[INFO] sentence-transformers/faiss not available ({e})")
    print("[INFO] Falling back to TF-IDF retrieval (pure Python, no compiled binaries).")
    from sklearn.feature_extraction.text import TfidfVectorizer
    BACKEND = "tfidf"


def extract_text_from_pdfs():
    """
    Loads documents from knowledge_base/, both .pdf and .txt.

    .txt support exists because some official PDFs (WHO/NCI/CDC/ACS) are
    scanned/image-based with no extractable text layer — rather than add
    OCR (out of scope for this project), save that page's content directly
    as a .txt file (e.g. copy-paste from the webpage, or browser "Save as
    text"). The source is still the same official organization either way
    — only the storage format differs. Mixed .pdf/.txt in the same folder
    is fully supported; both feed the same chunking/indexing pipeline.
    """
    if not KNOWLEDGE_BASE_DIR.exists():
        raise FileNotFoundError(f"{KNOWLEDGE_BASE_DIR} does not exist.")

    pdf_paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf"))
    txt_paths = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))

    if not pdf_paths and not txt_paths:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {KNOWLEDGE_BASE_DIR}. Download "
            "WHO/NCI/CDC/ACS lung cancer content and place it there (PDF, or "
            "save the page text as .txt if the PDF is scanned/image-based) — "
            "see project README."
        )

    documents = []  # list of (source_filename, full_text)

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                documents.append((pdf_path.name, text))
                print(f"  Extracted {len(text)} chars from {pdf_path.name}")
            else:
                print(
                    f"  [WARN] No extractable text in {pdf_path.name} "
                    "(likely scanned/image-based) — save its content as a "
                    ".txt file instead, see this function's docstring."
                )
        except Exception as e:
            print(f"  [WARN] Failed to read {pdf_path.name}: {e}")

    for txt_path in txt_paths:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append((txt_path.name, text))
                print(f"  Loaded {len(text)} chars from {txt_path.name}")
            else:
                print(f"  [WARN] {txt_path.name} is empty — skipping.")
        except Exception as e:
            print(f"  [WARN] Failed to read {txt_path.name}: {e}")

    return documents


def chunk_text(text: str, source: str):
    """Overlapping character-window chunking — simple and dependency-free."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + RAG_CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # skip near-empty trailing fragments
            chunks.append({"text": chunk, "source": source})
        start += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP
    return chunks


def build_index_embeddings(all_chunks):
    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)  # so inner product == cosine similarity

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(RAG_INDEX_DIR / "faiss.index"))
    with open(RAG_INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(RAG_INDEX_DIR / "backend.txt", "w") as f:
        f.write("embeddings")

    print(f"Saved FAISS index + {len(all_chunks)} chunks to: {RAG_INDEX_DIR}")


def build_index_tfidf(all_chunks):
    print(f"\nBuilding TF-IDF index for {len(all_chunks)} chunks...")
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(texts)

    with open(RAG_INDEX_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(RAG_INDEX_DIR / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(matrix, f)
    with open(RAG_INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(RAG_INDEX_DIR / "backend.txt", "w") as f:
        f.write("tfidf")

    print(f"Saved TF-IDF index + {len(all_chunks)} chunks to: {RAG_INDEX_DIR}")


def main():
    print("Loading documents from knowledge_base/ (.pdf and .txt)...")
    documents = extract_text_from_pdfs()

    all_chunks = []
    for source, text in documents:
        chunks = chunk_text(text, source)
        all_chunks.extend(chunks)
        print(f"  {source}: {len(chunks)} chunks")

    if not all_chunks:
        raise ValueError("No chunks produced — check that PDFs have extractable text.")

    if BACKEND == "embeddings":
        build_index_embeddings(all_chunks)
    else:
        build_index_tfidf(all_chunks)

    print(f"\nDone. Backend used: {BACKEND}. Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    main()