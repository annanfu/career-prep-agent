"""Semantic retriever with cosine-similarity deduplication over ChromaDB."""

from pathlib import Path
from typing import List

import numpy as np
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "chroma_data"
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dedup thresholds (per SPEC.md)
DEDUP_DROP_THRESHOLD = 0.9    # keep only higher-scoring chunk
DEDUP_RELATED_THRESHOLD = 0.7  # keep both, tag as related


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _deduplicate(
    results: List[dict],
    embeddings_model: HuggingFaceEmbeddings,
) -> List[dict]:
    """Remove near-duplicate chunks using pairwise cosine similarity.

    Rules (per SPEC.md):
    - similarity > 0.9: keep only the chunk with higher relevance score.
    - 0.7–0.9: keep both, add 'related_chunks' field listing the other's
      content.

    Args:
        results: List of result dicts with 'content', 'score', and 'embedding'
            keys.
        embeddings_model: Loaded HuggingFaceEmbeddings for re-encoding if
            needed.

    Returns:
        Deduplicated list of result dicts (without 'embedding' key).
    """
    if len(results) <= 1:
        for r in results:
            r.pop("embedding", None)
        return results

    n = len(results)
    vecs = np.array([r["embedding"] for r in results])

    # Pairwise similarity matrix
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = vecs / norms
    sim_matrix = normed @ normed.T

    dropped = set()
    related: dict[int, List[str]] = {i: [] for i in range(n)}

    for i in range(n):
        if i in dropped:
            continue
        for j in range(i + 1, n):
            if j in dropped:
                continue
            sim = sim_matrix[i, j]
            if sim > DEDUP_DROP_THRESHOLD:
                # Drop the lower-scoring chunk
                if results[i]["score"] >= results[j]["score"]:
                    dropped.add(j)
                else:
                    dropped.add(i)
                    break  # i is dropped; move on
            elif sim >= DEDUP_RELATED_THRESHOLD:
                related[i].append(results[j]["content"])
                related[j].append(results[i]["content"])

    deduped = []
    for i, r in enumerate(results):
        if i in dropped:
            continue
        r.pop("embedding", None)
        r["related_chunks"] = related[i]
        deduped.append(r)

    return deduped


def retrieve_experiences(
    queries: List[str],
    top_k: int = 5,
) -> List[dict]:
    """Search ChromaDB for relevant experience chunks, then deduplicate.

    For each query, retrieves top_k results from the 'knowledge_base'
    ChromaDB collection. After collecting all results across queries,
    applies cosine-similarity deduplication:
    - Chunks with similarity > 0.9: only the higher-scored one is kept.
    - Chunks with similarity 0.7–0.9: both are kept but cross-reference
      each other in a 'related_chunks' field.

    Args:
        queries: List of search query strings.
        top_k: Number of results to retrieve per query.

    Returns:
        List of dicts, each with:
            - content (str): chunk text
            - source_doc (str): originating filename
            - score (float): relevance score from ChromaDB (lower = closer)
            - related_chunks (list[str]): content of related (0.7–0.9) chunks
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            f"ChromaDB not found at '{CHROMA_DIR}'. "
            "Run src.rag.ingest first."
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    seen_contents: set[str] = set()
    raw_results: List[dict] = []

    for query in queries:
        hits = vectorstore.similarity_search_with_score(query, k=top_k)
        for doc, score in hits:
            content = doc.page_content.strip()
            if content in seen_contents:
                continue
            seen_contents.add(content)

            # Embed the chunk to compute pairwise similarity later
            embedding = embeddings.embed_query(content)

            raw_results.append({
                "content": content,
                "source_doc": doc.metadata.get("source_file", "unknown"),
                "score": score,
                "embedding": embedding,
            })

    return _deduplicate(raw_results, embeddings)


if __name__ == "__main__":
    test_queries = [
        "Python ETL pipeline",
        "distributed systems experience",
        "machine learning model training",
        "SQL database design",
    ]

    print("Running retriever test...\n")
    results = retrieve_experiences(test_queries, top_k=5)

    print(f"Retrieved {len(results)} deduplicated chunks:\n")
    for i, r in enumerate(results, 1):
        related_count = len(r["related_chunks"])
        print(f"[{i}] Source: {r['source_doc']} | Score: {r['score']:.4f}"
              f" | Related: {related_count}")
        print(f"     {r['content'][:120].replace(chr(10), ' ')}...")
        print()
