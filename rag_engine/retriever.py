"""
rag_engine/retriever.py
-----------------------
Handles semantic similarity search over the FAISS vector store.
Retrieves the most relevant document chunks for a given user query.
"""

import logging
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import FAISS

from config import Config
from rag_engine.ingest import load_vectorstore

import time

def call_llm_with_retry(query_fn, *args, retries=3, delay=5, **kwargs):
    """
    Calls the LLM with automatic retry if rate limit occurs.
    """
    for attempt in range(retries):
        result = query_fn(*args, **kwargs)

        if "Rate limit exceeded" not in str(result):
            return result

        print(f"   ⏳ Rate limit hit. Waiting {delay}s before retry...")
        time.sleep(delay)

    return result

# Module logger
logger = logging.getLogger(__name__)

# Module-level cache for the vector store to avoid reloading on every query
_vectorstore_cache: Optional[FAISS] = None


def get_vectorstore() -> Optional[FAISS]:
    """
    Get the FAISS vector store, using a module-level cache.

    On first call, loads the index from disk. Subsequent calls
    return the cached instance for performance.

    Returns:
        FAISS vector store instance, or None if not available.
    """
    global _vectorstore_cache

    if _vectorstore_cache is None:
        _vectorstore_cache = load_vectorstore()

    return _vectorstore_cache


def refresh_vectorstore() -> Optional[FAISS]:
    """
    Force-reload the FAISS vector store from disk.

    Called after new documents are ingested to pick up
    the updated index.

    Returns:
        Fresh FAISS vector store instance.
    """
    global _vectorstore_cache
    _vectorstore_cache = None  # Clear cache
    return get_vectorstore()


def retrieve_relevant_chunks(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically similar document chunks for a query.

    Performs similarity search with relevance scores using the FAISS index.
    Returns structured results including the text content, metadata, and
    similarity score for each retrieved chunk.

    Args:
        query: The user's question string.
        top_k: Number of chunks to retrieve (defaults to Config.TOP_K_CHUNKS).

    Returns:
        List of dicts, each containing:
          - content: The chunk text
          - source: Source PDF filename
          - page: Page number (1-indexed for display)
          - score: Similarity score (lower = more similar for L2)
          - snippet: First 200 chars of the chunk for citation display

        Returns empty list if vector store is unavailable.
    """
    if top_k is None:
        top_k = Config.TOP_K_CHUNKS

    vectorstore = get_vectorstore()

    if vectorstore is None:
        logger.error("Vector store is not available. Have documents been uploaded?")
        return []

    logger.info(f"Retrieving top {top_k} chunks for query: '{query[:80]}...'")

    # similarity_search_with_score returns (Document, score) tuples
    # For FAISS L2 distance: lower score = more similar
    # Fetch more candidates first, then let MMR choose diverse results
    fetch_k = top_k * 4
  # Step 1 — get diverse documents using MMR (no scores available)
    mmr_docs = vectorstore.max_marginal_relevance_search(
        query=query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=0.7,    
    )

# Step 2 — compute similarity scores separately
    results_with_scores = vectorstore.similarity_search_with_score(
        query=query,
        k=top_k * 2,   # fetch a few more to match with MMR docs
    )

# Step 3 — map doc -> score
    doc_score_map = {doc.page_content: score for doc, score in results_with_scores}

# Step 4 — combine MMR docs with their scores
    results_with_scores = []
    for doc in mmr_docs:
        score = doc_score_map.get(doc.page_content, 1.0)  # fallback weak score
        results_with_scores.append((doc, score))

    retrieved_chunks = []

    for doc, score in results_with_scores:
        chunk_info = {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,  # Convert 0-indexed → 1-indexed
            "score": round(float(score), 4),
            "snippet": doc.page_content[:200].strip() + "..."
                       if len(doc.page_content) > 200
                       else doc.page_content.strip(),
        }
        retrieved_chunks.append(chunk_info)

        logger.debug(
            f"  Chunk from {chunk_info['source']} p.{chunk_info['page']} "
            f"(score: {chunk_info['score']}): {chunk_info['snippet'][:60]}..."
        )

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks.")
    return retrieved_chunks


def format_context_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Each chunk is labeled with its source and page number so the LLM
    can reference specific locations in its answer.

    Args:
        chunks: List of chunk dicts from retrieve_relevant_chunks().

    Returns:
        Formatted context string ready for prompt injection.
    """
    if not chunks:
        return "No relevant context found in the documents."

    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']}]\n"
            f"{chunk['content']}\n"
        )

    return "\n---\n".join(context_parts)