"""
rag_engine/ingest.py
--------------------
Document ingestion pipeline:
  1. Load PDFs using PyPDFLoader
  2. Clean extracted text
  3. Split into overlapping chunks with metadata
  4. Generate embeddings with sentence-transformers
  5. Build and persist a FAISS vector store
"""

import os
import re
import logging
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from config import Config

# Configure module logger
logger = logging.getLogger(__name__)


def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """
    Load one or more PDF files and return a flat list of LangChain Documents.

    Each Document corresponds to one page and carries metadata:
      - source: original filename
      - page: zero-indexed page number

    Args:
        pdf_paths: List of absolute file paths to PDF files.

    Returns:
        List of Document objects with page_content and metadata.
    """
    all_documents = []

    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            logger.warning(f"PDF not found, skipping: {pdf_path}")
            continue

        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Enrich metadata with just the filename (not full path)
        filename = os.path.basename(pdf_path)
        for page_doc in pages:
            page_doc.metadata["source"] = filename
            # PyPDFLoader sets 'page' automatically (0-indexed)

        all_documents.extend(pages)
        logger.info(f"  → Loaded {len(pages)} pages from {filename}")

    logger.info(f"Total pages loaded across all PDFs: {len(all_documents)}")
    return all_documents


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from PDF pages.

    Operations:
      - Replace multiple whitespace/newlines with single space
      - Strip leading/trailing whitespace
      - Remove non-printable characters

    Args:
        text: Raw text string from a PDF page.

    Returns:
        Cleaned text string.
    """
    # Replace multiple whitespace (including newlines, tabs) with single space
    text = re.sub(r"\s+", " ", text)

    # Remove non-printable characters but keep basic punctuation
    text = re.sub(r"[^\x20-\x7E\n]", "", text)

    return text.strip()


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller, overlapping chunks for embedding.

    Uses RecursiveCharacterTextSplitter which tries to split on
    paragraph → sentence → word boundaries to keep chunks semantically coherent.

    Each resulting chunk inherits the metadata (source, page) from its parent.

    Args:
        documents: List of full-page Documents.

    Returns:
        List of chunked Documents with preserved metadata.
    """
    # Clean text in each document before splitting
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Try natural boundaries first
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Filter out chunks that are too short to be meaningful
    meaningful_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 30]

    logger.info(
        f"Split {len(documents)} pages into {len(meaningful_chunks)} chunks "
        f"(chunk_size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})"
    )

    return meaningful_chunks


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the sentence-transformer embedding model.

    Uses all-MiniLM-L6-v2 which produces 384-dimensional embeddings,
    offering a good balance between quality and speed.

    Returns:
        HuggingFaceEmbeddings instance ready for encoding.
    """
    logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},       # Use CPU for broad compatibility
        encode_kwargs={"normalize_embeddings": True},  # L2 normalize for cosine sim
    )

    return embeddings


def build_vectorstore(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Build a FAISS vector store from document chunks.

    Creates embeddings for all chunks and indexes them in a FAISS
    flat L2 index for exact nearest-neighbor search.

    Args:
        chunks: List of chunked Documents.
        embeddings: Initialized embedding model.

    Returns:
        FAISS vector store instance.
    """
    logger.info(f"Building FAISS index from {len(chunks)} chunks...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    logger.info("FAISS index built successfully.")
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Persist the FAISS vector store to disk.

    Saves both the FAISS index and the document store so the index
    can be reloaded without re-processing documents.

    Args:
        vectorstore: FAISS vector store to save.
    """
    os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(Config.FAISS_INDEX_PATH)
    logger.info(f"FAISS index saved to: {Config.FAISS_INDEX_PATH}")


def load_vectorstore() -> FAISS:
    """
    Load a previously persisted FAISS vector store from disk.

    Returns:
        FAISS vector store instance, or None if index doesn't exist.

    Raises:
        FileNotFoundError: If the index directory doesn't exist.
    """
    if not os.path.exists(Config.FAISS_INDEX_PATH):
        logger.warning(f"No FAISS index found at {Config.FAISS_INDEX_PATH}")
        return None

    logger.info(f"Loading FAISS index from: {Config.FAISS_INDEX_PATH}")
    embeddings = get_embedding_model()

    vectorstore = FAISS.load_local(
        Config.FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,  # Required for loading pickled data
    )

    logger.info("FAISS index loaded successfully.")
    return vectorstore


def ingest_documents(pdf_paths: List[str]) -> FAISS:
    """
    Complete ingestion pipeline: Load → Clean → Chunk → Embed → Index → Save.

    This is the main entry point for document processing. It takes a list
    of PDF file paths and produces a persisted FAISS vector store.

    Args:
        pdf_paths: List of absolute paths to PDF files to process.

    Returns:
        FAISS vector store populated with document chunks.

    Raises:
        ValueError: If no valid documents are loaded.
    """
    logger.info("=" * 60)
    logger.info("STARTING DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load PDFs
    documents = load_pdfs(pdf_paths)
    if not documents:
        raise ValueError("No documents were loaded. Check your PDF files.")

    # Step 2: Split into chunks (cleaning happens inside)
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No meaningful chunks were created from the documents.")

    # Step 3: Create embedding model
    embeddings = get_embedding_model()

    # Step 4: Build FAISS index
    vectorstore = build_vectorstore(chunks, embeddings)

    # Step 5: Save to disk
    save_vectorstore(vectorstore)

    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE COMPLETE")
    logger.info(f"  Total chunks indexed: {len(chunks)}")
    logger.info("=" * 60)

    return vectorstore