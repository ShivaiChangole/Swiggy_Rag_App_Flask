"""
config.py
---------
Centralized configuration for the RAG application.
Loads environment variables and defines all constants used across modules.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration constants."""

    # --- Flask ---
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "default-fallback-secret")
    DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    # --- File Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
    FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index")

    # --- Document Processing ---
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    ALLOWED_EXTENSIONS = {"pdf"}

    # --- Embeddings ---
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Groq LLM ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

    # --- Retrieval ---
    TOP_K_CHUNKS = 5  # Number of chunks to retrieve

    # --- Chat Memory ---
    MAX_HISTORY = 3  # Number of past Q&A pairs to keep in session


def validate_config():
    """Validate that critical configuration values are set."""
    if not Config.GROQ_API_KEY or Config.GROQ_API_KEY == "gsk_your_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set. Please add it to your .env file.\n"
            "Get your key at: https://console.groq.com"
        )

    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.VECTORSTORE_DIR, exist_ok=True)