"""
rag_engine package
------------------
Contains the core RAG pipeline modules:
  - ingest.py   : PDF loading, text cleaning, chunking, and FAISS indexing
  - retriever.py : Semantic similarity search over the FAISS vector store
  - generator.py : Grounded answer generation via Groq LLM
"""