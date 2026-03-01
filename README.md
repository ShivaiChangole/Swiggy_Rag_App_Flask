# Swiggy RAG System

A **Retrieval-Augmented Generation (RAG)** web application that enables intelligent question-answering over PDF documents (e.g., Swiggy Annual Reports) with built-in **hallucination prevention**.

## Overview

This system combines semantic search with a large language model (LLM) to provide accurate, context-grounded answers. It prevents hallucination by enforcing strict rules that ensure answers are only generated from the uploaded documents—never from external knowledge.

### Key Features

- 📄 **PDF Upload & Processing**: Upload AR (annual reports) or any PDF documents
- 🔍 **Semantic Search**: FAISS vector store for fast similarity-based retrieval
- 🤖 **Context-Grounded Answers**: Uses Groq LLM with strict hallucination prevention prompts
- 💬 **Chat Interface**: Web-based UI for interactive Q&A with chat history
- 📊 **Document Indexing**: Automatic chunking and embedding of documents
- ✅ **Hallucination Testing**: Built-in test suite to verify answer quality

The source file is in root folder
## Project Structure

```
swiggy-rag/
├── app.py                    # Main Flask application & routes
├── config.py                 # Configuration & environment setup
├── requirements.txt          # Python dependencies
├── test_hallucination.py     # Hallucination prevention tests
│
├── rag_engine/               # Core RAG logic
│   ├── __init__.py
│   ├── ingest.py             # Document ingestion & chunking
│   ├── retriever.py          # Semantic search with FAISS
│   └── generator.py          # Answer generation with Groq LLM
│
├── templates/                # HTML templates
│   ├── base.html             # Base layout template
│   ├── index.html            # Document upload page
│   └── chat.html             # Chat interface page
│
├── static/                   # Static assets
│   └── styles.css            # Application styling
│
├── uploads/                  # Uploaded PDF documents
├── vectorstore/              # Vector store data
│   └── faiss_index/          # FAISS index files
│
└── __pycache__/              # Python cache (compiled files)
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Web Framework** | Flask 3.1.1 |
| **Document Processing** | LangChain 0.3.25 + PyPDF |
| **Vector Database** | FAISS (CPU) 1.11.0 |
| **Embeddings** | Sentence Transformers (MiniLM-L6-v2) |
| **LLM API** | Groq (llama3-8b) |
| **Frontend** | HTML5, CSS3 |

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd swiggy-rag
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create `.env` file** in the project root
   ```env
   FLASK_SECRET_KEY=your-secret-key-here
   FLASK_DEBUG=True
   GROQ_API_KEY=gsk_your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```
   
   Get your Groq API key from: https://console.groq.com/

5. **Create necessary directories**
   ```bash
   mkdir -p uploads vectorstore
   ```

6. **Run the application**
   ```bash
   python app.py
   ```
   
   The app will be available at: `http://localhost:5000`

## Usage

### 1. Upload Documents

1. Navigate to the home page (`/`)
2. Click "Choose File" and select PDF documents
3. Click "Upload" to process and index the documents
4. The system will:
   - Extract text from PDFs
   - Split text into semantic chunks (800 chars with 150 char overlap)
   - Generate embeddings using Sentence Transformers
   - Store in FAISS vector index

### 2. Ask Questions

1. Go to the Chat page (`/chat`)
2. Type your question (e.g., "What was Swiggy's revenue in 2023?")
3. The system will:
   - Retrieve the 4 most relevant document chunks
   - Pass them to the Groq LLM with strict grounding rules
   - Return an answer grounded only in the provided context
4. Chat history is maintained in the session

### 3. Clear Chat History

Use the "Clear History" button to reset the conversation.

## Configuration

Key settings in `config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CHUNK_SIZE` | 800 | Characters per document chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks for continuity |
| `TOP_K_CHUNKS` | 4 | Number of retrieved chunks per query |
| `MAX_HISTORY` | 3 | Chat history pairs to maintain |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence Transformer model |
| `GROQ_MODEL` | llama3-8b-8192 | Groq LLM model |

Modify these in `config.py` or via environment variables.

## Hallucination Prevention

This system enforces strict hallucination prevention through:

### 1. **Strict System Prompt**
The LLM is explicitly instructed to:
- Answer ONLY from provided context
- Return "Not found in document" if answer isn't in context
- Never use outside knowledge or speculation

### 2. **Context-Only Generation**
- Questions are answered only with retrieved document chunks
- No external knowledge base is consulted

### 3. **Validation**
Run the test suite to verify hallucination prevention:
```bash
python test_hallucination.py
```

## API Routes

| Method | Route | Purpose |
|--------|-------|---------|
| `GET` | `/` | Home page with document upload |
| `POST` | `/upload` | Handle PDF file uploads |
| `GET` | `/chat` | Chat interface page |
| `POST` | `/ask` | Process question through RAG pipeline |
| `POST` | `/clear` | Clear chat session history |

### `/ask` Request Example
```json
{
  "question": "What was the operating profit?"
}
```

### `/ask` Response Example
```json
{
  "answer": "The operating profit was $150 million.",
  "sources": ["annual_report_2023.pdf"]
}
```

## Development

### Project Layout

- **`rag_engine/ingest.py`**: Document loading, chunking, and embedding
- **`rag_engine/retriever.py`**: FAISS vector store management and similarity search
- **`rag_engine/generator.py`**: LLM integration and answer generation
- **`app.py`**: Flask routes and request handling

### Logging

Application logs are printed to stderr with ISO format timestamps:
```
2025-03-01 10:30:15 [INFO] rag_engine.ingest: Ingesting: document.pdf
```

Adjust logging level in `app.py`:
```python
logging.basicConfig(level=logging.DEBUG)  # More verbose
```

### Extending the System

To add new features:

1. **Custom embeddings**: Modify `EMBEDDING_MODEL` in `config.py`
2. **Different LLM**: Update `rag_engine/generator.py` to use another API
3. **Persistent chat**: Replace session-based memory in `app.py`
4. **Database integration**: Add SQLAlchemy models for document metadata

## Troubleshooting

### Issue: "No existing FAISS index found"
- **Solution**: Upload documents first on the home page

### Issue: "GROQ_API_KEY not found"
- **Solution**: Ensure `.env` file contains valid Groq API key

### Issue: Slow query responses
- **Reasons**: First query loads FAISS index (~2-3 sec), or LLM API latency
- **Solution**: Index caching prevents repeated loads; latency depends on Groq API

### Issue: Poor answer quality
- **Check**: 
  - Retrieved chunks are relevant (visible in logs)
  - Question matches document content
  - Document text was extracted correctly from PDF

## Performance

- **First query**: ~2-3 seconds (index loading + LLM inference)
- **Subsequent queries**: ~1-2 seconds (cached index + LLM)
- **Typical response time**: Dominated by Groq API latency (usually <2s)

## Limits

- **Max upload size**: 50 MB
- **Allowed formats**: PDF only
- **Concurrent sessions**: Limited by Flask/compute resources
- **Vector store**: In-memory FAISS (CPU-based)

## License

This project is provided as-is for research and development purposes.

## Support

For issues or questions:
1. Check the logs in terminal for detailed error messages
2. Verify `.env` configuration
3. Test with `test_hallucination.py`
4. Review the Groq API status at https://console.groq.com/

---

**Version**: 1.0  
**Last Updated**: March 2025
