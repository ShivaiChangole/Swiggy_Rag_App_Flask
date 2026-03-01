"""
app.py
------
Main Flask application for the Swiggy Annual Report RAG system.

Routes:
  GET  /       → Home page (upload documents)
  POST /upload → Handle PDF upload and ingestion
  GET  /chat   → Chat interface
  POST /ask    → Process a question through the RAG pipeline
  POST /clear  → Clear chat history
"""

import os
import logging
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
)
from werkzeug.utils import secure_filename

from config import Config, validate_config
from rag_engine.ingest import ingest_documents
from rag_engine.retriever import refresh_vectorstore, get_vectorstore
from rag_engine.generator import answer_question

# ==============================================================================
# Configure Logging
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Validate Configuration
# ==============================================================================
validate_config()

# ==============================================================================
# Initialize Flask App
# ==============================================================================
app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload size


def allowed_file(filename: str) -> bool:
    """Check if the uploaded file has an allowed extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


# ==============================================================================
# Load FAISS index on startup (if it exists)
# ==============================================================================
with app.app_context():
    logger.info("Attempting to load existing FAISS index on startup...")
    vs = get_vectorstore()
    if vs:
        logger.info("Existing FAISS index loaded — ready for queries.")
    else:
        logger.info("No existing FAISS index found — upload documents to create one.")


# ==============================================================================
# ROUTES
# ==============================================================================


@app.route("/")
def index():
    """Home page — document upload interface."""
    # List already-uploaded files
    uploaded_files = []
    if os.path.exists(Config.UPLOAD_FOLDER):
        uploaded_files = [
            f for f in os.listdir(Config.UPLOAD_FOLDER)
            if f.lower().endswith(".pdf")
        ]

    # Check if vector store is ready
    vectorstore_ready = get_vectorstore() is not None

    return render_template(
        "index.html",
        uploaded_files=uploaded_files,
        vectorstore_ready=vectorstore_ready,
    )


@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF file upload and trigger document ingestion."""

    # --- Validate files in request ---
    if "files" not in request.files:
        flash("No files selected.", "error")
        return redirect(url_for("index"))

    files = request.files.getlist("files")

    if not files or all(f.filename == "" for f in files):
        flash("No files selected.", "error")
        return redirect(url_for("index"))

    # --- Save uploaded files ---
    saved_paths = []
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            saved_paths.append(filepath)
            logger.info(f"Saved uploaded file: {filename}")
        else:
            flash(f"Skipped invalid file: {file.filename}", "warning")

    if not saved_paths:
        flash("No valid PDF files uploaded.", "error")
        return redirect(url_for("index"))

    # --- Run ingestion pipeline ---
    try:
        flash(f"Processing {len(saved_paths)} PDF(s)... This may take a moment.", "info")
        ingest_documents(saved_paths)

        # Refresh the cached vector store
        refresh_vectorstore()

        flash(
            f"Successfully processed {len(saved_paths)} document(s). "
            f"You can now ask questions!",
            "success",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        flash(f"Error processing documents: {str(e)}", "error")

    return redirect(url_for("index"))


@app.route("/chat")
def chat():
    """Chat interface — ask questions about uploaded documents."""
    # Check if vector store exists
    if get_vectorstore() is None:
        flash("Please upload and process documents before chatting.", "warning")
        return redirect(url_for("index"))

    # Get chat history from session
    history = session.get("chat_history", [])

    return render_template("chat.html", history=history)


@app.route("/ask", methods=["POST"])
def ask():
    """Process a user question through the RAG pipeline."""
    # Get question from form or JSON
    if request.is_json:
        data = request.get_json()
        question = data.get("question", "").strip()
    else:
        question = request.form.get("question", "").strip()

    if not question:
        flash("Please enter a question.", "warning")
        return redirect(url_for("chat"))

    # Check vector store availability
    if get_vectorstore() is None:
        flash("No documents have been processed. Please upload PDFs first.", "error")
        return redirect(url_for("index"))

    # --- Run RAG pipeline ---
    try:
        result = answer_question(question)
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        result = {
            "question": question,
            "answer": f"An error occurred: {str(e)}",
            "sources": [],
            "has_sources": False,
        }

    # --- Update chat history in session (keep last N) ---
    history = session.get("chat_history", [])
    history.append({
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "has_sources": result["has_sources"],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    })

    # Keep only the last MAX_HISTORY entries
    session["chat_history"] = history[-Config.MAX_HISTORY:]

    return render_template(
        "chat.html",
        history=session["chat_history"],
        latest=result,
    )


@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear the chat history from session."""
    session.pop("chat_history", None)
    flash("Chat history cleared.", "info")
    return redirect(url_for("chat"))


# ==============================================================================
# Error Handlers
# ==============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large errors."""
    flash("File is too large. Maximum size is 50 MB.", "error")
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template("base.html", error="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}", exc_info=True)
    return render_template("base.html", error="Internal server error."), 500


# ==============================================================================
# Run the Application
# ==============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=Config.DEBUG,
    )