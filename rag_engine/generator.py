"""
rag_engine/generator.py
-----------------------
Handles grounded answer generation using the Groq LLM API.
Enforces strict context-only answering to prevent hallucination.
"""

import logging
import requests
from typing import Dict, Any, List, Optional

from config import Config
from rag_engine.retriever import retrieve_relevant_chunks, format_context_for_prompt

# Module logger
logger = logging.getLogger(__name__)

# ==============================================================================
# SYSTEM PROMPT — This is the heart of hallucination prevention
# ==============================================================================
SYSTEM_PROMPT = """You are a precise document analysis assistant. Your ONLY job is to answer questions using the provided context extracted from uploaded PDF documents.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information explicitly stated in the provided context.
2. If the answer is NOT found in the context, respond EXACTLY with: "Not found in document."
3. NEVER use outside knowledge, training data, or general knowledge.
4. NEVER guess, speculate, infer, or assume anything not in the context.
5. If asked for numbers, dates, or statistics, return the EXACT values from the context.
6. If the context partially answers the question, answer only the part you can support and state that the rest was not found.
7. When possible, mention which source and page the information comes from.
8. Keep answers concise and factual.

Remember: It is better to say "Not found in document." than to provide incorrect information."""


def build_prompt(question: str, context: str) -> List[Dict[str, str]]:
    """
    Build the message list for the Groq Chat Completions API.

    Constructs a conversation with:
      - System message: strict grounding rules
      - User message: context + question

    Args:
        question: The user's question.
        context: Formatted context string from retrieved chunks.

    Returns:
        List of message dicts in OpenAI chat format.
    """
    user_message = f"""CONTEXT FROM UPLOADED DOCUMENTS:
=================================
{context}
=================================

QUESTION: {question}

Based STRICTLY on the context above, provide a precise answer. If the information is not in the context, reply exactly: "Not found in document."
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    return messages


def call_groq_api(messages: List[Dict[str, str]]) -> str:
    """
    Call the Groq API for chat completion.

    Makes a POST request to the Groq OpenAI-compatible endpoint
    with the constructed messages. Uses low temperature for
    deterministic, factual responses.

    Args:
        messages: List of message dicts (system + user).

    Returns:
        The LLM's response text.

    Raises:
        requests.exceptions.RequestException: On API communication failure.
        ValueError: If API returns an unexpected response format.
    """
    headers = {
        "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": Config.GROQ_MODEL,
        "messages": messages,
        "temperature": 0.1,      # Low temperature = more deterministic
        "max_tokens": 1024,
        "top_p": 0.9,
        "stream": False,
    }

    logger.info(f"Calling Groq API with model: {Config.GROQ_MODEL}")

    try:
        response = requests.post(
            Config.GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        logger.error("Groq API request timed out.")
        return "Error: The AI service timed out. Please try again."

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Groq API.")
        return "Error: Could not connect to the AI service. Check your internet connection."

    except requests.exceptions.HTTPError as e:
        logger.error(f"Groq API HTTP error: {e}")
        if response.status_code == 401:
            return "Error: Invalid API key. Please check your GROQ_API_KEY."
        elif response.status_code == 429:
            return "Error: Rate limit exceeded. Please wait a moment and try again."
        return f"Error: AI service returned status {response.status_code}."

    # Parse response
    try:
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        logger.info(f"Groq API responded successfully ({len(answer)} chars).")
        return answer

    except (KeyError, IndexError) as e:
        logger.error(f"Unexpected Groq API response format: {e}")
        logger.debug(f"Response body: {response.text[:500]}")
        return "Error: Received an unexpected response from the AI service."


def answer_question(question: str) -> Dict[str, Any]:
    """
    Complete RAG pipeline: Retrieve → Build Prompt → Generate Answer.

    This is the main entry point for question answering. It:
      1. Retrieves relevant chunks from the FAISS index
      2. Formats them into a grounded prompt
      3. Sends to Groq LLM for answer generation
      4. Returns the answer along with source citations

    Args:
        question: The user's natural language question.

    Returns:
        Dict containing:
          - question: The original question
          - answer: The LLM's grounded answer
          - sources: List of source citation dicts
          - has_sources: Boolean indicating if context was found
    """
    logger.info(f"Processing question: '{question[:100]}...'")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve_relevant_chunks(question)

    if not chunks:
        logger.warning("No chunks retrieved. Vector store may be empty.")
        return {
            "question": question,
            "answer": "No documents have been uploaded yet. Please upload PDF documents first.",
            "sources": [],
            "has_sources": False,
        }

    # Step 2: Format context for prompt
    context = format_context_for_prompt(chunks)

    # Step 3: Build the grounded prompt
    messages = build_prompt(question, context)

    # Step 4: Get answer from Groq LLM
    answer = call_groq_api(messages)

    # Step 5: Prepare source citations
    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "snippet": chunk["snippet"],
            "score": chunk["score"],
        }
        for chunk in chunks
    ]

    logger.info("Question answered successfully.")

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "has_sources": True,
    }