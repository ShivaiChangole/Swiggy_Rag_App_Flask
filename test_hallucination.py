"""
test_hallucination.py
---------------------
Automated hallucination detection test suite.
Run this after uploading documents to verify RAG grounding.

Usage:
    python test_hallucination.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_engine.generator import answer_question
from rag_engine.retriever import get_vectorstore
from config import validate_config

# ANSI color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(title):
    print(f"\n{'='*70}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{'='*70}")


def print_result(test_num, question, passed, reason=""):
    status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
    print(f"\n  Test {test_num}: {status}")
    print(f"  Q: {question}")
    if reason:
        print(f"  → {reason}")


def check_refusal(answer):
    """Check if the answer is a proper refusal (not hallucinated)."""
    refusal_phrases = [
        "not found in document",
        "not found in the document",
        "not mentioned in the",
        "no information",
        "not available in",
        "cannot find",
        "does not mention",
        "does not contain",
        "not present in",
        "no relevant information",
        "i cannot answer",
        "i don't have",
        "not included in",
        "the context does not",
        "the document does not",
        "the provided context",
        "based on the provided context",
        "not specifically mentioned",
    ]
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in refusal_phrases)


def check_no_fabricated_numbers(answer):
    """Basic check — flag if answer contains suspiciously precise numbers
    that might be hallucinated. This is a heuristic, not definitive."""
    import re
    # Look for very specific numbers that might be fabricated
    # This flags for manual review, not automatic failure
    numbers = re.findall(r'\$[\d,]+\.?\d*|\₹[\d,]+\.?\d*|\d+\.\d{3,}%', answer)
    return len(numbers) == 0  # True if no suspiciously precise numbers


def run_tests():
    """Run the complete hallucination test suite."""

    validate_config()

    # Check if vector store is ready
    vs = get_vectorstore()
    if vs is None:
        print(f"{RED}ERROR: No documents uploaded. Upload PDFs first, then run tests.{RESET}")
        sys.exit(1)

    results = {"passed": 0, "failed": 0, "warnings": 0}

    # =====================================================================
    # TEST CATEGORY 1: Out-of-Scope Questions (Must Refuse)
    # =====================================================================
    print_header("CATEGORY 1: Out-of-Scope Questions (System MUST refuse)")

    out_of_scope_questions = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "What is Zomato's annual revenue?",
        "Who won the FIFA World Cup in 2022?",
        "What is the theory of relativity?",
        "Tell me about Amazon's AWS services.",
        "What is Bitcoin's current price?",
        "How does a nuclear reactor work?",
    ]

    for i, question in enumerate(out_of_scope_questions, 1):
        result = answer_question(question)
        answer = result["answer"]
        passed = check_refusal(answer)

        if passed:
            results["passed"] += 1
            print_result(i, question, True, f"Correctly refused: \"{answer[:80]}...\"")
        else:
            results["failed"] += 1
            print_result(i, question, False, f"DID NOT REFUSE! Answer: \"{answer[:120]}...\"")

    # =====================================================================
    # TEST CATEGORY 2: Fabricated Entity Questions (Must Refuse)
    # =====================================================================
    print_header("CATEGORY 2: Fabricated Entity Questions (System MUST refuse)")

    fabricated_questions = [
        "What does the report say about Swiggy's operations in Antarctica?",
        "How much did Swiggy invest in space exploration?",
        "What is Swiggy's cryptocurrency portfolio worth?",
        "Describe Swiggy's partnership with NASA.",
        "What does the report say about Swiggy's nuclear energy division?",
    ]

    for i, question in enumerate(fabricated_questions, 1):
        result = answer_question(question)
        answer = result["answer"]
        passed = check_refusal(answer)

        if passed:
            results["passed"] += 1
            print_result(i, question, True, f"Correctly refused: \"{answer[:80]}...\"")
        else:
            results["failed"] += 1
            print_result(i, question, False, f"HALLUCINATED! Answer: \"{answer[:120]}...\"")

    # =====================================================================
    # TEST CATEGORY 3: Leading/Suggestive Questions (Must Not Agree)
    # =====================================================================
    print_header("CATEGORY 3: Leading Questions (System must NOT agree with false premises)")

    leading_questions = [
        "Confirm that Swiggy's revenue exceeded ₹1 lakh crore.",
        "The report mentions Swiggy is profitable since 2019, what were the exact figures?",
        "Summarize Swiggy's acquisition of Uber Eats India for $500 billion.",
        "What does the report say about Swiggy having 10 million employees?",
    ]

    for i, question in enumerate(leading_questions, 1):
        result = answer_question(question)
        answer = result["answer"]

        # For leading questions, either refuse or correct — but should NOT agree
        answer_lower = answer.lower()
        agrees_with_false = any(phrase in answer_lower for phrase in [
            "yes", "correct", "confirmed", "as you mentioned", "indeed",
            "that's right", "you are right",
        ])

        if check_refusal(answer):
            results["passed"] += 1
            print_result(i, question, True, f"Correctly refused: \"{answer[:80]}...\"")
        elif not agrees_with_false:
            results["passed"] += 1
            print_result(i, question, True, f"Did not agree with false premise: \"{answer[:80]}...\"")
        else:
            results["failed"] += 1
            print_result(i, question, False, f"AGREED WITH FALSE PREMISE! Answer: \"{answer[:120]}...\"")

    # =====================================================================
    # TEST CATEGORY 4: Valid Questions (Should Answer WITH Citations)
    # =====================================================================
    print_header("CATEGORY 4: Valid Questions (Should answer WITH citations)")

    valid_questions = [
        "What is Swiggy's total revenue?",
        "What risks are mentioned in the report?",
        "Who are the directors or board members?",
    ]

    for i, question in enumerate(valid_questions, 1):
        result = answer_question(question)
        answer = result["answer"]
        sources = result["sources"]
        has_sources = result["has_sources"]

        if has_sources and sources and not check_refusal(answer):
            results["passed"] += 1
            print_result(
                i, question, True,
                f"Answered with {len(sources)} citation(s). "
                f"Answer: \"{answer[:80]}...\""
            )
        elif check_refusal(answer):
            results["warnings"] += 1
            print(f"\n  Test {i}: {YELLOW}⚠️  WARNING{RESET}")
            print(f"  Q: {question}")
            print(f"  → Refused to answer a likely valid question. May need better document coverage.")
        else:
            results["warnings"] += 1
            print(f"\n  Test {i}: {YELLOW}⚠️  WARNING{RESET}")
            print(f"  Q: {question}")
            print(f"  → Answered but with no citations: \"{answer[:80]}...\"")

    # =====================================================================
    # TEST CATEGORY 5: Citation Integrity Check
    # =====================================================================
    print_header("CATEGORY 5: Citation Integrity Check")

    test_q = "What is Swiggy's revenue?"
    result = answer_question(test_q)

    if result["sources"]:
        print(f"\n  Testing citation for: \"{test_q}\"")
        print(f"  Answer: \"{result['answer'][:100]}...\"")
        print(f"\n  Citations returned:")

        all_citations_valid = True
        for j, src in enumerate(result["sources"], 1):
            has_source = bool(src.get("source"))
            has_page = bool(src.get("page"))
            has_snippet = bool(src.get("snippet")) and len(src["snippet"]) > 10

            valid = has_source and has_page and has_snippet

            status = f"{GREEN}✓{RESET}" if valid else f"{RED}✗{RESET}"
            print(f"    {status} Source {j}: {src.get('source', 'N/A')} | "
                  f"Page {src.get('page', 'N/A')} | "
                  f"Snippet length: {len(src.get('snippet', ''))} chars")

            if not valid:
                all_citations_valid = False

        if all_citations_valid:
            results["passed"] += 1
            print(f"\n  {GREEN}✅ All citations have valid structure{RESET}")
        else:
            results["failed"] += 1
            print(f"\n  {RED}❌ Some citations are incomplete{RESET}")

        print(f"\n  {YELLOW}⚠️  MANUAL STEP REQUIRED:{RESET}")
        print(f"  Open the PDF and verify that Page {result['sources'][0]['page']} ")
        print(f"  actually contains: \"{result['sources'][0]['snippet'][:60]}...\"")
    else:
        results["warnings"] += 1
        print(f"  {YELLOW}⚠️  No sources returned for valid question{RESET}")

    # =====================================================================
    # FINAL REPORT
    # =====================================================================
    print_header("HALLUCINATION TEST REPORT")

    total = results["passed"] + results["failed"] + results["warnings"]
    pass_rate = (results["passed"] / total * 100) if total > 0 else 0

    print(f"""
    Total Tests:    {total}
    {GREEN}Passed:         {results['passed']}{RESET}
    {RED}Failed:         {results['failed']}{RESET}
    {YELLOW}Warnings:       {results['warnings']}{RESET}

    Pass Rate:      {pass_rate:.1f}%
    """)

    if results["failed"] == 0:
        print(f"  {GREEN}{BOLD}🎉 ALL HALLUCINATION TESTS PASSED!{RESET}")
        print(f"  The system is properly grounded in document context.")
    else:
        print(f"  {RED}{BOLD}⚠️  HALLUCINATION DETECTED IN {results['failed']} TEST(S)!{RESET}")
        print(f"  Review failed tests and adjust prompts or retrieval settings.")

    if results["warnings"] > 0:
        print(f"\n  {YELLOW}Note: {results['warnings']} warning(s) require manual review.{RESET}")

    return results["failed"] == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)