"""Golden-set retrieval evaluation with human-provided relevant page labels."""
from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from src.rag import RAGIndex, hybrid_retrieve


def load_goldens(raw_json: str) -> dict:
    data = json.loads(raw_json)
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("Golden JSON requires a 'cases' list.")
    for case in data["cases"]:
        if not isinstance(case.get("question"), str):
            raise ValueError("Every case needs a string question.")
        if not isinstance(case.get("relevant_pages"), list):
            raise ValueError("Every case needs relevant_pages (list of page numbers).")
    return data


def page_metrics(docs: list[Document], relevant_pages: list[int]) -> dict:
    """Chunk precision, page recall, and hit based on *page-level* gold labels."""
    if not relevant_pages:  # An unanswerable case; neither metric is defined.
        return {"hit_at_k": None, "precision_at_k": None, "page_recall_at_k": None}
    expected = set(relevant_pages)
    retrieved_pages = [d.metadata["page"] for d in docs]
    relevant_chunk_count = sum(page in expected for page in retrieved_pages)
    return {
        "hit_at_k": int(relevant_chunk_count > 0),
        "precision_at_k": relevant_chunk_count / len(docs) if docs else 0.0,
        "page_recall_at_k": len(set(retrieved_pages) & expected) / len(expected),
    }


def article_metrics(
    docs: list[Document],
    relevant_articles: list[int],
) -> dict:
    """Evaluate retrieved chunks using expected article numbers."""

    if not relevant_articles:
        return {
            "article_hit_at_k": None,
            "article_precision_at_k": None,
            "article_recall_at_k": None,
        }

    expected = set(relevant_articles)
    retrieved = [
        doc.metadata.get("article")
        for doc in docs
    ]

    matching_count = sum(
        article in expected for article in retrieved
    )

    found = set(retrieved) & expected

    return {
        "article_hit_at_k": int(matching_count > 0),
        "article_precision_at_k": (
            matching_count / len(docs) if docs else 0.0
        ),
        "article_recall_at_k": len(found) / len(expected),
    }


def run_retrieval_evaluation(index: RAGIndex, cases: list[dict], k: int = 5) -> list[dict]:
    rows = []
    for case in cases:
        question = case["question"]
        retrieved, best = hybrid_retrieve(index, question, k=k)
        values = page_metrics(retrieved, case["relevant_pages"])

        if "relevant_articles" in case:
            values.update(
                article_metrics(
                    retrieved,
                    case["relevant_articles"],
                )
            )

        rows.append({"question": question, "expected_answerable": case.get("answerable", True),
                     **values, "best_cosine": round(best, 3)})
    return rows



def run_abstention_evaluation(
    index: RAGIndex,
    cases: list[dict],
    k: int = 1,
    threshold: float = 0.22,
) -> list[dict]:
    """Evaluate cases explicitly labelled as unanswerable."""

    from src.scoring import answer_question

    rows = []

    for case in cases:
        if case.get("answerable") is not False:
            continue

        row = {
            "id": case.get("id", ""),
            "question": case["question"],
            "expected_status": "ABSTAINED",
        }

        try:
            result = answer_question(
                index,
                case["question"],
                threshold=threshold,
                k=k,
            )

            actual = result["status"]

            row["actual_status"] = actual
            row["passed"] = actual == "ABSTAINED"
            row["error_type"] = None

        except Exception as exc:
            row["actual_status"] = "ERROR"
            row["passed"] = False
            row["error_type"] = type(exc).__name__

        rows.append(row)

    return rows



def default_golden_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "golden_rag_triad.json"
