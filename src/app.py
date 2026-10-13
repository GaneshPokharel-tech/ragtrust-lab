"""RAGTrust Lab: PDF RAG, hallucination gate, guardrails and evaluation UI."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from statistics import mean

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Allow `uv run streamlit run src/app.py` from a non-packaged uv project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (default_golden_path, load_goldens, run_retrieval_evaluation, run_abstention_evaluation)
from src.rag import build_index
from src.scoring import answer_question

load_dotenv()
st.set_page_config(page_title="RAGTrust Lab", layout="wide")
st.title("RAGTrust Lab")
st.caption("RAG Triad | Hallucination Detection | Guardrails G1–G5 | Evaluation")

st.info(
    "Educational demo. Answers come from the selected document, "
    "not verified current legal advice."
)

# Select the built-in Constitution or upload a custom PDF.
source_mode = st.radio(
    "Knowledge source",
    [
        "Nepal Constitution (English, amended through 2020)",
        "Upload a PDF"
    ],
    horizontal=True
)

if source_mode.startswith("Nepal Constitution"):

    # Read the official English Constitution from local storage.
    constitution_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "constitution"
        / "constitution_en.pdf"
    )

    if not constitution_path.is_file():
        st.error("Constitution PDF not found.")
        st.stop()

    pdf_bytes = constitution_path.read_bytes()
    filename = constitution_path.name

    st.caption(
        "English Constitution PDF incorporating amendments through 2020. "
        "Later changes are not verified."
    )

else:

    # Preserve the existing custom PDF upload functionality.
    uploaded_pdf = st.file_uploader(
        "Upload a text-based PDF",
        type=["pdf"]
    )

    if uploaded_pdf is None:
        st.stop()

    pdf_bytes = uploaded_pdf.getvalue()
    filename = uploaded_pdf.name

# Enforce the existing 20 MB upload limit.
if len(pdf_bytes) > 20 * 1024 * 1024:
    st.error("Demo limit: PDF must be 20 MB or smaller.")
    st.stop()

# Index only on a new file. Keep this uploader's index in Streamlit session state.
doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
if st.session_state.get("doc_hash") != doc_hash:
    try:
        with st.spinner("Extracting pages, downloading embedding model if needed, indexing..."):
            index = build_index(pdf_bytes, filename)
        st.session_state.doc_hash = doc_hash
        st.session_state.index = index
        st.session_state.last_result = None
    except Exception as exc:
        st.error(f"Could not index PDF: {exc}")
        st.stop()

index = st.session_state.index
st.success(f"Indexed {index.total_pages} pages into {len(index.chunks)} chunks.")
st.caption("Embeddings: local MiniLM | Vector search: in-memory Chroma | Lexical search: TF-IDF")

with st.expander("Inspect ingestion and chunk metadata"):
    st.json(index.chunks[0].metadata)
    st.text_area("First chunk", index.chunks[0].page_content, height=180)

tab_chat, tab_eval = st.tabs(["Ask the PDF", "Evaluation dashboard"])

with tab_chat:
    question = st.text_input("Ask a question based only on the uploaded PDF")
    col_a, col_b = st.columns(2)
    with col_a:
        k = st.slider("Retrieved context chunks (k)", 1, 8, 5)
    with col_b:
        floor = st.slider("Minimum cosine similarity (demo threshold)", 0.0, 0.6, 0.22, 0.02)

    if st.button("Ask and verify", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            try:
                with st.spinner("Retrieving, generating, verifying and scoring..."):
                    st.session_state.last_result = answer_question(index, question, floor, k)
            except Exception as exc:
                # Do not expose provider error bodies or secrets in the UI.
                st.session_state.last_result = None
                st.error(f"Could not complete verification ({type(exc).__name__}). Check the terminal safely.")

    result = st.session_state.get("last_result")
    if result:
        st.subheader("Verified output")
        st.write(f"Status: **{result['status']}** | Trace: `{result['trace_id']}`")
        st.write(result["answer"])
        if result["status"] == "PARTIALLY_VERIFIED":
            st.warning("Some generated claims were withheld after checking their evidence.")

        st.subheader("RAG Triad")
        cols = st.columns(3)
        for col, (key, label) in zip(cols, [
            ("context_relevance", "Context Relevance"),
            ("groundedness", "Groundedness"),
            ("answer_relevance", "Answer Relevance"),
        ]):
            value = result["scores"].get(key)
            col.metric(label, "Not evaluated" if value is None else f"{value:.2f}")
        st.caption("Judges are imperfect estimates, not ground-truth accuracy. "
                   "Groundedness is computed over original generated claims.")

        if result["source_rows"]:
            st.subheader("Verified source citations")
            for row in result["source_rows"]:
                end = row.get("page_end", row["page"])
                page_label = (
                    f"pages {row['page']}-{end}"
                    if end > row["page"]
                    else f"page {row['page']}"
                )
                st.markdown(
                    f"**[{row['id']}] {row['source']}, PDF {page_label}**"
                )
                st.code(row["quote"])
        with st.expander("Retrieved evidence (G2/G3)"):
            for i, doc in enumerate(result["docs"], 1):
                start = doc.metadata["page"]
                end = doc.metadata.get("page_end", start)
                page_label = (
                    f"Pages {start}-{end}"
                    if end > start
                    else f"Page {start}"
                )
                st.markdown(f"**[S{i}] {page_label}**")
                st.text(doc.page_content[:1200])
        with st.expander("Guardrail decisions / trace"):
            for event in result["events"]:
                st.write("• " + event)
            if result["rejected_claims"]:
                st.write("Rejected claim checks:")
                for reason in result["rejected_claims"]:
                    st.write("• " + reason)

with tab_eval:
    st.markdown("**Golden set:** Questions with expected source labels.")

    project_root = Path(__file__).resolve().parents[1]

    if source_mode.startswith("Nepal Constitution"):
        default_dataset = (
            project_root / "data" / "golden_constitution.json"
        )
    else:
        default_dataset = default_golden_path()

    example = default_dataset.read_text(encoding="utf-8")

    st.caption(f"Default dataset: {default_dataset.name}")

    golden_file = st.file_uploader(
        "Optional: upload your own golden-set JSON",
        type=["json"],
    )

    try:
        dataset = load_goldens(
            golden_file.getvalue().decode("utf-8")
            if golden_file
            else example
        )
    except Exception as exc:
        st.error(f"Invalid golden dataset: {exc}")
        st.stop()

    expected = dataset.get("filename_contains", "")

    article_mode = any(
        bool(case.get("relevant_articles"))
        for case in dataset["cases"]
    )

    has_article_metadata = any(
        isinstance(doc.metadata.get("article"), int)
        for doc in index.chunks
    )

    if expected and expected.lower() not in index.filename.lower():
        st.warning(
            f"These labels belong to a file containing '{expected}'. "
            "Please provide labels for the selected PDF."
        )

    elif article_mode and not has_article_metadata:
        st.warning(
            "This dataset requires article metadata, "
            "but the selected PDF is not article-indexed."
        )

    else:
        st.write(f"Loaded {len(dataset['cases'])} evaluation cases.")

        evaluation_k = st.slider(
            "Evaluation retrieval depth (k)",
            min_value=1,
            max_value=8,
            value=5,
            key="evaluation_k",
        )

        if st.button("Run retrieval evaluation"):
            with st.spinner("Running deterministic retrieval tests..."):
                rows = run_retrieval_evaluation(
                    index,
                    dataset["cases"],
                    k=evaluation_k,
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
            )

            if article_mode:
                metric_keys = (
                    "article_hit_at_k",
                    "article_precision_at_k",
                    "article_recall_at_k",
                )
                labels = (
                    f"Article Hit@{evaluation_k}",
                    f"Article Precision@{evaluation_k}",
                    f"Article Recall@{evaluation_k}",
                )
            else:
                metric_keys = (
                    "hit_at_k",
                    "precision_at_k",
                    "page_recall_at_k",
                )
                labels = (
                    f"Page Hit@{evaluation_k}",
                    f"Chunk Precision@{evaluation_k}",
                    f"Page Recall@{evaluation_k}",
                )

            positive = [
                row for row in rows
                if row.get(metric_keys[1]) is not None
            ]

            if positive:
                cols = st.columns(3)

                for col, key, label in zip(
                    cols,
                    metric_keys,
                    labels,
                ):
                    values = [
                        row[key]
                        for row in positive
                        if row.get(key) is not None
                    ]

                    col.metric(
                        label,
                        f"{mean(values):.3f}" if values else "N/A",
                    )

            st.caption(
                "Unanswerable cases are excluded from retrieval "
                "precision and recall. Abstention must be tested separately."
            )

        st.divider()
        st.write("**Abstention evaluation:** Test unanswerable questions.")

        unanswerable_cases = [
            case for case in dataset["cases"]
            if case.get("answerable") is False
        ]

        st.caption(
            f"{len(unanswerable_cases)} unanswerable cases available. "
            "Running this evaluation may call Gemini."
        )

        if st.button(
            "Run abstention evaluation",
            disabled=not unanswerable_cases,
        ):
            with st.spinner("Evaluating abstention behavior..."):
                abstention_rows = run_abstention_evaluation(
                    index,
                    unanswerable_cases,
                    k=1,
                    threshold=0.22,
                )

            st.dataframe(
                pd.DataFrame(abstention_rows),
                use_container_width=True,
            )

            passed = sum(
                row["passed"] for row in abstention_rows
            )
            total = len(abstention_rows)

            st.metric(
                "Abstention tests passed",
                f"{passed}/{total}",
            )

            if passed == total:
                st.success("All abstention tests passed.")
            else:
                st.warning(
                    "Some abstention tests failed. "
                    "Review the results above."
                )

        st.divider()
        st.write("**End-to-end evaluation:** runs generation + claim verification + RAG Triad judges.")
        st.caption("Calls Gemini several times per case. Uses first three labelled cases to limit free-tier usage.")
        if st.button("Run first 3 end-to-end cases"):
            rows = []
            progress = st.progress(0)
            for i, case in enumerate(dataset["cases"][:3]):
                try:
                    result = answer_question(index, case["question"])
                    rows.append({"question": case["question"], "status": result["status"],
                                 **result["scores"]})
                except Exception as exc:
                    rows.append({"question": case["question"],
                                 "status": f"ERROR: {type(exc).__name__}"})
                progress.progress((i + 1) / min(3, len(dataset["cases"])))
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption("Full release evaluation needs many more human-reviewed cases and judge calibration.")
