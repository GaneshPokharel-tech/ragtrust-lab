# RAGTrust Lab

**A Nepal Constitution RAG Chatbot with Hallucination Detection, Guardrails, and Evaluation**

RAGTrust Lab is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the **Constitution of Nepal** and receive evidence-based answers with source citations.

The project combines hybrid document retrieval, Google Gemini, claim verification, five-layer guardrails, and the RAG Triad evaluation framework.

It is built as an educational, English-language MVP focused on reliable retrieval, grounded answer generation, and measurable system performance.

---

## 1. Key Features

### Constitution-Based Question Answering
- Supports the English Constitution of Nepal, incorporating amendments through 2020.
- Extracts and indexes all 308 constitutional articles.
- Answers questions about constitutional rights, provisions, and responsibilities.
- Supports direct article references and natural-language questions.
- Supports questions comparing multiple articles.

### Hybrid Retrieval
- **Chroma:** Vector similarity search.
- **Sentence Transformers:** Local text embeddings.
- **TF-IDF:** Keyword-based retrieval.
- **Reciprocal Rank Fusion (RRF):** Combines retrieval results.
- **Article-Title Reranking:** Improves retrieval using constitutional article headings.
- **Article-Aware Chunking:** Preserves article identity and metadata.

### Hallucination Detection
- Generates answers using retrieved evidence.
- Requires claims to reference supporting source passages.
- Verifies citations and supporting quotations.
- Uses claim-level verification to identify unsupported information.
- Abstains when sufficient evidence is unavailable.

### Five-Layer Guardrails

| Layer | Purpose |
|---|---|
| G1 | Input validation and basic input safety checks |
| G2 | Document-restricted retrieval and article reference validation |
| G3 | Treat retrieved documents as untrusted evidence |
| G4 | Closed-book, structured answer generation |
| G5 | Citation validation and claim verification |

### RAG Triad Evaluation

The application evaluates three dimensions:

1. **Context Relevance:** How relevant is the retrieved context to the question?
2. **Groundedness:** Are the generated claims supported by retrieved evidence?
3. **Answer Relevance:** How well does the answer address the user's question?

Evaluation scores are model-based estimates, not guarantees of factual accuracy.

### Evaluation Dashboard

- Article Hit@K
- Article Precision@K
- Article Recall@K
- Page-based retrieval evaluation
- Automated abstention evaluation
- End-to-end answer evaluation
- Custom JSON golden dataset upload

---

## 2. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12+ | Core programming language |
| Streamlit | Web application interface |
| LangChain | LLM and document-processing integration |
| Google Gemini | Answer generation and evaluation |
| Sentence Transformers | Local text embeddings |
| Chroma | Vector database |
| Scikit-learn | TF-IDF keyword retrieval |
| PyPDF | PDF text extraction |
| python-docx | Word document processing utilities |
| NumPy | Vector and numerical operations |
| Pandas | Evaluation result tables |
| Pydantic | Structured model outputs |
| pytest | Automated testing |
| uv | Package and environment management |

---

## 3. Project Structure

```text
ragtrust-lab/
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── rag.py
│   ├── scoring.py
│   ├── safety.py
│   └── evaluation.py
│
├── data/
│   ├── constitution/
│   │   └── constitution_en.pdf
│   │
│   ├── constitution_archive/
│   ├── golden_constitution.json
│   └── golden_rag_triad.json
│
├── tests/
│   └── test_core.py
│
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── main.py
└── README.md
```

### Important Files

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface |
| `rag.py` | Document ingestion, indexing, and retrieval |
| `scoring.py` | Gemini generation, claim verification, and RAG Triad |
| `safety.py` | Input and output safety checks |
| `evaluation.py` | Retrieval and abstention evaluation |
| `golden_constitution.json` | Constitution evaluation questions and expected labels |
| `test_core.py` | Automated regression tests |

The local project may contain additional Constitution source documents and archived files.

---

## 4. System Architecture

```text
               USER QUESTION
                     |
                     v
            G1: INPUT VALIDATION
                     |
                     v
          ARTICLE REFERENCE CHECK
                     |
                     v
             G2: RETRIEVAL
                     |
          +----------+----------+
          |                     |
          v                     v
     CHROMA SEARCH          TF-IDF SEARCH
          |                     |
          +----------+----------+
                     |
                     v
          RECIPROCAL RANK FUSION
                     |
                     v
          ARTICLE-TITLE RERANKING
                     |
                     v
           RETRIEVED EVIDENCE
                     |
                     v
        G3: UNTRUSTED DATA HANDLING
                     |
                     v
       G4: GEMINI ANSWER GENERATION
                     |
                     v
         G5: CLAIM AND CITATION CHECK
                     |
                     v
           VERIFIED ANSWER
                     |
          +----------+----------+
          |                     |
          v                     v
     SOURCE CITATIONS       RAG TRIAD
```

---

## 5. Installation and Setup

### Prerequisites

- Python 3.12 or newer
- uv package manager
- Google Gemini API key
- Internet connection for model downloads and Gemini API calls

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ragtrust-lab
```

Replace `<repository-url>` with the actual GitHub repository URL after publication.

### Step 2: Install Dependencies

```bash
uv sync
```

This installs the dependencies declared in `pyproject.toml` and resolved in `uv.lock`.

### Step 3: Configure Environment Variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your actual API key.

**Security:** Never commit your `.env` file or publish real API keys.

### Step 4: Prepare the Constitution Document

The English Constitution PDF must be available at:

```text
data/constitution/constitution_en.pdf
```

The built-in article-aware parser is designed for the verified English Constitution PDF used in this project.

Other PDF versions may require parser changes and additional validation.

### Step 5: Start the Application

```bash
uv run streamlit run src/app.py
```

Open the local address displayed by Streamlit, normally:

```text
http://localhost:8501
```

---

## 6. How to Use

### Ask Questions About the Constitution

1. Open the Streamlit application.
2. Select **Nepal Constitution (English, amended through 2020)**.
3. Open the **Ask the PDF** tab.
4. Enter your question.
5. Select the number of retrieved context chunks.
6. Click **Ask and verify**.
7. Review the answer, citations, RAG Triad, and guardrail trace.

### Example Questions

**Single Article**

```text
What does Article 18 say about the right to equality?
```

**Natural Language**

```text
What constitutional protection exists against censorship of news?
```

**Multiple Articles**

```text
Compare Article 18 and Article 38 of the Constitution of Nepal.
```

**Unanswerable Question**

```text
What does Article 400 of the Constitution of Nepal state?
```

The system is designed to abstain when the selected document cannot support a reliable answer.

### Upload a Custom PDF

Select **Upload a PDF** to use a different text-based document.

The application creates a document index and processes questions using the existing RAG and guardrail workflow.

---

## 7. Evaluation

The **Evaluation dashboard** supports retrieval testing, abstention testing, and end-to-end evaluation.

### Golden Dataset

The Constitution dataset is located at:

```text
data/golden_constitution.json
```

It currently contains:

| Test Type | Count |
|---|---:|
| Answerable questions | 11 |
| Unanswerable questions | 2 |
| Total | 13 |

The dataset contains provisional labels based on the local Constitution source. Broader human review is required before using it as a comprehensive benchmark.

### Retrieval Metrics

**Article Hit@K**

Measures whether at least one expected article is retrieved within the top K chunks.

**Article Precision@K**

Measures the proportion of retrieved chunks belonging to expected articles.

**Article Recall@K**

Measures the proportion of expected articles represented in retrieved results.

### Abstention Evaluation

Measures whether questions labelled as unanswerable return the expected `ABSTAINED` status.

### End-to-End Evaluation

Tests answer generation, citation verification, and RAG Triad scoring.

End-to-end and abstention evaluations may make Gemini API calls and consume API quota.

---

## 8. Test Results

The following results were observed during English MVP development.

| Test | Result |
|---|---|
| Article extraction | 308/308 articles |
| Explicit article retrieval | 11/11 passed |
| Natural-language retrieval | 9/9 passed |
| Multi-article retrieval examples | 5/5 passed |
| Article reference formats | 6/6 passed |
| Automated abstention cases | 2/2 passed |
| Automated pytest suite | 26/26 passed |

### RAG Triad Example

For a tested comparison of Article 18(1) and Article 38(1):

| Metric | Score |
|---|---:|
| Context Relevance | 1.00 |
| Groundedness | 1.00 |
| Answer Relevance | 1.00 |

Both generated claims were verified against retrieved evidence in that test.

**Note:** These results represent specific tested cases. They do not establish perfect accuracy or security across all possible questions.

### Run Automated Tests

```bash
uv run python -m pytest -q
```

---

## 9. Source Citations

The application returns citations linked to retrieved passages.

Example:

```text
[S1] constitution_en.pdf, PDF page 10
```

For articles spanning multiple pages:

```text
[S2] constitution_en.pdf, PDF pages 17-18
```

Page ranges identify the location of the article within the PDF. They do not necessarily identify the exact page containing each quoted sentence.

---

## 10. Security and Reliability

The application includes several safeguards:

- Input validation
- Article number validation
- Document-restricted retrieval
- Treatment of retrieved text as untrusted data
- Structured answer generation
- Source ID and quotation validation
- Claim-level groundedness verification
- Abstention when evidence is insufficient
- Protection against some common prompt-injection and information-leakage patterns

These safeguards reduce certain failure risks but do not eliminate hallucinations, prompt injection, or incorrect answers.

---

## 11. Current Limitations

- English-only MVP; full Nepali-language question answering is not implemented.
- The built-in Constitution source incorporates amendments through 2020. Later changes have not been verified.
- Article-aware extraction is tailored to the local English Constitution PDF.
- Some long articles are split into multiple chunks.
- Citations may show article page ranges instead of precise quotation pages.
- Retrieval ranking parameters are not fully calibrated.
- Article-reference parsing does not cover every possible format or range.
- The golden dataset needs additional independently reviewed test cases.
- LLM-based evaluation is not a substitute for expert legal review.
- Scanned PDFs require OCR, which is outside the current MVP.
- Vector indexes are stored in memory rather than persistent production storage.
- The application does not provide production-grade authentication, authorization, or multi-user isolation.
- Gemini processes supplied document passages during answer generation.

**Disclaimer:** This application is an educational demonstration and must not be treated as verified, current legal advice.

---

## 12. Future Improvements

Potential improvements after completion of the English MVP:

- Full Nepali-language question answering
- Multilingual embeddings
- Additional Constitution document version support
- Exact quotation-level page attribution
- Larger expert-reviewed evaluation datasets
- Improved handling of multi-article and long-article questions
- Persistent vector database
- Production deployment
- Authentication and multi-user support
- Automated monitoring and evaluation reporting

These improvements are outside the current MVP scope.

---

## 13. Project Status

**English Constitution MVP — Functional, final verification and GitHub preparation in progress.**

The current implementation supports Constitution-based RAG, grounded answer generation, source citations, hallucination checks, G1–G5 guardrails, and an evaluation dashboard.

The MVP is intended for learning, experimentation, and demonstration of practical RAG engineering concepts.