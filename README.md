# CONTRADIX

### Beyond Words. Detect the Contradiction.

**Semantic Contradiction Detection and Document Consistency Analysis Using Natural Language Processing**

CONTRADIX is an NLP-powered semantic analysis platform designed to identify potentially conflicting information across statements and documents.

It combines semantic similarity, sentence-level evidence retrieval, and Natural Language Inference (NLI) to examine whether two pieces of information agree, contradict one another, or have no clear logical relationship.

Built with Python and Streamlit, CONTRADIX provides an interactive analysis workspace, explainable results, source references, and downloadable reports.

> **CONTRADIX does not claim to determine absolute truth.** It identifies potential contradictions and presents supporting evidence for human review.

---

## Overview

Documents often contain overlapping information, revised policies, changing figures, or statements that appear similar but communicate opposing meanings.

Traditional keyword matching is insufficient for identifying these relationships.

For example:

**Statement A:** The university library is open every Sunday.

**Statement B:** The university library remains closed on all Sundays.

These statements are semantically similar because they discuss the same subject. However, their meanings directly contradict one another.

CONTRADIX addresses this distinction by combining two complementary NLP techniques:

- **Semantic Similarity:** Identifies sentences discussing related information.
- **Natural Language Inference:** Determines whether one statement entails, contradicts, or is neutral toward another.

This approach allows the system to retrieve relevant evidence before performing more detailed contextual analysis.

---

## Features

### Three Analysis Modes

| Mode | Description |
|---|---|
| Text vs Text | Compare two statements and classify their logical relationship. |
| Document vs Document | Upload two documents, retrieve relevant sentence pairs, and identify potential conflicts. |
| Text vs Document | Evaluate a statement against evidence extracted from a reference document. |

### Additional Capabilities

- Semantic similarity using sentence embeddings.
- Contextual inference using a pretrained DeBERTa model.
- PDF, DOCX, and TXT document support.
- Sentence segmentation and basic linguistic analysis using spaCy.
- Source references, including page and paragraph information where available.
- Confidence indicators and probability distributions.
- Detection of potentially conflicting numerical values and dates.
- Interactive demo examples.
- Expandable evidence comparisons.
- CSV and PDF report generation.
- Cached model loading and analysis results.
- Local model inference without paid APIs or API keys.

---

## Interface

CONTRADIX features a minimalist, sumi-e-inspired landing page with an ink-wash mountain landscape, muted colors, and a streamlined entry point.

The interface separates the introductory experience from the actual analysis environment.

### Landing Page

Introduces CONTRADIX through its central message:

**Beyond Words. Detect the Contradiction.**

The *Start analyzing* button opens the analysis workspace.

### Analysis Workspace

Provides access to all three comparison modes, input fields, document uploads, demo examples, and generated results.

### About & Methodology

Explains the underlying NLP concepts, semantic similarity, Natural Language Inference, interpretation of results, technologies, and known limitations.

---

## How CONTRADIX Works

The system follows a retrieval-first NLP pipeline.

```text
                  USER INPUT
                      |
                      v
              Text / PDF / DOCX
                      |
                      v
              Text Extraction
                      |
                      v
           Sentence Segmentation
                      |
                      v
          Semantic Representation
              (MiniLM Embeddings)
                      |
                      v
             Cosine Similarity
                      |
                      v
          Relevant Pair Retrieval
                      |
                      v
        Natural Language Inference
          (DeBERTa Cross-Encoder)
                      |
                      v
           Classification & Rules
                      |
                      v
             Analysis Dashboard
                      |
                      v
              CSV / PDF Export
```

### 1. Input Processing

Users provide statements or upload supported documents.

The system extracts text and structures it into sentences while retaining source information where available.

### 2. Semantic Representation

A pretrained Sentence-Transformers model converts sentences into dense numerical embeddings.

These embeddings represent semantic information, allowing the system to compare sentences beyond exact keyword overlap.

### 3. Relevant Evidence Retrieval

Cosine similarity is used to identify related sentence pairs.

Rather than evaluating every possible sentence combination with the NLI model, CONTRADIX selects relevant candidates for further analysis.

### 4. Natural Language Inference

A pretrained DeBERTa cross-encoder examines each selected premise–hypothesis pair and predicts three relationships:

- Entailment
- Contradiction
- Neutral

### 5. Interpretation

The application combines model predictions with its classification and review rules.

Potential contradictions, uncertain relationships, and differing numerical values are presented for inspection.

### 6. Reporting

Results can be examined interactively or exported as CSV and PDF reports.

---

## NLP Models

### Semantic Similarity

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

MiniLM converts sentences into 384-dimensional embeddings.

Sentence similarity is calculated using cosine similarity:

```text
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

A high similarity score indicates that two statements discuss related information. It does not necessarily mean that they agree.

### Natural Language Inference

**Model:** `cross-encoder/nli-deberta-v3-small`

The NLI model processes a premise and hypothesis together to estimate their logical relationship.

| Relationship | Meaning |
|---|---|
| Entailment | The hypothesis follows from the premise. |
| Contradiction | The hypothesis conflicts with the premise. |
| Neutral | The premise neither establishes nor disproves the hypothesis. |

**NLI is directional.** Reversing the premise and hypothesis can change the prediction.

---

## Analysis Modes

### 1. Text vs Text

Compare two individual statements.

**Example**

Statement A:

```text
The university library is open every Sunday.
```

Statement B:

```text
The university library remains closed on all Sundays.
```

Expected human interpretation: **Contradiction**

The interface displays the actual model prediction, including:

- Predicted relationship.
- Entailment probability.
- Contradiction probability.
- Neutral probability.
- Semantic similarity score.
- Model-confidence indicator.
- Side-by-side statement comparison.

The demo playground also provides predefined examples for quickly exploring different logical relationships.

### 2. Document vs Document

Upload two supported documents and compare their contents.

Supported formats:

| Format | Processing Library |
|---|---|
| PDF | PyMuPDF |
| DOCX | python-docx |
| TXT | Python text processing |

CONTRADIX extracts sentences, identifies relevant candidate pairs, performs NLI, and presents potentially conflicting information with source references.

This mode is useful for exploring differences between revised notices, policy documents, circulars, and other text-based materials.

Bundled fictional sample circulars are available for demonstration.

### 3. Text vs Document

Enter a statement and upload a reference document.

CONTRADIX retrieves relevant evidence sentences from the document and evaluates their relationship with the supplied statement.

In this mode:

- The document evidence acts as the premise.
- The user-entered statement acts as the hypothesis.

This makes it possible to inspect whether a claim is supported, contradicted, or not established by the available document evidence.

---

## Understanding the Results

CONTRADIX distinguishes between semantic relevance and logical agreement.

| Status | Interpretation |
|---|---|
| Potential Contradiction | The model identifies a potentially conflicting relationship. |
| Needs Review | Related statements contain differences requiring further inspection. |
| Consistent | The model predicts entailment. |
| Neutral | No clear logical relationship was identified. |

Numerical differences, percentages, dates, and durations are not automatically treated as contradictions.

For example, two policies may contain different deadlines because one is a revised version of the other.

Such findings require contextual interpretation.

The application also displays confidence indicators. These represent model confidence, not independently verified factual certainty.

---

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python 3.11 |
| Application Framework | Streamlit |
| NLP Processing | spaCy |
| Sentence Embeddings | Sentence-Transformers |
| Semantic Similarity Model | all-MiniLM-L6-v2 |
| NLI Model | nli-deberta-v3-small |
| Model Framework | PyTorch |
| Transformer Models | Hugging Face Transformers |
| PDF Extraction | PyMuPDF |
| DOCX Extraction | python-docx |
| PDF Report Generation | ReportLab |
| Testing | pytest |
| Interface | Custom HTML and CSS |

The application uses locally executed pretrained models. An internet connection is required for their initial download.

---

## Getting Started

### Prerequisites

Before installation, ensure you have:

- Python 3.11.
- Git.
- A working internet connection for the initial model download.
- A terminal or IDE such as Visual Studio Code.

### 1. Clone the Repository

```bash
git clone https://github.com/alizahh-7/Contradix.git
cd Contradix
```

> This repository is currently private. GitHub authentication and repository access are required.

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the spaCy Language Model

```powershell
python -m spacy download en_core_web_sm
```

### 5. Launch CONTRADIX

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

The application initially displays the landing page.

Select **Start analyzing** to enter the workspace.

**Note:** The first analysis may take longer because the pretrained models must be downloaded and initialized.

---

## Running Tests

CONTRADIX includes automated tests and dedicated verification scripts.

### Run the Test Suite

```powershell
python -m pytest -v
```

### Run Final Verification

```powershell
python verify_final.py
```

The final verification script exercises the NLP pipeline, comparison modes, and report generation.

### Fast Tests Without Model Downloads

```powershell
$env:CONTRADIX_SKIP_MODEL_TESTS="1"
python -m pytest -v
```

This skips model-dependent tests while allowing faster checks of other functionality.

### Additional Verification Scripts

```powershell
python verify_phase1.py
python verify_phase2.py
```

These scripts are retained for earlier development-stage verification.

---

## Project Structure

```text
Contradix/
│
├── .streamlit/
│   └── config.toml
│
├── nlp/
│   ├── preprocessing.py
│   ├── similarity.py
│   ├── inference.py
│   ├── pipeline.py
│   ├── document_processor.py
│   └── comparison.py
│
├── utils/
│   ├── components.py
│   ├── styles.py
│   ├── examples.py
│   ├── file_handler.py
│   ├── results.py
│   └── report.py
│
├── sample_data/
├── tests/
│
├── app.py
├── requirements.txt
├── pytest.ini
├── verify_phase1.py
├── verify_phase2.py
├── verify_final.py
├── .gitignore
└── README.md
```

### Key Components

| File | Responsibility |
|---|---|
| `app.py` | Streamlit interface, application state, and backend integration. |
| `nlp/preprocessing.py` | Text preprocessing and linguistic analysis. |
| `nlp/similarity.py` | Sentence embeddings and cosine similarity. |
| `nlp/inference.py` | Natural Language Inference model loading and predictions. |
| `nlp/pipeline.py` | Statement-level comparison. |
| `nlp/document_processor.py` | Document extraction, validation, and source references. |
| `nlp/comparison.py` | Document comparison, retrieval, and classification rules. |
| `utils/components.py` | Reusable interface components. |
| `utils/styles.py` | Application styling. |
| `utils/results.py` | Result formatting and CSV export. |
| `utils/report.py` | PDF report generation. |
| `utils/examples.py` | Demonstration statements. |
| `tests/` | Automated application tests. |

---

## Retrieval Configuration

Document comparison uses fixed internal retrieval settings rather than exposing configuration controls in the interface.

| Parameter | Value |
|---|---:|
| Similarity threshold | 0.43 |
| Maximum sentence pairs | 25 |
| Matches retained per sentence | 2 |
| Contradiction threshold | 0.29 |
| Bidirectional NLI | Enabled |

These values control candidate selection and document comparison behavior.

They are implementation settings, not guarantees of accuracy across every document type or domain.

---

## Performance and Caching

CONTRADIX includes several optimizations to reduce unnecessary processing:

- Pretrained models are cached and reused after loading.
- Extracted document content is cached.
- Identical analyses can reuse existing results.
- Results are maintained separately for each comparison mode during a session.
- Document comparison limits the number of sentence pairs passed to NLI.
- Expanding results, changing display filters, or downloading reports does not require repeating inference on unchanged inputs.

These optimizations help keep local execution practical.

---

## NLP Concepts Implemented

CONTRADIX also demonstrates several concepts from an academic Natural Language Processing syllabus.

### Unit I — Finding the Structure of Documents

- Text extraction from PDF, DOCX, and TXT.
- Sentence and paragraph segmentation.
- Basic linguistic preprocessing.
- Preservation of source references.

### Unit III — Semantic Interpretation and Ambiguity Resolution

- Contextual interpretation through NLI.
- Negation-sensitive sentence comparison.
- Distinguishing semantic similarity from entailment.
- Identification of potentially conflicting conditions and values.
- Reporting uncertain predictions.

### Unit IV — Meaning Representation

- Dense vector representations of sentences.
- Distributional semantic similarity.
- Cosine-based vector comparison.
- Premise–hypothesis relationship classification.

The system uses vector-based semantic representations rather than formal symbolic logic.

---

## Sample Documents

The repository includes fictional documents for testing and demonstration.

| Sample | Example Content |
|---|---|
| Original Policy Circular | 75% attendance requirement, November 15 deadline, individual submissions. |
| Revised Notice | 65% attendance requirement, November 20 deadline, teams of two, revised late-submission conditions. |

These documents allow users to explore how CONTRADIX retrieves related sentences and identifies potential policy inconsistencies.

**All bundled sample content is fictional and intended solely for testing.**

---

## Known Limitations

CONTRADIX is an assistive semantic analysis tool, not a factual verification authority.

Important limitations include:

1. **Model uncertainty:** Small pretrained language models can produce incorrect predictions, including high-confidence errors.
2. **Context dependency:** Exceptions, dates, conditions, and surrounding paragraphs can change a statement's interpretation.
3. **Multi-sentence reasoning:** Contradictions requiring information distributed across several sentences may be missed.
4. **Document extraction:** Scanned or image-only PDFs require OCR, which is not included.
5. **Retrieval limitations:** Candidate-pair filtering can miss relevant statements when their semantic similarity falls below the configured threshold.
6. **Numerical interpretation:** Different values do not necessarily represent contradictions.
7. **External knowledge:** The system does not independently verify statements against real-world facts.
8. **PDF character support:** Some non-Latin characters may not render correctly in generated PDF reports.

Results should always be reviewed alongside their source material.

---

## Troubleshooting

**The application takes a long time during its first analysis.**

The pretrained models are downloaded on first use. Keep your internet connection active and allow the download to complete.

**The spaCy language model is missing.**

Run:

```powershell
python -m spacy download en_core_web_sm
```

**PowerShell prevents virtual environment activation.**

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**Port 8501 is already in use.**

Launch the application on another port:

```powershell
python -m streamlit run app.py --server.port 8502
```

**Model downloads fail.**

Check your internet connection and ensure Hugging Face is accessible from your network.

**The application uses the wrong Python environment.**

Activate the virtual environment and launch Streamlit using:

```powershell
python -m streamlit run app.py
```

---

## Project Scope

CONTRADIX explores how semantic retrieval and Natural Language Inference can work together to support document consistency analysis.

Its central objective is not merely to identify similar sentences, but to investigate whether semantically related information expresses compatible or conflicting meanings.

The project demonstrates a practical application of pretrained NLP models through an interactive system that connects document processing, semantic representation, contextual inference, and explainable reporting.

---

## Author

**Syeda Alizah**

Computer Science and Engineering

GitHub: [@alizahh-7](https://github.com/alizahh-7)

Project Repository: [CONTRADIX](https://github.com/alizahh-7/Contradix)

---

**CONTRADIX — Beyond Words. Detect the Contradiction.**
