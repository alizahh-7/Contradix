# CONTRADIX
**Semantic Contradiction Detection and Document Consistency Analysis Using Natural Language Processing**

*Beyond Words. Detect the Contradiction.*

CONTRADIX is an NLP-powered semantic analysis platform that identifies **potentially** conflicting information across
statements and documents. It combines **semantic similarity** (Sentence Transformers) with **Natural Language Inference**
(a pretrained DeBERTa cross-encoder) and presents the results in a dark, dashboard-style Streamlit application.
Everything runs locally on CPU - no paid APIs, no API keys.

## Features
| Mode | What it does |
|------|--------------|
| **Text vs Text** | Two statements -> relationship (Entailment / Contradiction / Neutral), the three probabilities, semantic similarity, model-confidence note |
| **Document vs Document** | Upload two PDF / DOCX / TXT files, retrieve relevant sentence pairs, run NLI, show conflicts with page / paragraph references |
| **Text vs Document** | Test a statement against a reference document and inspect the evidence sentences behind each result |

Dashboard sections: **Home** (hero + mode cards) - **Analysis workspace** (mode selector, inputs, Analyze, loading status) -
**Results dashboard** (classification card, probability bars, side-by-side statements, highlighted conflicts, results table,
expandable details) - **Demo playground** (4 predefined pairs) - **Export** (CSV + PDF report) - **About & methodology**.

Red = potential contradiction - Green = entailment - Grey-blue = neutral - Amber = needs review (differing values).
Colours, numbers and labels always come from the real model output.

## Setup (Windows, VS Code, PowerShell, Python 3.11)

Extract the ZIP and open the folder in VS Code:

```powershell
Expand-Archive -Path "$HOME\Downloads\Contradix_Final.zip" -DestinationPath "$HOME\Documents" -Force
cd "$HOME\Documents\Contradix_Final"
code .
```

In the VS Code terminal (Terminal > New Terminal), inside the `Contradix_Final` folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python verify_final.py
python -m pytest -v
python -m streamlit run app.py
```

Then open <http://localhost:8501>.

`verify_final.py` is the final verification script (real models, all three modes, CSV and PDF export). The earlier
`verify_phase1.py` and `verify_phase2.py` still work and are kept.

Fast tests only (no model download): `$env:CONTRADIX_SKIP_MODEL_TESTS="1"; python -m pytest -v`

## Using the app
1. **Text vs Text** - click one of the four demo examples (it only fills the two boxes), or type your own, then press **Analyze**.
   Statement A is the *premise*, Statement B the *hypothesis*. NLI is directional - swapping them can change the answer.
2. **Document vs Document** - upload two files (or press *Load the two bundled sample circulars*), press **Analyze**.
3. **Text vs Document** - enter a statement, upload a reference document (or press *Load demo statement + bundled sample circular*).
4. Tune the retrieval settings in the sidebar (similarity threshold, maximum sentence pairs, matches kept per sentence,
   contradiction threshold, reverse NLI direction). *Reset to defaults* restores them.
5. Use **Download results as CSV** / **Download PDF report** under the results. Both contain the compared statements,
   source references, predictions and scores exactly as displayed.

Results are stored per mode in the session, so expanding rows, filtering, downloading or switching modes never re-runs the
models. Pressing Analyze again with identical inputs and settings reuses the existing result.

## How it works
```
Input -> Text extraction -> Sentence segmentation -> Semantic similarity (MiniLM embeddings)
      -> Relevant-pair retrieval -> NLI (DeBERTa cross-encoder) -> Verdict rules -> Dashboard / CSV / PDF
```
| File | Role |
|------|------|
| `app.py` | Streamlit dashboard: layout, session state, wiring |
| `nlp/preprocessing.py` | spaCy cleaning, sentence splitting, linguistic analysis |
| `nlp/similarity.py` | MiniLM embeddings, cosine similarity (cached model) |
| `nlp/inference.py` | NLI model (cached). `predict_nli_batch` is the only inference code |
| `nlp/pipeline.py` | `compare_statements(a, b)` - Text vs Text |
| `nlp/document_processor.py` | PDF (PyMuPDF) / DOCX (python-docx) / TXT extraction, validation, page + paragraph references |
| `nlp/comparison.py` | `compare_documents`, `compare_text_to_document`, retrieval, verdict rules |
| `utils/results.py` | Reshapes backend output into rows; confidence levels; CSV export |
| `utils/report.py` | PDF report (ReportLab) |
| `utils/components.py`, `utils/styles.py` | HTML components and the custom CSS theme |
| `utils/examples.py` | The four demo pairs (inputs only - never results) |
| `utils/file_handler.py` | Bundled sample documents |
| `sample_data/` | Fictional sample circulars (PDF, DOCX, TXT) |
| `tests/` | Automated tests |
| `.streamlit/config.toml` | Dark theme, upload limit |

### Decision rules (unchanged NLP engine)
* **Retrieval first.** NLI is not run on every sentence combination - only on pairs above the similarity threshold (capped per sentence and overall).
* **Direction is explicit.** Text vs Text: A = premise, B = hypothesis. Document vs Document: sentence from Document A = premise, from Document B = hypothesis.
  Text vs Document: the document's evidence sentence = premise, your statement = hypothesis. Optional reverse NLI is reported separately.
* **Similarity is not agreement.** High similarity never implies entailment.
* **Numbers and dates are not automatic contradictions.** Differing percentages / dates / durations on a shared subject are shown as **Needs review**.
* **Uncertainty is shown.** Every prediction carries a model-confidence level (High / Moderate / Low) and a caution that it is a statistical prediction.
* **No evidence -> no verdict.**

### Performance
* Both models are loaded once per process (`lru_cache`) and reused by every Streamlit rerun and session; the sidebar shows whether each is loaded.
* Uploaded files are extracted once (`st.cache_data`), and identical analyses are not recomputed.
* CPU only; the sentence-pair cap keeps document comparison practical.

## NLP syllabus mapping
* **Unit I - Finding the Structure of Documents:** PDF/DOCX/TXT extraction, paragraph and sentence segmentation, page/paragraph references.
* **Unit III - Semantic Interpretation and Ambiguity Resolution:** NLI between sentences, negation handled by a contextual model, similarity vs entailment, condition/scope cues, uncertainty reporting.
* **Unit IV - Meaning Representation:** dense sentence embeddings (vector meaning representation) and premise-hypothesis proposition testing.

## Sample documents (`sample_data/`) - FICTIONAL TEST DATA
| File | Content |
|------|---------|
| `circular_a_original_policy.pdf` (+ `.txt`) | 75% attendance, deadline November 15, individual submission, no late submissions |
| `circular_b_revised_notice.docx` (+ `.txt`) | 65% attendance, deadline November 20, teams of two, late submission within three days |

## Limits
* Scanned / image-only PDFs need OCR and are not supported (a clear message is shown).
* Max upload 10 MB per file; at most 3000 sentences per document are used.
* The NLI model is small and can be wrong - results are *potential* contradictions for a human to review.
* Thresholds are defaults, not tuned values.
* Contradictions that depend on several sentences of context or outside knowledge can be missed.
* PDF report text uses a built-in Latin font; characters outside Latin-1 are replaced with `?` in the PDF (the CSV keeps full Unicode).

## Troubleshooting
* **PowerShell blocks activation** ("running scripts is disabled"): run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then `.\.venv\Scripts\Activate.ps1` again. Or use `.\.venv\Scripts\python.exe -m streamlit run app.py` without activating.
* **`py -3.11` not found:** install Python 3.11 from python.org (tick "Add python.exe to PATH"), or use `python -m venv .venv` if `python --version` already shows 3.10/3.11.
* **First run is slow / seems stuck:** on first use the two models (a few hundred MB) are downloaded from Hugging Face into `C:\Users\<you>\.cache\huggingface`. Keep the internet on; later runs are offline and fast. The first analysis in the app shows a "Running the NLP models" status while this happens.
* **Model download fails** (proxy / firewall / offline): check your connection and retry; if Hugging Face is blocked on your network, download once on another network - the cache folder can be copied.
* **`en_core_web_sm` missing:** `python -m spacy download en_core_web_sm` (the app falls back to a basic sentence splitter without it).
* **A `torch.jit.script` FutureWarning** appears in test output: it comes from a dependency and is harmless.
* **Port 8501 already in use:** `python -m streamlit run app.py --server.port 8502`.
* Always start the app with `python -m streamlit run app.py` so the virtual environment's Python is used.
