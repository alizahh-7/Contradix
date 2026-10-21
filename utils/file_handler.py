"""File helpers used by the Streamlit UI: uploads and bundled sample documents."""
from __future__ import annotations

from pathlib import Path

from nlp.document_processor import (
    SUPPORTED_EXTENSIONS,
    ExtractedDocument,
    extract_document,
    extract_document_from_path,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_ROOT / "sample_data"

# Bundled FICTIONAL test documents (Circular A = PDF, Circular B = DOCX)
SAMPLE_CIRCULAR_A = SAMPLE_DIR / "circular_a_original_policy.pdf"
SAMPLE_CIRCULAR_B = SAMPLE_DIR / "circular_b_revised_notice.docx"

UPLOAD_TYPES = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]


def load_uploaded_file(uploaded_file) -> ExtractedDocument:
    """Turn a Streamlit UploadedFile into an ExtractedDocument (raises DocumentError)."""
    return extract_document(uploaded_file.name, uploaded_file.getvalue())


def load_sample(path: Path) -> ExtractedDocument:
    """Load one of the bundled sample documents (raises DocumentError / FileNotFoundError)."""
    return extract_document_from_path(str(path))
