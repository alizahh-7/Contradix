"""CONTRADIX NLP package.

Public API:
    compare_statements(text_a, text_b)                 Phase 1 - Text vs Text
    compare_documents(doc_a, doc_b, config)            Phase 2 - Document vs Document
    compare_text_to_document(statement, doc, config)   Phase 2 - Text vs Document
    extract_document(filename, data)                   Phase 2 - PDF / DOCX / TXT extraction
"""
from nlp import inference, similarity  # noqa: F401  (exposed for model-status display)
from nlp.comparison import ComparisonConfig, compare_documents, compare_text_to_document
from nlp.document_processor import extract_document, extract_document_from_path
from nlp.pipeline import compare_statements

__all__ = [
    "compare_statements", "compare_documents", "compare_text_to_document",
    "ComparisonConfig", "extract_document", "extract_document_from_path",
]
