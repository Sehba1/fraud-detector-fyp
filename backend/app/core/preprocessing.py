"""
Shared text preprocessing pipeline for the Fraud Detector backend.

Used by both the text-input path (/predict/text) and the
OCR-output path (/predict/image) so both go through identical
cleaning before reaching the BERT model.

Mirrors the clean_text() function validated in ml/notebooks/01_eda.ipynb.
"""

import re
from bs4 import BeautifulSoup

# Pre-compiled regex patterns for efficiency
# Compiled once at module load, reused for every request
_PLACEHOLDER_PATTERN = re.compile(r"#[a-z]+_[0-9a-f]+#")
_ESCAPED_NEWLINES = re.compile(r"\\r\\n|\\n|\\r")
_URLS = re.compile(r"http\S+|www\S+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean and normalise a job posting text for BERT inference.

    Steps (in order):
    1. BeautifulSoup HTML parsing — strips tags and discards
       <script>/<style> block contents entirely
    2. Lowercase normalisation
    3. EMSCAD placeholder removal (#URL_hash#, #EMAIL_hash#, #PHONE_hash#)
    4. Escaped newline removal
    5. URL removal
    6. Punctuation and digit removal
    7. Whitespace collapsing

    Args:
        text: Raw job posting text (may contain HTML, placeholders, URLs).

    Returns:
        Cleaned lowercase text string, ready for tokenization.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Parse HTML, discard script/style contents
    soup = BeautifulSoup(text, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    text = soup.get_text(separator=" ")

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Remove EMSCAD anonymization placeholders
    text = _PLACEHOLDER_PATTERN.sub(" ", text)

    # Step 4: Remove literal escaped newline sequences
    text = _ESCAPED_NEWLINES.sub(" ", text)

    # Step 5: Remove URLs
    text = _URLS.sub(" ", text)

    # Step 6: Remove punctuation and digits
    text = _NON_ALPHA.sub(" ", text)

    # Step 7: Collapse whitespace
    text = _MULTI_SPACE.sub(" ", text).strip()

    return text


def clean_texts(texts: list) -> list:
    """
    Clean a list of texts. Convenience wrapper around clean_text().

    Args:
        texts: List of raw text strings.

    Returns:
        List of cleaned text strings.
    """
    return [clean_text(t) for t in texts]