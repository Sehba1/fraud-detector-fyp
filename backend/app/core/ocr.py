"""
OCR module for extracting text from image-based job postings.
Uses Tesseract via pytesseract with a hardcoded executable path
(more reliable than PATH-based detection on Windows).
"""

import os
import pytesseract
from PIL import Image
import cv2
import numpy as np

# Hardcode Tesseract path — bypasses Windows PATH issues entirely
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image_for_ocr(image_path: str) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy.
    Converts to grayscale and applies binary thresholding.
    Common job posting screenshots have compressed JPEG artifacts
    that thresholding helps remove.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at path: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using Tesseract OCR.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        Raw extracted text string. May contain OCR errors on
        low-quality or compressed images — caller should pass
        this through clean_text() before model inference.

    Raises:
        ValueError: If the image cannot be read.
        RuntimeError: If Tesseract is not found at the configured path.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")

    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        raise RuntimeError(
            f"Tesseract not found at: {pytesseract.pytesseract.tesseract_cmd}\n"
            "Please verify Tesseract is installed at C:\\Program Files\\Tesseract-OCR\\"
        )

    try:
        preprocessed = preprocess_image_for_ocr(image_path)
        raw_text = pytesseract.image_to_string(preprocessed)
        return raw_text.strip()
    except Exception as e:
        raise RuntimeError(f"OCR extraction failed: {e}")


def extract_text_from_bytes(image_bytes: bytes, suffix: str = ".jpg") -> str:
    """
    Extract text from image bytes (used when receiving file uploads via FastAPI).
    Saves to a temporary file, extracts text, then cleans up.

    Args:
        image_bytes: Raw image bytes from the uploaded file.
        suffix: File extension hint for the temp file (.jpg, .png, etc.)

    Returns:
        Raw extracted text string.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        text = extract_text_from_image(tmp_path)
    finally:
        os.unlink(tmp_path)  # always clean up temp file

    return text