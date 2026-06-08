"""PDF ingestion utilities for uploaded documents."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import BinaryIO

import PyPDF2
from PyPDF2.errors import PdfReadError


@dataclass(frozen=True)
class PDFExtractionResult:
    """Structured result returned after extracting text from a PDF."""

    text: str
    page_count: int
    character_count: int
    # tuples instead of lists — frozen=True prevents field reassignment but
    # cannot stop list.append() mutation; tuples are truly immutable.
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def has_text(self) -> bool:
        """Return True when the PDF produced readable text."""
        return bool(self.text.strip())


def _read_pdf_bytes(pdf_file: bytes | bytearray | BinaryIO) -> bytes:
    """Convert Streamlit uploads, file-like objects, or raw bytes into bytes."""
    if isinstance(pdf_file, bytes):
        return pdf_file

    if isinstance(pdf_file, bytearray):
        return bytes(pdf_file)

    try:
        current_position = pdf_file.tell()
    except (AttributeError, OSError):
        current_position = None

    raw_bytes = pdf_file.read()

    if current_position is not None:
        try:
            pdf_file.seek(current_position)
        except OSError:
            pass

    return raw_bytes


def extract_text_from_pdf(pdf_file: bytes | bytearray | BinaryIO) -> PDFExtractionResult:
    """
    Extract text from all pages of a PDF.

    The function accepts raw bytes or a Streamlit UploadedFile object. It handles
    corrupted PDFs, encrypted files, empty uploads, and pages with no extractable
    text without crashing the Streamlit app.
    """
    errors: list[str] = []
    warnings: list[str] = []

    try:
        pdf_bytes = _read_pdf_bytes(pdf_file)
    except Exception as exc:
        return PDFExtractionResult(
            text="",
            page_count=0,
            character_count=0,
            errors=(f"Could not read uploaded file: {exc}",),
        )

    if not pdf_bytes:
        return PDFExtractionResult(
            text="",
            page_count=0,
            character_count=0,
            errors=("The uploaded PDF is empty.",),
        )

    try:
        reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
    except PdfReadError as exc:
        return PDFExtractionResult(
            text="",
            page_count=0,
            character_count=0,
            errors=(f"The uploaded file is not a readable PDF: {exc}",),
        )
    except Exception as exc:
        return PDFExtractionResult(
            text="",
            page_count=0,
            character_count=0,
            errors=(f"Unexpected PDF parsing error: {exc}",),
        )

    if getattr(reader, "is_encrypted", False):
        try:
            decrypt_result = reader.decrypt("")
            if decrypt_result == 0:
                return PDFExtractionResult(
                    text="",
                    page_count=0,
                    character_count=0,
                    errors=("The PDF is encrypted and could not be opened.",),
                )
        except Exception as exc:
            return PDFExtractionResult(
                text="",
                page_count=0,
                character_count=0,
                errors=(f"The PDF is encrypted and could not be opened: {exc}",),
            )

    page_count = len(reader.pages)
    page_text: list[str] = []
    warnings: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"Page {page_number} could not be extracted: {exc}")
            continue

        if extracted.strip():
            page_text.append(extracted)
        else:
            warnings.append(f"Page {page_number} did not contain readable text.")

    text = "\n\n".join(page_text)

    if page_count > 0 and not text.strip():
        warnings.append(
            "No readable text was found. The PDF may be scanned or image-based."
        )

    return PDFExtractionResult(
        text=text,
        page_count=page_count,
        character_count=len(text),
        errors=(),
        warnings=tuple(warnings),
    )
