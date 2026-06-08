"""Text normalization helpers for extracted PDF content."""

from __future__ import annotations

import re
import unicodedata

# Patterns that commonly appear as repeated noise in PDF-extracted chunks:
# page numbers, running headers/footers, table-of-contents dotted lines, etc.
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),                   # standalone page numbers
    re.compile(r"\.{4,}", re.MULTILINE),                         # dotted lines (TOC)
    re.compile(r"_{4,}", re.MULTILINE),                          # underline separators
    re.compile(r"-{4,}", re.MULTILINE),                          # dash separators
    re.compile(r"={4,}", re.MULTILINE),                          # equals separators
    re.compile(r"\*{4,}", re.MULTILINE),                         # asterisk separators
    re.compile(r"\[\s*\d+\s*\]"),                                # citation markers [1]
    re.compile(r"https?://\S+"),                                  # bare URLs
    re.compile(r"www\.\S+"),                                      # bare www links
]


def clean_text(text: str) -> str:
    """
    Normalize extracted PDF text for chunking and retrieval.

    Removes excessive whitespace, repairs common PDF line wrapping,
    normalizes line breaks, and keeps paragraph boundaries where possible.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"-\n(?=\w)", "", normalized)
    normalized = re.sub(r"(?<![.!?;:])\n(?!\n)", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r" {2,}", " ", normalized)

    return normalized.strip()


def clean_chunk(chunk: str) -> str:
    """
    Deep-clean a single text chunk before it is embedded.

    Applied after chunking so each piece is independently cleaned without
    disturbing sentence boundaries detected during the chunking step.

    Operations (in order):
        1. Normalize Unicode — convert fancy quotes, ligatures, etc. to ASCII
        2. Remove non-breaking spaces and zero-width characters
        3. Strip noise patterns (page numbers, TOC lines, separators, URLs)
        4. Collapse leftover whitespace
        5. Drop the chunk if it is pure boilerplate (< 40 meaningful chars)

    Args:
        chunk: A single raw text chunk.

    Returns:
        A cleaned chunk string, or an empty string if the chunk is noise-only.
    """
    if not chunk or not chunk.strip():
        return ""

    # 1. Unicode normalisation — NFKC maps ligatures (ﬁ→fi) and fancy
    #    punctuation (\u2019→') to their standard ASCII equivalents.
    cleaned = unicodedata.normalize("NFKC", chunk)

    # 2. Remove zero-width / invisible characters that survive PDF extraction
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", cleaned)

    # 3. Apply every noise pattern
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    # 4. Collapse runs of whitespace created by the removals above
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    # 5. Discard if the chunk no longer carries meaningful content
    meaningful_chars = re.sub(r"\s+", "", cleaned)
    if len(meaningful_chars) < 40:
        return ""

    return cleaned


def clean_chunks(chunks: list[str]) -> list[str]:
    """
    Apply clean_chunk() to every chunk and remove empty results.

    Args:
        chunks: Raw chunk list produced by chunk_text().

    Returns:
        Filtered list of non-empty cleaned chunks.
    """
    return [c for raw in chunks if (c := clean_chunk(raw))]
