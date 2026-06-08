"""Sentence-aware text chunking utilities."""

from __future__ import annotations

import re


def _find_sentence_boundary(text: str, start: int, end: int) -> int:
    """Find a sentence boundary near the desired chunk end."""
    boundary_pattern = re.compile(r"[.!?]\s+")
    minimum_useful_end = start + max(1, int((end - start) * 0.6))
    search_start = max(minimum_useful_end, end - 250)
    candidates = [
        match.end()
        for match in boundary_pattern.finditer(text, search_start, min(len(text), end + 150))
        if match.end() >= minimum_useful_end
    ]

    before_end = [candidate for candidate in candidates if candidate <= end]
    if before_end:
        return before_end[-1]

    if candidates:
        return candidates[0]

    whitespace = text.rfind(" ", minimum_useful_end, end)
    if whitespace >= minimum_useful_end:
        return whitespace + 1

    return min(end, len(text))


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping character chunks.

    Args:
        text: Cleaned document text.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Number of characters repeated between adjacent chunks.

    Returns:
        A list of non-empty text chunks.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized = text.strip()
    chunks: list[str] = []
    start = 0

    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = (
            len(normalized)
            if target_end == len(normalized)
            else _find_sentence_boundary(normalized, start, target_end)
        )

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(normalized):
            break

        next_start = max(end - chunk_overlap, start + 1)
        if next_start > 0 and not normalized[next_start - 1].isspace():
            previous_space = normalized.rfind(" ", start, next_start)
            previous_newline = normalized.rfind("\n", start, next_start)
            word_boundary = max(previous_space, previous_newline)
            if word_boundary >= start:
                next_start = word_boundary + 1

        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks
