"""Conversation memory manager — stores chat history and builds prompt context."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

from src.config import MEMORY_CONTEXT_WINDOW


@dataclass
class ChatTurn:
    """A single question-answer exchange."""

    question: str
    answer: str
    source: str                      # "document" | "general_knowledge"
    doc_names: list[str]             # which PDFs were active during this turn
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


class MemoryManager:
    """
    Manages conversational memory for multi-turn RAG sessions.

    Responsibilities:
        - Store each Q&A turn with metadata (source, active docs, timestamp)
        - Build a condensed history string to inject into Gemini prompts
          so the model has context from previous turns
        - Export full history to JSON or plain text for download
        - Clear memory on demand

    The context window is deliberately kept small (default 3 turns) to avoid
    pushing the conversation history past Gemini's context limit.
    """

    def __init__(self, context_window: int = MEMORY_CONTEXT_WINDOW) -> None:
        self._turns: list[ChatTurn] = []
        self._context_window = context_window

    # ── Mutations ─────────────────────────────────────────────────────────────

    def add_turn(
        self,
        question: str,
        answer: str,
        source: str,
        doc_names: list[str],
    ) -> None:
        """Append a completed Q&A turn to memory."""
        self._turns.append(
            ChatTurn(
                question=question,
                answer=answer,
                source=source,
                doc_names=doc_names,
            )
        )

    def clear(self) -> None:
        """Wipe all stored turns."""
        self._turns.clear()

    # ── Read-only access ──────────────────────────────────────────────────────

    @property
    def turns(self) -> list[ChatTurn]:
        """All stored turns in chronological order."""
        return list(self._turns)

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    # ── Prompt context builder ────────────────────────────────────────────────

    def build_context_string(self) -> str:
        """
        Build a compact conversation history string for injection into prompts.

        Only the most recent `context_window` turns are included to keep the
        injected text short and within token budgets.

        Returns:
            Multi-line string of prior Q&A pairs, or empty string if no history.
        """
        if not self._turns:
            return ""

        recent = self._turns[-self._context_window :]
        lines: list[str] = ["Previous conversation:"]

        for turn in recent:
            lines.append(f"User: {turn.question}")
            # Truncate long answers so history doesn't dominate the prompt
            answer_preview = (
                turn.answer[:300] + "..." if len(turn.answer) > 300 else turn.answer
            )
            lines.append(f"Assistant: {answer_preview}")

        return "\n".join(lines)

    # ── Export ────────────────────────────────────────────────────────────────

    def export_as_json(self) -> str:
        """Serialize full chat history to a JSON string for download."""
        data = [
            {
                "turn": i + 1,
                "timestamp": t.timestamp,
                "question": t.question,
                "answer": t.answer,
                "source": t.source,
                "documents": t.doc_names,
            }
            for i, t in enumerate(self._turns)
        ]
        return json.dumps(data, indent=2, ensure_ascii=False)

    def export_as_text(self) -> str:
        """Serialize full chat history to plain text for download."""
        lines: list[str] = [
            "=== PDF RAG Chat History ===",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        for i, turn in enumerate(self._turns, start=1):
            lines += [
                f"[Turn {i}] {turn.timestamp}",
                f"Documents: {', '.join(turn.doc_names) or 'none'}",
                f"Q: {turn.question}",
                f"A ({turn.source}): {turn.answer}",
                "",
            ]
        return "\n".join(lines)
