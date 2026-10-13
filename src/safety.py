"""Small, explicit guardrails. These are demo rules, not a full security boundary."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# G1: deterministic checks for obvious prompt override / credential requests.
INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all |any )?(?:previous|prior) instructions", re.I),
    re.compile(r"(?:reveal|print|show) (?:the |your )?(?:system|developer) prompt", re.I),
    re.compile(r"(?:disable|bypass) (?:all |your )?(?:guardrails|safety checks)", re.I),
]
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
SECRET = re.compile(
    r"(?i)(?:\bBearer\s+[A-Za-z0-9_.-]{10,}|\bAIza[A-Za-z0-9_-]{20,}|\bsk-[A-Za-z0-9_-]{20,})"
)


@dataclass
class InputCheck:
    allowed: bool
    cleaned: str = ""
    reasons: list[str] = field(default_factory=list)


def check_input(question: str) -> InputCheck:
    """Reject obvious injection/credentials; redact email from external queries."""
    q = question.strip()
    if not (3 <= len(q) <= 500):
        return InputCheck(False, reasons=["G1: Question must be 3–500 characters."])
    if SECRET.search(q):
        return InputCheck(False, reasons=["G1: Credential-like string detected."])
    if any(pattern.search(q) for pattern in INJECTION_PATTERNS):
        return InputCheck(False, reasons=["G1: Prompt-override pattern detected."])
    cleaned = EMAIL.sub("[REDACTED_EMAIL]", q)
    reasons = ["G1: Email address redacted."] if cleaned != q else ["G1: Input passed."]
    return InputCheck(True, cleaned, reasons)


def normalize(s: str) -> str:
    """Whitespace-insensitive matching for an exact quoted source span."""
    return " ".join(s.split())


def citation_exists(quote: str, source_text: str) -> bool:
    """G5: quoted evidence must literally appear in its cited chunk."""
    return bool(quote and len(normalize(quote)) >= 8 and
                normalize(quote) in normalize(source_text))


def output_has_leak(text: str) -> bool:
    # G5: basic detector. Other personal data needs a stronger policy/detector.
    return bool(SECRET.search(text) or EMAIL.search(text))


def retrieval_sufficient(best_cosine: float, threshold: float = 0.22) -> bool:
    """Demo relevance floor; calibrate on labelled queries before production."""
    return best_cosine >= threshold
