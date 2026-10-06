"""Text normalization and WER/CER metrics (standard library only)."""

import unicodedata
from collections.abc import Sequence

_APOSTROPHES = {"'", "\u2019", "\u2018", "`"}


def normalize_text(text: str) -> str:
    """Normalize a transcript for scoring.

    Policy (applied identically to references and hypotheses):

    1. Lowercase.
    2. Apostrophes (``'``, ``’``, ``‘``, `` ` ``) are deleted, so "don't" -> "dont".
    3. Every other punctuation or symbol character (Unicode categories P* and S*)
       is replaced by a space, so "well-known" -> "well known".
    4. Runs of whitespace collapse to one space; leading/trailing space is trimmed.

    Letters and digits (including non-ASCII letters) are preserved.
    Raises ValueError if ``text`` is not a ``str``.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a str, got {type(text).__name__}")
    chars = []
    for ch in text.lower():
        if ch in _APOSTROPHES:
            continue
        if unicodedata.category(ch)[0] in ("P", "S"):
            chars.append(" ")
        else:
            chars.append(ch)
    return " ".join("".join(chars).split())


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance (substitution, deletion, insertion each cost 1)."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (r != h),  # substitution / match
                )
            )
        previous = current
    return previous[-1]


def _word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (word edit errors, reference word count) after normalization."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    return edit_distance(ref_words, hyp_words), len(ref_words)


def _char_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (character edit errors, reference character count) after normalization.

    Single spaces between words are counted as characters.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return edit_distance(ref, hyp), len(ref)


def _rate(errors: int, total: int, what: str) -> float:
    if total == 0:
        raise ValueError(f"normalized reference has no {what}; error rate is undefined")
    return errors / total


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = (substitutions + deletions + insertions) / reference word count.

    Both transcripts are normalized with ``normalize_text``. May exceed 1.0.
    Raises ValueError if the normalized reference has no words.
    """
    errors, total = _word_errors(reference, hypothesis)
    return _rate(errors, total, "words")


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = character-level edit distance / reference character count.

    Uses the same normalization as WER; single spaces count as characters.
    Raises ValueError if the normalized reference is empty.
    """
    errors, total = _char_errors(reference, hypothesis)
    return _rate(errors, total, "characters")


def _validate_corpus(references: list[str], hypotheses: list[str]) -> None:
    for name, seq in (("references", references), ("hypotheses", hypotheses)):
        if not isinstance(seq, (list, tuple)):
            raise ValueError(f"{name} must be a list of strings, got {type(seq).__name__}")
    if not references or not hypotheses:
        raise ValueError("references and hypotheses must not be empty")
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references and hypotheses must have the same length, "
            f"got {len(references)} and {len(hypotheses)}"
        )
    for item in (*references, *hypotheses):
        if not isinstance(item, str):
            raise ValueError(f"all transcripts must be str, got {type(item).__name__}")


def _corpus_rate(references, hypotheses, error_fn, what: str) -> float:
    _validate_corpus(references, hypotheses)
    errors = total = 0
    for ref, hyp in zip(references, hypotheses):
        e, n = error_fn(ref, hyp)
        errors += e
        total += n
    return _rate(errors, total, what)


def corpus_word_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Total word errors divided by total reference words (not a mean of per-utterance WERs)."""
    return _corpus_rate(references, hypotheses, _word_errors, "words")


def corpus_character_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Total character errors divided by total reference characters."""
    return _corpus_rate(references, hypotheses, _char_errors, "characters")
