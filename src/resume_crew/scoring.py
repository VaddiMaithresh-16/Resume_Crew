"""Deterministic, explainable keyword scoring."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordMatch:
    score: float
    # Tuples instead of lists — consistent with frozen=True immutability intent.
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    total_keywords: int


STOPWORDS = frozenset(
    "a an and are as at be by for from has have if in into is it its of on or such "
    "that the their there these this to was were will with within your you we our us "
    "they them he she his her i am but not no do does did can could should would may "
    "might must than then also more most other some each all any using use used able "
    "across per etc via".split()
)

# Fixed: `*` (zero-or-more) instead of `{1,}` so single uppercase tokens like
# `R` (the programming language) and `C` are captured as valid terms.
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-/]*")


def tokenize(text: str) -> list[str]:
    return [token.lower().strip(".-/") for token in TOKEN_PATTERN.findall(text)]


def extract_keywords(text: str, max_keywords: int = 40) -> list[str]:
    counts = Counter(
        token for token in tokenize(text)
        if token not in STOPWORDS and not token.isdigit()
    )
    return [
        term for term, _ in
        sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:max_keywords]
    ]


def keyword_match_score(resume_text: str, job_description_text: str) -> KeywordMatch:
    keywords = extract_keywords(job_description_text)
    resume_tokens = set(tokenize(resume_text))
    matched = tuple(keyword for keyword in keywords if keyword in resume_tokens)
    missing = tuple(keyword for keyword in keywords if keyword not in resume_tokens)
    score = round(100 * len(matched) / len(keywords), 1) if keywords else 0.0
    return KeywordMatch(score, matched, missing, len(keywords))


def _format_keyword_chips(terms: tuple[str, ...]) -> str:
    """Render keywords as inline code chips instead of one long comma wall."""
    if not terms:
        return "_(none)_"
    return " ".join(f"`{term}`" for term in terms)


def highlight_resume_matches(resume_text: str, match: KeywordMatch) -> str:
    """Render the resume text as HTML with matched keywords highlighted.

    Only matched terms are highlighted — a missing term by definition doesn't
    appear in the resume text, so it has nothing to highlight there. Missing
    terms are still listed separately (see format_keyword_score).

    Uses the exact same TOKEN_PATTERN as tokenize()/keyword_match_score,
    rather than a \\b-anchored alternation regex. Keyword tokens can end in
    symbol characters TOKEN_PATTERN allows (e.g. 'c++', 'c#', 'node.js') —
    those aren't "word" characters, so a \\b boundary right after them never
    actually falls between a word and non-word char and silently fails to
    match. Walking the text with the same tokenizer sidesteps that entirely.
    """
    import html as _html

    matched_lookup = {term.lower() for term in match.matched}
    if not matched_lookup:
        return f"<pre style='white-space:pre-wrap'>{_html.escape(resume_text)}</pre>"

    def _wrap(m: re.Match) -> str:
        token = m.group(0)
        normalized = token.lower().strip(".-/")
        escaped_token = _html.escape(token)
        if normalized in matched_lookup:
            return (
                "<mark style='background:#10b98155;color:inherit;border-radius:3px;"
                f"padding:0 2px'>{escaped_token}</mark>"
            )
        return escaped_token

    # Rebuild the output piece by piece: TOKEN_PATTERN.split keeps every
    # non-token separator (whitespace, punctuation) untouched and unescaped
    # here, so escape those chunks manually alongside the token substitution.
    pieces: list[str] = []
    pos = 0
    for m in TOKEN_PATTERN.finditer(resume_text):
        if m.start() > pos:
            pieces.append(_html.escape(resume_text[pos:m.start()]))
        pieces.append(_wrap(m))
        pos = m.end()
    if pos < len(resume_text):
        pieces.append(_html.escape(resume_text[pos:]))

    return f"<pre style='white-space:pre-wrap;font-family:inherit'>{''.join(pieces)}</pre>"


def format_keyword_score(match: KeywordMatch) -> str:
    matched = _format_keyword_chips(match.matched)
    missing = _format_keyword_chips(match.missing)
    return (
        f"**Score:** {match.score}% of the top {match.total_keywords} job-description keywords "
        "appear as whole terms in the resume.\n\n"
        f"**✅ Matched ({len(match.matched)}):**\n\n{matched}\n\n"
        f"**❌ Missing ({len(match.missing)}):**\n\n{missing}\n\n"
        "_This is an explainable keyword-overlap signal, not an ATS simulation._"
    )
