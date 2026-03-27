"""
Keyword Heuristic Spoiler Detector (Signal 1: Baseline)

Detects YouTube trailer comments that warn about or contain spoilers
using pattern matching. Serves as a baseline that ML-based signals 
should improve upon.

Two detection modes:
1. Meta-warning detection: comments warning the trailer contains spoilers
   e.g., "this trailer shows way too much", "don't watch, full of spoilers"
2. Plot-reveal indicators: comments that hint at revealing plot details
   e.g., "I can't believe they showed him dying"

Each comment gets a score from 0.0 (no spoiler signal) to 1.0 (strong signal).
"""

import re
from dataclasses import dataclass


@dataclass
class KeywordResult:
    """Result of keyword analysis for a single comment."""
    text: str
    score: float  # 0.0 to 1.0
    matched_patterns: list  # which patterns triggered
    is_flagged: bool  # True if score >= threshold


# --- Pattern Definitions ---
# Each pattern is a tuple of (compiled_regex, weight, description).
# Weights reflect how strong a spoiler signal the pattern represents.

META_WARNING_PATTERNS = [
    # Direct spoiler warnings about the trailer
    (re.compile(r"\bspoilers?\s*(alert|warning)?\b", re.IGNORECASE), 0.7, "spoiler mention"),
    (re.compile(r"\bspoil(s|ed|ing)?\b", re.IGNORECASE), 0.6, "spoil verb"),
    (re.compile(r"\bshows?\s+too\s+much\b", re.IGNORECASE), 0.8, "shows too much"),
    (re.compile(r"\bgives?\s+away\b", re.IGNORECASE), 0.7, "gives away"),
    (re.compile(r"\bgave\s+away\b", re.IGNORECASE), 0.7, "gave away"),
    (re.compile(r"\bruined\b", re.IGNORECASE), 0.5, "ruined"),
    (re.compile(r"\bdon'?t\s+watch\s+(this\s+)?trailer\b", re.IGNORECASE), 0.9, "don't watch trailer"),
    (re.compile(r"\bwhole\s+(movie|plot|story)\b", re.IGNORECASE), 0.6, "whole movie/plot"),
    (re.compile(r"\bentire\s+(movie|plot|story)\b", re.IGNORECASE), 0.6, "entire movie/plot"),
    (re.compile(r"\breveal(s|ed)?\s+(too\s+much|everything|the\s+(ending|plot|twist))\b", re.IGNORECASE), 0.8, "reveals too much"),
    (re.compile(r"\bwhy\s+did\s+they\s+show\b", re.IGNORECASE), 0.7, "why did they show"),
    (re.compile(r"\bshould\s*(not|n'?t)\s+(have\s+)?(shown|put|included)\b", re.IGNORECASE), 0.7, "shouldn't have shown"),
]

PLOT_REVEAL_PATTERNS = [
    # Indicators that the comment itself reveals plot details
    (re.compile(r"\b(he|she|they)\s+(dies?|died|kills?|killed)\b", re.IGNORECASE), 0.6, "death reference"),
    (re.compile(r"\bturns?\s+out\s+(to\s+be|that)\b", re.IGNORECASE), 0.5, "plot twist language"),
    (re.compile(r"\bplot\s+twist\b", re.IGNORECASE), 0.6, "plot twist mention"),
    (re.compile(r"\bending\s+(is|was|scene)\b", re.IGNORECASE), 0.5, "ending reference"),
    (re.compile(r"\bthe\s+villain\s+(is|was|turns)\b", re.IGNORECASE), 0.6, "villain reveal"),
    (re.compile(r"\bat\s+\d{1,2}:\d{2}\b", re.IGNORECASE), 0.4, "timestamp reference"),
]

ALL_PATTERNS = META_WARNING_PATTERNS + PLOT_REVEAL_PATTERNS

# Default threshold for flagging a comment
DEFAULT_THRESHOLD = 0.5


def _build_custom_patterns(keywords: list) -> list:
    """Build regex patterns from user-provided keyword strings."""
    patterns = []
    for kw in keywords:
        kw = kw.strip()
        if kw:
            escaped = re.escape(kw)
            patterns.append(
                (re.compile(r"\b" + escaped + r"\b", re.IGNORECASE), 0.6, f"custom: {kw}")
            )
    return patterns


def score_comment(text: str, custom_keywords: list = None) -> KeywordResult:
    """
    Score a single comment for spoiler signals using keyword patterns.

    Args:
        text: The comment text to analyze.
        custom_keywords: Optional list of additional keyword strings to match.

    Returns:
        KeywordResult with score, matched patterns, and flag status.
    """
    patterns = ALL_PATTERNS
    if custom_keywords:
        patterns = patterns + _build_custom_patterns(custom_keywords)

    matched = []
    max_score = 0.0

    for pattern, weight, description in patterns:
        if pattern.search(text):
            matched.append(description)
            max_score = max(max_score, weight)

    # Boost score if multiple patterns match (stronger signal)
    if len(matched) >= 2:
        max_score = min(max_score + 0.15, 1.0)
    if len(matched) >= 3:
        max_score = min(max_score + 0.10, 1.0)

    return KeywordResult(
        text=text,
        score=round(max_score, 2),
        matched_patterns=matched,
        is_flagged=max_score >= DEFAULT_THRESHOLD,
    )


def score_comments(comments: list, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Score a batch of comments and return summary statistics.

    Args:
        comments: List of comment strings.
        threshold: Minimum score to flag a comment (default 0.5).

    Returns:
        Dictionary with results, flagged comments, and aggregate stats.
    """
    results = [score_comment(text) for text in comments]

    # Override threshold if different from default
    if threshold != DEFAULT_THRESHOLD:
        for r in results:
            r.is_flagged = r.score >= threshold

    flagged = [r for r in results if r.is_flagged]
    scores = [r.score for r in results]

    return {
        "total_comments": len(comments),
        "flagged_count": len(flagged),
        "flagged_percentage": round(len(flagged) / len(comments) * 100, 1) if comments else 0,
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "flagged_comments": sorted(flagged, key=lambda r: r.score, reverse=True),
        "all_results": results,
    }
