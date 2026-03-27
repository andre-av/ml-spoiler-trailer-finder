"""
Combined Spoiler Scorer

Merges keyword heuristic and zero-shot classification signals into a single
spoiler risk assessment for a YouTube trailer's comment section.

Scoring approach:
- Each comment gets a keyword score (0-1) and a zero-shot score (0-1)
- The combined comment score = weighted average of both signals
- The overall trailer risk = based on the proportion and confidence of flagged comments
"""

from dataclasses import dataclass, field
from src.model.keyword_detector import score_comment as kw_score
from src.model.zero_shot import classify_comment as zs_classify


# Weight for each signal in the combined score.
# These can be tuned as we add more signals or gather evaluation data.
KEYWORD_WEIGHT = 0.4
ZERO_SHOT_WEIGHT = 0.6

COMMENT_FLAG_THRESHOLD = 0.4


@dataclass
class CommentAnalysis:
    """Combined analysis for a single comment."""
    text: str
    keyword_score: float
    zero_shot_score: float
    combined_score: float
    keyword_patterns: list
    is_flagged: bool


@dataclass
class TrailerAnalysis:
    """Full analysis of a trailer's comment section."""
    total_comments: int
    flagged_count: int
    flagged_percentage: float
    spoiler_risk_score: float  # 0-100 overall risk rating
    risk_label: str  # "Low", "Medium", "High"
    # Per-signal aggregate stats
    keyword_flagged_count: int = 0
    zero_shot_flagged_count: int = 0
    avg_keyword_score: float = 0.0
    avg_zero_shot_score: float = 0.0
    flagged_comments: list = field(default_factory=list)
    all_comments: list = field(default_factory=list)


def analyze_comment(text: str, custom_keywords: list = None) -> CommentAnalysis:
    """
    Analyze a single comment using both keyword and zero-shot signals.

    Args:
        text: Comment text to analyze.
        custom_keywords: Optional list of additional keyword strings for the keyword detector.

    Returns:
        CommentAnalysis with combined score.
    """
    kw = kw_score(text, custom_keywords=custom_keywords)
    zs = zs_classify(text)

    combined = (KEYWORD_WEIGHT * kw.score) + (ZERO_SHOT_WEIGHT * zs.spoiler_score)
    combined = round(min(combined, 1.0), 3)

    return CommentAnalysis(
        text=text,
        keyword_score=kw.score,
        zero_shot_score=zs.spoiler_score,
        combined_score=combined,
        keyword_patterns=kw.matched_patterns,
        is_flagged=combined >= COMMENT_FLAG_THRESHOLD,
    )


def _compute_risk_score(analyses: list) -> float:
    """
    Compute an overall spoiler risk score (0-100) from comment analyses.

    Factors in:
    - Percentage of flagged comments (primary signal)
    - Average confidence of flagged comments (secondary signal)
    - Maximum single comment score (catches even one strong warning)
    """
    if not analyses:
        return 0.0

    flagged = [a for a in analyses if a.is_flagged]
    if not flagged:
        return 0.0

    flagged_pct = len(flagged) / len(analyses)
    avg_flagged_score = sum(a.combined_score for a in flagged) / len(flagged)
    max_score = max(a.combined_score for a in analyses)

    # Weighted combination of signals
    risk = (
        0.45 * flagged_pct +          # proportion of spoiler comments
        0.30 * avg_flagged_score +     # how confident are the flags
        0.25 * max_score               # strongest single signal
    )

    # Scale to 0-100 and clamp
    return round(min(risk * 100, 100), 1)


def _risk_label(score: float) -> str:
    """Convert a 0-100 risk score to a human-readable label."""
    if score < 20:
        return "Low"
    elif score < 45:
        return "Medium"
    else:
        return "High"


def analyze_trailer(comments: list, custom_keywords: list = None) -> TrailerAnalysis:
    """
    Full spoiler analysis of a trailer's comment section.

    Args:
        comments: List of comment text strings.
        custom_keywords: Optional list of additional keyword strings for the keyword detector.

    Returns:
        TrailerAnalysis with overall risk score and per-comment details.
    """
    analyses = [analyze_comment(text, custom_keywords=custom_keywords) for text in comments]
    flagged = [a for a in analyses if a.is_flagged]
    risk_score = _compute_risk_score(analyses)

    kw_flagged = sum(1 for a in analyses if a.keyword_score >= 0.5)
    zs_flagged = sum(1 for a in analyses if a.zero_shot_score >= 0.4)
    avg_kw = round(sum(a.keyword_score for a in analyses) / len(analyses), 3) if analyses else 0
    avg_zs = round(sum(a.zero_shot_score for a in analyses) / len(analyses), 3) if analyses else 0

    return TrailerAnalysis(
        total_comments=len(comments),
        flagged_count=len(flagged),
        flagged_percentage=round(len(flagged) / len(comments) * 100, 1) if comments else 0,
        spoiler_risk_score=risk_score,
        risk_label=_risk_label(risk_score),
        keyword_flagged_count=kw_flagged,
        zero_shot_flagged_count=zs_flagged,
        avg_keyword_score=avg_kw,
        avg_zero_shot_score=avg_zs,
        flagged_comments=sorted(flagged, key=lambda a: a.combined_score, reverse=True),
        all_comments=analyses,
    )
