"""
Zero-Shot Spoiler Classification (Signal 2)

Uses a pre-trained NLI (Natural Language Inference) model to classify
YouTube comments without any spoiler-specific training data.

The model determines how well a comment matches candidate labels like
"this comment warns about spoilers in the trailer" without ever being
explicitly trained on spoiler data.

Model: facebook/bart-large-mnli (trained on Multi-Genre NLI dataset)
"""

from dataclasses import dataclass
from transformers import pipeline


@dataclass
class ZeroShotResult:
    """Result of zero-shot classification for a single comment."""
    text: str
    spoiler_score: float  # probability of the spoiler-related label
    top_label: str
    all_scores: dict  # label -> score mapping
    is_flagged: bool


# Labels for zero-shot classification.
# The model scores how well each comment matches each label.
# Label phrasing significantly affects performance -- these were
# tuned by testing on real YouTube trailer comments.
CANDIDATE_LABELS = [
    "the trailer reveals too much of the movie",
    "the trailer looks good",
    "general comment",
]

# The spoiler label is the first one -- used to extract the spoiler score
SPOILER_LABEL = CANDIDATE_LABELS[0]

DEFAULT_THRESHOLD = 0.4

# Module-level cache for the pipeline (expensive to load, do it once)
_classifier = None


def _get_classifier():
    """Lazy-load the zero-shot classification pipeline."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    return _classifier


def classify_comment(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    candidate_labels: list = None,
) -> ZeroShotResult:
    """
    Classify a single comment using zero-shot classification.

    Args:
        text: The comment text to classify.
        threshold: Minimum spoiler score to flag (default 0.5).
        candidate_labels: Custom labels to use (optional).

    Returns:
        ZeroShotResult with scores and flag status.
    """
    classifier = _get_classifier()
    labels = candidate_labels or CANDIDATE_LABELS

    result = classifier(text, labels)

    scores = dict(zip(result["labels"], result["scores"]))
    spoiler_score = scores.get(SPOILER_LABEL, 0.0)

    return ZeroShotResult(
        text=text,
        spoiler_score=round(spoiler_score, 3),
        top_label=result["labels"][0],
        all_scores={k: round(v, 3) for k, v in scores.items()},
        is_flagged=spoiler_score >= threshold,
    )


def classify_comments(
    comments: list,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Classify a batch of comments and return summary statistics.

    Args:
        comments: List of comment strings.
        threshold: Minimum spoiler score to flag (default 0.5).

    Returns:
        Dictionary with results, flagged comments, and aggregate stats.
    """
    results = [classify_comment(text, threshold) for text in comments]

    flagged = [r for r in results if r.is_flagged]
    scores = [r.spoiler_score for r in results]

    return {
        "total_comments": len(comments),
        "flagged_count": len(flagged),
        "flagged_percentage": round(len(flagged) / len(comments) * 100, 1) if comments else 0,
        "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
        "max_score": round(max(scores), 3) if scores else 0,
        "flagged_comments": sorted(flagged, key=lambda r: r.spoiler_score, reverse=True),
        "all_results": results,
    }
