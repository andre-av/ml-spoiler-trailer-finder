"""
Movie Trailer Spoiler Detector -- Streamlit Web App

Paste a YouTube trailer URL and get a spoiler risk assessment
based on comment analysis.
"""

import sys
import os

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.scraper.youtube_comments import fetch_comments
from src.model.combined_scorer import analyze_trailer


# --- Page Config ---
st.set_page_config(
    page_title="BUCI",
    page_icon="🎬",
    layout="centered",
)

# --- Header ---
st.title("🎬 Movie Trailer Spoiler Detector")

st.markdown(
    "Do you want your movie experience to be ruined? "
    "You've come to the wrong place!" #\n\n
)
st.markdown(
    "Paste a YouTube trailer URL below to find out if the comment section "
    "warns about spoilers -- **before** you watch the trailer."
)

# --- URL Input ---
url = st.text_input(
    "YouTube Trailer URL",
    placeholder="https://www.youtube.com/watch?v=...",
)

col1, col2 = st.columns([1, 3])
with col1:
    max_comments = st.number_input(
        "Comments to analyze",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )
with col2:
    custom_kw_input = st.text_input(
        "Custom keywords (comma-separated, optional)",
        placeholder='e.g. dies, killed, twist, ending',
    )

custom_keywords = [k.strip() for k in custom_kw_input.split(",") if k.strip()] if custom_kw_input else None

analyze_clicked = st.button("Analyze Trailer", type="primary", use_container_width=True)

# --- Analysis ---
if analyze_clicked and url:
    if "youtube.com/watch" not in url and "youtu.be/" not in url:
        st.error("Please enter a valid YouTube URL.")
    else:
        # Step 1: Fetch comments
        with st.status("Analyzing trailer comments...", expanded=True) as status:
            st.write("Fetching YouTube comments...")
            try:
                raw_comments = fetch_comments(url, max_comments=max_comments)
            except Exception as e:
                st.error(f"Failed to fetch comments: {e}")
                st.stop()

            if not raw_comments:
                st.warning("No comments found for this video.")
                st.stop()

            comment_texts = [c["text"] for c in raw_comments]
            st.write(f"Fetched **{len(comment_texts)}** comments. Running spoiler analysis...")

            # Step 2: Run combined analysis
            result = analyze_trailer(comment_texts, custom_keywords=custom_keywords)
            status.update(label="Analysis complete!", state="complete", expanded=False)

        # --- Results Display ---
        st.divider()

        # Risk score with color coding
        risk_color = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }.get(result.risk_label, "gray")

        st.markdown(f"### Spoiler Risk: :{risk_color}[{result.risk_label}]")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{result.spoiler_risk_score}/100")
        m2.metric("Flagged Comments", f"{result.flagged_count}/{result.total_comments}")
        m3.metric("Flagged %", f"{result.flagged_percentage}%")

        # Per-signal breakdown
        with st.expander("Signal Breakdown"):
            s1, s2, s3 = st.columns(3)
            s1.markdown("**Keyword Detector**")
            s1.metric("Flagged", f"{result.keyword_flagged_count}/{result.total_comments}")
            s1.metric("Avg Score", f"{result.avg_keyword_score:.3f}")

            s2.markdown("**Zero-Shot Classifier**")
            s2.metric("Flagged", f"{result.zero_shot_flagged_count}/{result.total_comments}")
            s2.metric("Avg Score", f"{result.avg_zero_shot_score:.3f}")

            s3.markdown("**Combined**")
            s3.metric("Flagged", f"{result.flagged_count}/{result.total_comments}")
            s3.metric("Risk Score", f"{result.spoiler_risk_score}/100")

            if custom_keywords:
                st.caption(f"Custom keywords included: {', '.join(custom_keywords)}")

        st.divider()

        # Flagged comments
        if result.flagged_comments:
            st.subheader("Flagged Comments")
            st.caption("Comments that indicate the trailer may contain spoilers, sorted by confidence.")

            for i, comment in enumerate(result.flagged_comments, 1):
                score_pct = int(comment.combined_score * 100)

                with st.container(border=True):
                    st.markdown(f"**#{i}** — Confidence: **{score_pct}%**")
                    st.text(comment.text[:500])

                    # Signal breakdown in expander
                    with st.expander("Signal breakdown"):
                        c1, c2 = st.columns(2)
                        c1.metric("Keyword Score", f"{comment.keyword_score:.2f}")
                        c2.metric("Zero-Shot Score", f"{comment.zero_shot_score:.3f}")
                        if comment.keyword_patterns:
                            st.caption(f"Keyword patterns matched: {', '.join(comment.keyword_patterns)}")
        else:
            st.success("No spoiler warnings detected in the comments!")

        # All comments (collapsible)
        with st.expander(f"View all {result.total_comments} analyzed comments"):
            for comment in result.all_comments:
                flag = "🔴" if comment.is_flagged else "⚪"
                st.text(f"{flag} [{comment.combined_score:.2f}] {comment.text[:120]}")

elif analyze_clicked and not url:
    st.warning("Enter a valid url :(")

# --- Footer ---
st.divider()
st.caption(
    "Built with DistilBERT zero-shot classification + keyword heuristics. "
    "This tool analyzes YouTube comments to assess spoiler risk — "
    "it does not analyze the trailer video itself."
)
