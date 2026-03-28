"""
Movie Trailer Spoiler Detector -- Streamlit Web App

Paste a YouTube trailer URL and get a spoiler risk assessment
based on comment analysis.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st
from src.scraper.youtube_comments import fetch_comments, fetch_video_title
from src.model.combined_scorer import analyze_trailer


# --- Page Config ---
st.set_page_config(
    page_title="Fentaza's Spoiler Detector",
    page_icon="🎬",
    layout="centered",
)

has_results = "result" in st.session_state

COMMENT_OPTIONS = {"50": 50, "100": 100, "200": 200, "10000 (slower)": 10000}

# --- Custom CSS: split-view UX (only when results exist) ---
if has_results:
    # animate_results is True only on the first render after a new analysis,
    # so the slide-in plays once and not on every widget interaction.
    should_animate = st.session_state.pop("animate_results", False)
    animation_css = """
    @keyframes fadeSlideIn {
        from { opacity: 0; transform: translateX(40px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(2) {
        animation: fadeSlideIn 0.55s ease-out;
    }
    """ if should_animate else ""

    st.markdown(f"""
    <style>
    {animation_css}

    /* Let left and right columns have independent heights */
    .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type {{
        align-items: flex-start !important;
    }}

    /* Pin the left panel so it stays visible while the right side scrolls */
    .stMainBlockContainer [data-testid="stHorizontalBlock"]:first-of-type > div:first-child {{
        position: sticky;
        top: 3.5rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Layout: full-width before analysis, split after ---
if has_results:
    left_col, right_col = st.columns([4, 5])
else:
    left_col = st.container()
    right_col = None

# --- Left side (or full-width before analysis): input controls ---
with left_col:
    st.title("🎬 Movie Trailer Spoiler Detector")

    st.markdown("## Do you want to ruin your movie experience?")
    st.markdown(
        "Probably not. Run this analysis to find out if the comment section "
        "warns about spoilers."
    )
    st.markdown(
        "Paste a YouTube trailer URL below to find out if the comment section "
        "warns about spoilers -- **before** you watch the trailer."
    )

    url = st.text_input(
        "YouTube Trailer URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    col1, col2 = st.columns([5, 4])
    with col1:
        comment_choice = st.radio(
            "Comments to analyze",
            options=list(COMMENT_OPTIONS.keys()),
            index=0,
            horizontal=True,
            help="'10000' fetches, realistically, every available comment. This will be slower.",
        )
        max_comments = COMMENT_OPTIONS[comment_choice]

    with col2:
        custom_kw_input = st.text_input(
            "Custom keywords (comma-separated, optional)",
            placeholder='e.g. dies, killed, twist, ending',
        )

    custom_keywords = [k.strip() for k in custom_kw_input.split(",") if k.strip()] if custom_kw_input else None

    analyze_clicked = st.button("Analyze Trailer", type="primary", use_container_width=True)

    # Analysis runs inside left_col so the loading status appears here
    if analyze_clicked and url:
        if "youtube.com/watch" not in url and "youtu.be/" not in url:
            st.error("Please enter a valid YouTube URL.")
        else:
            with st.status("Analyzing trailer comments...", expanded=True) as status:
                st.write("Fetching video title...")
                video_title = fetch_video_title(url)

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

                result = analyze_trailer(comment_texts, custom_keywords=custom_keywords)
                status.update(label="Analysis complete!", state="complete", expanded=False)

            st.session_state["result"] = result
            st.session_state["analyzed_count"] = len(comment_texts)
            st.session_state["video_title"] = video_title
            st.session_state["video_url"] = url
            st.session_state["params"] = {
                "comments_requested": comment_choice,
                "custom_keywords": custom_keywords,
            }
            st.session_state["animate_results"] = True
            st.rerun()

    elif analyze_clicked and not url:
        st.warning("Enter a valid url :(")

# --- Right side: results (only rendered when we have results) ---
if has_results and right_col is not None:
    st.set_page_config(layout="wide")

    result = st.session_state["result"]
    meta = st.session_state["params"]

    with right_col:
        st.markdown(f"## {st.session_state['video_title']}")
        st.success(f"Analysis complete! ({st.session_state['analyzed_count']} comments analyzed)")

        risk_color = {
            "Low": "green",
            "Medium": "orange",
            "High": "red",
        }.get(result.risk_label, "gray")

        st.markdown(f"### Spoiler Risk: :{risk_color}[{result.risk_label}]")

        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", f"{result.spoiler_risk_score}/100")
        m2.metric("Flagged Comments", f"{result.flagged_count}/{result.total_comments}")
        m3.metric("Flagged %", f"{result.flagged_percentage}%")

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

            if meta["custom_keywords"]:
                st.caption(f"Custom keywords included: {', '.join(meta['custom_keywords'])}")

        st.divider()

        if result.flagged_comments:
            st.subheader("Flagged Comments")
            st.caption("Comments that indicate the trailer may contain spoilers, sorted by confidence.")

            for i, comment in enumerate(result.flagged_comments, 1):
                score_pct = int(comment.combined_score * 100)

                with st.container(border=True):
                    st.markdown(f"**#{i}** — Confidence: **{score_pct}%**")
                    st.text(comment.text[:500])

                    with st.expander("Signal breakdown"):
                        c1, c2 = st.columns(2)
                        c1.metric("Keyword Score", f"{comment.keyword_score:.2f}")
                        c2.metric("Zero-Shot Score", f"{comment.zero_shot_score:.3f}")
                        if comment.keyword_patterns:
                            st.caption(f"Keyword patterns matched: {', '.join(comment.keyword_patterns)}")
        else:
            st.success("No spoiler warnings detected in the comments!")

        with st.expander(f"View all {result.total_comments} analyzed comments"):
            sorted_comments = sorted(result.all_comments, key=lambda c: c.combined_score, reverse=True)

            df_data = []
            for comment in sorted_comments:
                df_data.append({
                    "Confidence": f"{'🔴' if comment.is_flagged else '⚪'} {round(comment.combined_score * 100):.0f}%",
                    "Comment": comment.text[:150].replace("\n", " "),
                    "Keyword": round(comment.keyword_score, 2),
                    "Zero-Shot": round(comment.zero_shot_score, 3),
                    "Patterns": ", ".join(comment.keyword_patterns) if comment.keyword_patterns else "",
                })

            df = pd.DataFrame(df_data)

            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    "Confidence": st.column_config.TextColumn(width="small"),
                    "Keyword": st.column_config.NumberColumn(format="%.2f", width="small"),
                    "Zero-Shot": st.column_config.NumberColumn(format="%.3f", width="small"),
                    "Patterns": st.column_config.TextColumn(width="medium"),
                    "Comment": st.column_config.TextColumn(width="large"),
                },
            )

# --- Footer ---
st.divider()
st.caption(
    "Built with BART zero-shot classification + keyword heuristics. "
    "This tool analyzes YouTube comments to assess spoiler risk — "
    "it does not analyze the trailer video itself."
)
