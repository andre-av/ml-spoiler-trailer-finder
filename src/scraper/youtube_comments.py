"""
YouTube Comment Scraper

Fetches comments from a YouTube video URL using youtube-comment-downloader.
No API key required.
"""

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR


def fetch_comments(url: str, max_comments: int = 200, sort_by_popular: bool = True) -> list:
    """
    Fetch comments from a YouTube video.

    Args:
        url: Full YouTube video URL.
        max_comments: Maximum number of comments to fetch.
        sort_by_popular: If True, sort by most popular (default).
                         If False, sort by newest.

    Returns:
        List of dicts, each with keys: 'text', 'author', 'votes', 'time'.
    """
    downloader = YoutubeCommentDownloader()
    sort = SORT_BY_POPULAR if sort_by_popular else 1

    comments = []
    for comment in downloader.get_comments_from_url(url, sort_by=sort):
        comments.append({
            "text": comment.get("text", ""),
            "author": comment.get("author", ""),
            "votes": comment.get("votes", 0),
            "time": comment.get("time", ""),
        })
        if len(comments) >= max_comments:
            break

    return comments


def fetch_comment_texts(url: str, max_comments: int = 200) -> list:
    """
    Convenience function: fetch only the comment text strings.

    Args:
        url: Full YouTube video URL.
        max_comments: Maximum number of comments to fetch.

    Returns:
        List of comment text strings.
    """
    comments = fetch_comments(url, max_comments=max_comments)
    return [c["text"] for c in comments]
