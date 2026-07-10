"""
Local Search Server for Search-R1

This module provides a local search engine interface that mimics the google_search_server.py API.
It sends requests to a local retrieval server (e.g., running retrieval_server.py from Search-R1)
and formats the results to match the expected output format.

Usage:
    In your generate_with_search.py, replace:
        from google_search_server import google_search
    with:
        from local_search_server import local_search as google_search

    And update SEARCH_R1_CONFIGS:
        SEARCH_R1_CONFIGS = {
            "search_url": "http://127.0.0.1:8000/retrieve,http://127.0.0.1:8001/retrieve",
            "topk": 3,
            ...
        }
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import aiohttp


_SEARCH_URL_COUNTER = itertools.count()


def _normalize_search_urls(search_url: str | Sequence[str]) -> list[str]:
    if isinstance(search_url, str):
        urls = [url.strip() for url in search_url.split(",")]
    else:
        urls = [str(url).strip() for url in search_url]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("search_url must contain at least one URL")
    return urls


def _round_robin_urls(search_url: str | Sequence[str]) -> list[str]:
    urls = _normalize_search_urls(search_url)
    offset = next(_SEARCH_URL_COUNTER) % len(urls)
    return urls[offset:] + urls[:offset]


async def local_search(
    search_url: str | Sequence[str],
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
) -> list[dict]:
    """
    Call local search engine server and format results to match google_search_server.py output.

    This function provides the same interface as google_search() from google_search_server.py,
    making it a drop-in replacement. The only difference is that instead of using an API key,
    it uses a search_url parameter.

    Args:
        search_url: URL of the local retrieval server, or comma-separated URLs.
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds (default: 60)
        proxy: Proxy URL if needed (not used for local retrieval, kept for API compatibility)
        snippet_only: If True, only return snippet (kept for API compatibility with google_search)

    Returns:
        List of dictionaries with format: [{"document": {"contents": '"<title>"\n<text>'}}]
        This matches the output format of google_search() from google_search_server.py
    """
    # Prepare request payload for local retrieval server
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,  # We don't need scores for compatibility with google_search_server
    }

    # Send async request to a local retrieval server. If multiple comma-separated
    # URLs are provided, load balance with round-robin and fail over to the next.
    timeout_obj = aiohttp.ClientTimeout(total=timeout)

    result = None
    errors = []
    try:
        urls = _round_robin_urls(search_url)
    except Exception as e:
        raise ValueError(f"Invalid local search URL configuration {search_url!r}: {e}") from e
    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.post(url, json=payload, timeout=timeout_obj, proxy=proxy) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    break
            except Exception as e:
                errors.append(f"{url}: {e}")

    if result is None:
        raise RuntimeError(f"All local retriever URLs failed: {'; '.join(errors)}")

    # Parse retrieval results
    # Format from retrieval_server.py: {"result": [[{"document": {"id": "...", "contents": "..."}}]]}
    retrieval_results = result.get("result", [[]])[0]
    # Format to match google_search_server.py output
    # Google format: [{"document": {"contents": '"<title>"\n<context>'}}]
    contexts = []

    for item in retrieval_results:
        # Extract contents from retrieval result
        # retrieval_server returns: {"document": {"id": "...", "contents": '"Title"\nText...'}}
        if isinstance(item, dict):
            # Access the document dict first, then get contents
            content = item.get("contents", "")

            if content:
                # The contents are already in the correct format: '"Title"\nText content...'
                # Just pass through as-is to match google_search format
                contexts.append({"document": {"contents": content}})
            else:
                # Empty content case - provide default values
                contexts.append({"document": {"contents": '"No title."\nNo snippet available.'}})

    # If no results found, return empty list (consistent with google_search_server.py)
    return contexts
