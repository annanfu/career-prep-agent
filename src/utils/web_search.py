"""Web search utility wrapping Tavily API for interview prep research."""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def search_web(query: str, max_results: int = 5) -> List[dict]:
    """Search the web using Tavily API.

    Args:
        query: Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts, each with keys: title, url, content.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("  Warning: TAVILY_API_KEY not set, returning empty results.")
        return []

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=max_results)

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        })
    return results


if __name__ == "__main__":
    hits = search_web("Planet Labs software engineer interview questions", 3)
    print(f"Found {len(hits)} results:")
    for h in hits:
        print(f"  {h['title'][:60]} — {h['url'][:80]}")
        print(f"    {h['content'][:120]}...")
        print()
