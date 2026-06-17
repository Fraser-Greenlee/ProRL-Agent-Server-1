#!/usr/bin/env python3
"""MCP bridge exposing LiteResearcher `search` / `visit` to the Hermes agent.

Speaks the LiteResearcher retrieval server's native HTTP API
(https://github.com/simplexai-labs/LiteResearcher, ``server/local_rag_server.py``):

- ``POST {SEARCH_URL}``  {"query", "limit", "search_type"} -> {"results": [{link,title,snippet,score}]}
- ``POST {PARSER_URL}``  {"url"}                            -> {"found", "url", "title", "text"}

Point ``LITERESEARCHER_SEARCH_URL`` / ``LITERESEARCHER_PARSER_URL`` at a running
service (default ``http://127.0.0.1:8018`` reached over the host network).
``mcp`` and ``httpx`` are preinstalled in the runtime image.
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("literesearcher")

SEARCH_URL = os.environ.get("LITERESEARCHER_SEARCH_URL", "http://127.0.0.1:8018/search")
PARSER_URL = os.environ.get("LITERESEARCHER_PARSER_URL", "http://127.0.0.1:8018/web_parser")
SEARCH_LIMIT = int(os.environ.get("LITERESEARCHER_SEARCH_LIMIT", "10"))
SEARCH_TYPE = os.environ.get("LITERESEARCHER_SEARCH_TYPE", "hybrid")
TIMEOUT = float(os.environ.get("LITERESEARCHER_TIMEOUT", "120"))


def _as_list(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


@mcp.tool()
async def search(query: str | list[str]) -> str:
    """Search the LiteResearcher corpus; returns ranked title / url / snippet results."""
    queries = _as_list(query)
    if not queries:
        return "No query provided."
    blocks: list[str] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for q in queries:
            try:
                response = await client.post(
                    SEARCH_URL,
                    json={"query": q, "limit": SEARCH_LIMIT, "search_type": SEARCH_TYPE},
                )
                response.raise_for_status()
                results = response.json().get("results") or []
            except Exception as exc:  # noqa: BLE001 — surface the error to the agent
                blocks.append(f'Search error for "{q}": {exc}')
                continue
            lines = [f'Results for "{q}" ({len(results)}):']
            for idx, item in enumerate(results, start=1):
                lines.append(
                    f"{idx}. [{item.get('title') or item.get('link')}]({item.get('link')})\n"
                    f"Score: {item.get('score')}\n"
                    f"{item.get('snippet', '')}"
                )
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


@mcp.tool()
async def visit(url: str | list[str], goal: str = "") -> str:
    """Fetch the full text of LiteResearcher source URL(s) for the given goal."""
    urls = _as_list(url)
    if not urls:
        return "No URL provided."
    parts: list[str] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for u in urls:
            try:
                response = await client.post(PARSER_URL, json={"url": u})
                if response.status_code == 503:
                    parts.append(
                        f"{u}: full-text fetch is disabled on the retrieval server "
                        "(set ENABLE_SQL_FULLTEXT=true). Rely on search snippets instead."
                    )
                    continue
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:  # noqa: BLE001 — surface the error to the agent
                parts.append(f"{u}: visit error: {exc}")
                continue
            if not payload.get("found"):
                parts.append(f"{u}: not found in the corpus. Use search to find available URLs.")
                continue
            text = str(payload.get("text") or "")
            header = f"Title: {payload.get('title')}\nURL: {payload.get('url')}"
            if goal:
                header += f"\nGoal: {goal}"
            parts.append(f"{header}\n\n{text}")
    return "\n\n=======\n\n".join(parts)


if __name__ == "__main__":
    mcp.run()
