"""Built-in tool: fetch and extract readable content from a URL."""

from __future__ import annotations

from typing import Any

from openclaw.agent.types import ToolDefinition, ToolParameter, ToolResult

DEFINITION = ToolDefinition(
    name="web_fetch",
    description="Fetch a URL and extract readable text content (HTML → text conversion).",
    parameters=[
        ToolParameter(name="url", description="The URL to fetch"),
        ToolParameter(name="max_chars", type="integer", description="Maximum characters to return (default 50000)", required=False),
    ],
)


async def execute(args: dict[str, Any], **_: Any) -> ToolResult:
    url = args.get("url", "")
    max_chars = int(args.get("max_chars", 50000))

    if not url:
        return ToolResult(tool_use_id="", content="Error: url is required", is_error=True)

    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        return ToolResult(
            tool_use_id="", content=f"Error: Only HTTP(S) URLs allowed, got '{parsed_url.scheme}'",
            is_error=True,
        )

    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as e:
        return ToolResult(
            tool_use_id="",
            content=f"Error: Missing dependency: {e}",
            is_error=True,
        )

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        return ToolResult(tool_use_id="", content=f"Error fetching URL: {e}", is_error=True)

    content_type = response.headers.get("content-type", "")
    body = response.text

    if "html" in content_type or body.strip().startswith("<"):
        soup = BeautifulSoup(body, "html.parser")
        # Remove script/style tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        # Convert <a href="url">text</a> → [text](url) to preserve links
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            link_text = a_tag.get_text(strip=True)
            if href and link_text and not href.startswith(("#", "javascript:")):
                a_tag.replace_with(f"[{link_text}]({href})")
            elif href and not href.startswith(("#", "javascript:")):
                a_tag.replace_with(f"({href})")
        text = soup.get_text(separator="\n", strip=True)
    else:
        text = body

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... [truncated at {max_chars} chars]"

    return ToolResult(tool_use_id="", content=text)
