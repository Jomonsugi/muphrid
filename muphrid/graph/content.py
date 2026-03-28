"""
Content normalization for LangChain messages.

Different LLM providers return message content in different formats:
- Together AI / OpenAI: str
- Anthropic: list[dict] with content blocks ({"type": "text", "text": "..."})

These helpers normalize content to a consistent format so the rest of
the codebase doesn't need provider-specific isinstance checks.
"""

from __future__ import annotations


def text_content(msg_content) -> str:
    """
    Normalize message content to a plain string.

    Handles:
    - str → str (pass through)
    - list[dict] → extract and join text from content blocks
    - None → ""
    """
    if msg_content is None:
        return ""
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        texts = []
        for block in msg_content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else str(msg_content)
    return str(msg_content)


def image_blocks(msg_content) -> list[dict]:
    """
    Extract image content blocks from a message.

    Returns empty list for string content or messages without images.
    """
    if not isinstance(msg_content, list):
        return []
    return [
        b for b in msg_content
        if isinstance(b, dict) and b.get("type") == "image_url"
    ]
