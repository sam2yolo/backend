from __future__ import annotations


def join_public_url(base_url: str | None, path: str) -> str:
    if not base_url:
        return path
    if path.startswith(("http://", "https://", "ws://", "wss://")):
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"
