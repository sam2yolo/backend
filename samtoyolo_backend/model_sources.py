from __future__ import annotations

from urllib.parse import parse_qs, urlparse


DEFAULT_SAM3_MODEL_URL = (
    "https://drive.google.com/file/d/"
    "1U_SBWxdyRFx-519v_UQZh48cm4y4qLVm/view?usp=sharing"
)
DEFAULT_SAM3_MODEL_FILENAME = "sam3.1.zip"


def google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts):
            return parts[index + 1]
    return parse_qs(parsed.query).get("id", [None])[0]


def google_drive_download_url(url: str) -> str:
    file_id = google_drive_file_id(url)
    if not file_id:
        return url
    return f"https://drive.google.com/uc?export=download&id={file_id}"
