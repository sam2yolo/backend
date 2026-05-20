from __future__ import annotations

import asyncio
import html
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from .model_sources import google_drive_download_url
from .records import utc_now


ProgressCallback = Callable[[float, str], Awaitable[None]]


@dataclass(slots=True)
class PreparedModelAsset:
    model_name: str
    source_url: str
    download_url: str
    archive_path: Path
    extract_dir: Path
    manifest_path: Path
    size_bytes: int
    downloaded: bool
    extracted: bool
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "source_url": self.source_url,
            "download_url": self.download_url,
            "archive_path": str(self.archive_path),
            "extract_dir": str(self.extract_dir),
            "manifest_path": str(self.manifest_path),
            "size_bytes": self.size_bytes,
            "downloaded": self.downloaded,
            "extracted": self.extracted,
            "reused": self.reused,
        }


async def prepare_zip_model_asset(
    *,
    model_name: str,
    source_url: str,
    download_url: str | None,
    filename: str,
    cache_dir: str | Path,
    progress: ProgressCallback | None = None,
) -> PreparedModelAsset:
    """Download and extract a zip model asset into the shared model cache."""

    resolved_download_url = google_drive_download_url(download_url or source_url)
    model_dir = Path(cache_dir).expanduser().resolve() / _safe_name(model_name)
    archive_dir = model_dir / "downloads"
    extract_dir = model_dir / "extracted" / Path(filename).stem
    archive_path = archive_dir / _safe_name(filename)
    manifest_path = extract_dir / ".samtoyolo_model_manifest.json"

    archive_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False
    extracted = False

    if archive_path.exists() and manifest_path.exists():
        if progress:
            await progress(100, f"{model_name} model already prepared")
        return PreparedModelAsset(
            model_name=model_name,
            source_url=source_url,
            download_url=resolved_download_url,
            archive_path=archive_path,
            extract_dir=extract_dir,
            manifest_path=manifest_path,
            size_bytes=archive_path.stat().st_size,
            downloaded=False,
            extracted=False,
            reused=True,
        )

    if not archive_path.exists():
        if progress:
            await progress(5, f"downloading {model_name} model archive")
        await _download_to_path(resolved_download_url, archive_path, progress=progress)
        downloaded = True

    if not zipfile.is_zipfile(archive_path):
        _raise_not_zip(archive_path)

    if not manifest_path.exists():
        if progress:
            await progress(85, f"extracting {model_name} model archive")
        await asyncio.to_thread(_extract_zip_safely, archive_path, extract_dir)
        _write_manifest(
            manifest_path,
            {
                "model_name": model_name,
                "source_url": source_url,
                "download_url": resolved_download_url,
                "archive_path": str(archive_path),
                "extract_dir": str(extract_dir),
                "size_bytes": archive_path.stat().st_size,
                "prepared_at": utc_now(),
            },
        )
        extracted = True

    if progress:
        await progress(100, f"{model_name} model ready")
    return PreparedModelAsset(
        model_name=model_name,
        source_url=source_url,
        download_url=resolved_download_url,
        archive_path=archive_path,
        extract_dir=extract_dir,
        manifest_path=manifest_path,
        size_bytes=archive_path.stat().st_size,
        downloaded=downloaded,
        extracted=extracted,
        reused=not downloaded and not extracted,
    )


async def _download_to_path(
    url: str, target_path: Path, *, progress: ProgressCallback | None
) -> None:
    parsed = urlparse(url)
    if parsed.scheme in {"", "file"}:
        source_path = Path(parsed.path if parsed.scheme == "file" else url).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"model archive not found: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
        await asyncio.to_thread(shutil.copy2, source_path, tmp_path)
        tmp_path.replace(target_path)
        if progress:
            await progress(80, f"copied model archive from {source_path}")
        return

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"unsupported model archive URL scheme: {parsed.scheme}")

    await _download_http(url, target_path, progress=progress)


async def _download_http(
    url: str, target_path: Path, *, progress: ProgressCallback | None
) -> None:
    try:
        import httpx
    except Exception as exc:
        raise RuntimeError("httpx is required for model downloads") from exc

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(f"{target_path.suffix}.tmp")
    current_url = url

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        for attempt in range(3):
            async with client.stream("GET", current_url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    html = (await response.aread()).decode(errors="replace")
                    confirm_url = _google_drive_confirm_url(
                        html, current_url, dict(client.cookies.items())
                    )
                    if confirm_url and confirm_url != current_url:
                        current_url = confirm_url
                        continue
                    raise RuntimeError(
                        "model download returned an HTML page instead of an archive; "
                        "check that the Google Drive link is public"
                    )

                total = int(response.headers.get("content-length") or "0")
                bytes_written = 0
                with tmp_path.open("wb") as handle:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        handle.write(chunk)
                        bytes_written += len(chunk)
                        if progress and total:
                            percent = 5 + min(75, bytes_written / total * 75)
                            await progress(percent, f"downloaded {bytes_written} bytes")
                tmp_path.replace(target_path)
                return

        raise RuntimeError(f"could not resolve Google Drive confirmation for {url}")


def _google_drive_confirm_url(
    html: str, current_url: str, cookies: dict[str, str]
) -> str | None:
    parsed = urlparse(current_url)
    if "google.com" not in parsed.netloc:
        return None

    for name, value in cookies.items():
        if name.startswith("download_warning"):
            return _replace_query_param(current_url, "confirm", value)

    normalised_html = html.replace("&amp;", "&")
    form_url = _google_drive_form_url(normalised_html)
    if form_url:
        return form_url

    href_match = re.search(
        r'href="([^"]*(?:/uc\?|/download\?)[^"]*confirm=[^"]+)"',
        normalised_html,
    )
    if href_match:
        return urljoin("https://drive.google.com", href_match.group(1))

    confirm_match = re.search(r"confirm=([0-9A-Za-z_-]+)", normalised_html)
    if confirm_match:
        return _replace_query_param(current_url, "confirm", confirm_match.group(1))

    return None


def _google_drive_form_url(document: str) -> str | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        document,
        re.IGNORECASE | re.DOTALL,
    )
    if not form_match:
        return None

    action = html.unescape(form_match.group(1))
    form_body = form_match.group(2)
    query: list[tuple[str, str]] = []
    for input_match in re.finditer(r"<input\b[^>]*>", form_body, re.IGNORECASE):
        attrs = _html_attrs(input_match.group(0))
        name = attrs.get("name")
        if name:
            query.append((name, attrs.get("value", "")))
    if not query:
        return None

    parsed = urlparse(action)
    existing = parse_qsl(parsed.query)
    return urlunparse(parsed._replace(query=urlencode([*existing, *query])))


def _html_attrs(tag: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in re.finditer(
        r"""([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""",
        tag,
    ):
        value = next(group for group in match.groups()[1:] if group is not None)
        attrs[match.group(1).lower()] = html.unescape(value)
    return attrs


def _replace_query_param(url: str, key: str, value: str) -> str:
    parsed = urlparse(url)
    pairs = [
        (item_key, item_value)
        for item_key, item_value in parse_qsl(parsed.query)
        if item_key != key
    ]
    pairs.append((key, value))
    return urlunparse(parsed._replace(query=urlencode(pairs)))


def _extract_zip_safely(archive_path: Path, extract_dir: Path) -> None:
    tmp_dir = extract_dir.with_name(f"{extract_dir.name}.tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        root = tmp_dir.resolve()
        for member in archive.infolist():
            destination = (tmp_dir / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"unsafe path in model archive: {member.filename}")
        archive.extractall(tmp_dir)
    shutil.rmtree(extract_dir, ignore_errors=True)
    tmp_dir.replace(extract_dir)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _raise_not_zip(path: Path) -> None:
    preview = path.read_bytes()[:200].decode(errors="replace")
    raise RuntimeError(
        f"downloaded model artifact is not a zip archive: {path}. "
        f"First bytes: {preview!r}"
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "asset"
