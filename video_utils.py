import logging
import os
import subprocess
from pathlib import Path
from typing import Optional


def is_dav_name(file_name: Optional[str]) -> bool:
    return Path(file_name or "").suffix.lower() == ".dav"


def storage_path_for_file(file_id: str, file_name: Optional[str]) -> str:
    suffix = ".dav" if is_dav_name(file_name) else ""
    return f"files/{file_id}{suffix}"


def maybe_convert_dav(file_id: str, file_path: str, file_name: Optional[str]) -> dict:
    original_name = file_name or file_id
    meta = {"path": file_path, "name": original_name}
    if not is_dav_name(original_name):
        return meta

    meta.update({"original_name": original_name, "converted": False})
    converted_path = f"files/{file_id}.mp4"
    converted_name = f"{Path(original_name).stem or file_id}.mp4"
    temp_path = f"files/{file_id}.tmp.mp4"
    last_error = ""

    attempts = [
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            file_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            temp_path,
        ],
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            file_path,
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            temp_path,
        ],
    ]

    for command in attempts:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                last_error = (result.stderr or result.stdout or "").strip()[-2000:]
                continue
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                last_error = "ffmpeg produced an empty output file"
                continue
            os.replace(temp_path, converted_path)
            if os.path.abspath(converted_path) != os.path.abspath(file_path):
                os.remove(file_path)
            logging.info("Converted DAV file %s to %s", original_name, converted_path)
            return {
                "path": converted_path,
                "name": converted_name,
                "original_name": original_name,
                "converted": True,
                "converted_from": "dav",
            }
        except FileNotFoundError:
            last_error = "ffmpeg executable was not found"
            break
        except Exception as exc:
            last_error = str(exc)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    if last_error:
        meta["conversion_error"] = last_error
        logging.warning("DAV conversion failed for %s: %s", original_name, last_error)
    return meta
