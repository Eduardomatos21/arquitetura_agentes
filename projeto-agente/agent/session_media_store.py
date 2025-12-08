"""Session-scoped helpers to persist and retrieve the last uploaded image.

Images arrive through CopilotKit/ADK as inline_data blobs or JSON payloads. Once a
user uploads an image we persist lightweight metadata inside the ADK session state
so future tool invocations (e.g., text follow-ups) can still access the bytes even
if the request payload no longer carries inline_data. We also keep an optional disk
copy to avoid storing large base64 strings only in memory.
"""
from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    from google.adk.sessions.state import State
except ImportError:  # pragma: no cover - makes module import-safe in tooling
    State = None  # type: ignore

STATE_KEY_LAST_IMAGE = "media:last_image"
CACHE_DIR = Path(os.environ.get("SESSION_MEDIA_CACHE_DIR", Path(__file__).parent / "session_media_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MAX_CACHED_FILES = int(os.environ.get("SESSION_MEDIA_MAX_FILES", "64"))


def _persist_to_disk(image_bytes: bytes) -> Optional[str]:
    """Saves bytes to disk and returns the absolute path."""
    if not image_bytes:
        return None
    timestamp = int(time.time() * 1000)
    filename = f"img_{timestamp}.bin"
    file_path = CACHE_DIR / filename
    try:
        file_path.write_bytes(image_bytes)
    except OSError:
        return None
    _prune_cache()
    return str(file_path)


def _prune_cache() -> None:
    """Keeps on-disk cache bounded to avoid uncontrolled growth."""
    try:
        files = sorted(CACHE_DIR.glob("img_*.bin"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    while len(files) > MAX_CACHED_FILES:
        file_path = files.pop(0)
        try:
            file_path.unlink()
        except OSError:
            continue


def store_image_in_state(state: Optional[State], image_bytes: bytes, mime_type: str, *, source: str) -> None:
    """Persists image metadata (base64 + optional file path) under the session state."""
    if state is None or not image_bytes:
        return
    safe_mime = mime_type or "image/png"
    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
    except Exception:
        encoded = ""
    payload = {
        "mime_type": safe_mime,
        "source": source,
        "updated_at": int(time.time()),
    }
    if encoded:
        payload["base64"] = encoded
    disk_path = _persist_to_disk(image_bytes)
    if disk_path:
        payload["path"] = disk_path
    state[STATE_KEY_LAST_IMAGE] = payload


def load_image_from_state(state: Optional[State]) -> Tuple[Optional[bytes], Optional[str]]:
    """Returns the last stored image bytes + mime_type if available."""
    if state is None:
        return None, None
    payload = state.get(STATE_KEY_LAST_IMAGE)
    if not isinstance(payload, dict):
        return None, None
    mime_type = payload.get("mime_type", "image/png")
    data_b64 = payload.get("base64")
    if data_b64:
        try:
            return base64.b64decode(data_b64), mime_type
        except Exception:
            pass
    disk_path = payload.get("path")
    if disk_path and os.path.exists(disk_path):
        try:
            return Path(disk_path).read_bytes(), mime_type
        except OSError:
            return None, None
    return None, None
