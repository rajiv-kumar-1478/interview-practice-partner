"""Router for serving temporary TTS audio files."""

from __future__ import annotations

import pathlib

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/media", tags=["media"])

# Temp directory where TTS audio files are stored
_MEDIA_DIR = pathlib.Path("/tmp/ipp_media")
_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/{filename}")
async def serve_media(filename: str) -> FileResponse:
    """Serve a temporary TTS audio file by filename.

    Files are stored in /tmp/ipp_media/ with a UUID-based name.
    Returns 404 if the file does not exist (e.g. TTL expired and cleaned up).

    Args:
        filename: The UUID-based filename (e.g. ``"a1b2c3d4.mp3"``).

    Requirements: 4.2
    """
    # Prevent path traversal by stripping any directory components
    safe_name = pathlib.Path(filename).name
    file_path = _MEDIA_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/mpeg",
        filename=safe_name,
    )
