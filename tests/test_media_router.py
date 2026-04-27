"""Unit tests for the GET /media/{filename} endpoint.

Tests:
- Returns 404 for a filename that does not exist in ``_MEDIA_DIR``
- Path traversal attempt (e.g. ``../etc/passwd``) returns 404
- Returns 200 with ``audio/mpeg`` content type for a valid ``.mp3`` file

Requirements: 4.2
"""

from __future__ import annotations

import pathlib

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

import interview_practice_partner.api.routers.media as media_module
from interview_practice_partner.api.routers.media import router as media_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_app() -> FastAPI:
    """Build a minimal FastAPI app that includes only the media router."""
    app = FastAPI()
    app.include_router(media_router)
    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMediaRouter:
    """Unit tests for GET /media/{filename}.

    Requirements: 4.2
    """

    async def test_returns_404_for_nonexistent_file(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """GET /media/{filename} returns 404 when the file does not exist in _MEDIA_DIR.

        Requirements: 4.2
        """
        monkeypatch.setattr(media_module, "_MEDIA_DIR", tmp_path)

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/media/nonexistent_file.mp3")

        assert response.status_code == 404

    async def test_path_traversal_returns_404(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """GET /media/../etc/passwd returns 404 — safe_name strips directory components.

        The router uses ``pathlib.Path(filename).name`` to strip any directory
        components, so ``../etc/passwd`` becomes ``passwd``, which does not exist
        in ``_MEDIA_DIR``.

        Requirements: 4.2
        """
        monkeypatch.setattr(media_module, "_MEDIA_DIR", tmp_path)

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/media/../etc/passwd")

        assert response.status_code == 404

    async def test_returns_200_with_audio_mpeg_for_valid_mp3(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """GET /media/{filename} returns 200 with audio/mpeg for a valid .mp3 file.

        A real ``.mp3`` file is written to the temporary directory (which is
        patched as ``_MEDIA_DIR``) so that the endpoint can serve it.

        Requirements: 4.2
        """
        monkeypatch.setattr(media_module, "_MEDIA_DIR", tmp_path)

        # Write a dummy mp3 file to the temp directory
        mp3_file = tmp_path / "test_audio.mp3"
        mp3_file.write_bytes(b"\xff\xfb\x90\x00" * 16)  # minimal fake MP3 bytes

        app = _build_test_app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/media/test_audio.mp3")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/mpeg")
