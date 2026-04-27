"""Unit tests for the media cleanup background task.

Tests:
- Files older than ``media_ttl_seconds`` are deleted by the cleanup task
- Files newer than ``media_ttl_seconds`` are not deleted

Requirements: 4.2
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import time

import pytest

import interview_practice_partner.main as main_module
from interview_practice_partner.main import _media_cleanup_loop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_file(directory: pathlib.Path, name: str, age_seconds: float) -> pathlib.Path:
    """Create a file in *directory* and set its mtime to *age_seconds* ago.

    Args:
        directory: Directory in which to create the file.
        name: Filename.
        age_seconds: How many seconds old the file should appear to be.

    Returns:
        The created file path.
    """
    file_path = directory / name
    file_path.write_bytes(b"dummy audio content")
    old_mtime = time.time() - age_seconds
    os.utime(file_path, (old_mtime, old_mtime))
    return file_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMediaCleanup:
    """Unit tests for _media_cleanup_loop.

    Requirements: 4.2
    """

    async def test_old_file_is_deleted(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """Files older than media_ttl_seconds are deleted by the cleanup task.

        Requirements: 4.2
        """
        monkeypatch.setattr(main_module, "_MEDIA_DIR", tmp_path)

        ttl_seconds = 60
        # Create a file that is 120 seconds old (twice the TTL)
        old_file = _create_file(tmp_path, "old_audio.mp3", age_seconds=120)

        assert old_file.exists(), "Precondition: old file should exist before cleanup"

        # The loop structure is: sleep → cleanup → sleep → cleanup → ...
        # We let the first sleep return normally so the cleanup pass runs,
        # then raise CancelledError on the second sleep to exit the loop.
        sleep_call_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                raise asyncio.CancelledError
            # First call: return normally so the cleanup code executes

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        with pytest.raises(asyncio.CancelledError):
            await _media_cleanup_loop(ttl_seconds=ttl_seconds, interval_seconds=1)

        assert not old_file.exists(), "Old file should have been deleted by cleanup"

    async def test_new_file_is_not_deleted(self, tmp_path: pathlib.Path, monkeypatch) -> None:
        """Files newer than media_ttl_seconds are NOT deleted by the cleanup task.

        Requirements: 4.2
        """
        monkeypatch.setattr(main_module, "_MEDIA_DIR", tmp_path)

        ttl_seconds = 600
        # Create a file that is only 10 seconds old (well within the TTL)
        new_file = _create_file(tmp_path, "new_audio.mp3", age_seconds=10)

        assert new_file.exists(), "Precondition: new file should exist before cleanup"

        # Let the first sleep return so the cleanup pass actually runs,
        # then raise CancelledError on the second sleep to exit the loop.
        sleep_call_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        with pytest.raises(asyncio.CancelledError):
            await _media_cleanup_loop(ttl_seconds=ttl_seconds, interval_seconds=1)

        assert new_file.exists(), "New file should NOT have been deleted by cleanup"

    async def test_only_old_files_deleted_when_mixed(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """Only files older than media_ttl_seconds are deleted; newer files survive.

        Requirements: 4.2
        """
        monkeypatch.setattr(main_module, "_MEDIA_DIR", tmp_path)

        ttl_seconds = 60
        # Old file: 120 seconds old (should be deleted)
        old_file = _create_file(tmp_path, "old_audio.mp3", age_seconds=120)
        # New file: 10 seconds old (should survive)
        new_file = _create_file(tmp_path, "new_audio.mp3", age_seconds=10)

        sleep_call_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        with pytest.raises(asyncio.CancelledError):
            await _media_cleanup_loop(ttl_seconds=ttl_seconds, interval_seconds=1)

        assert not old_file.exists(), "Old file should have been deleted"
        assert new_file.exists(), "New file should NOT have been deleted"

    async def test_file_just_under_ttl_is_not_deleted(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """A file whose age is just under media_ttl_seconds is NOT deleted.

        The cleanup condition is strictly greater-than (age > ttl), so a file
        that is 5 seconds younger than the TTL should survive.

        Requirements: 4.2
        """
        monkeypatch.setattr(main_module, "_MEDIA_DIR", tmp_path)

        ttl_seconds = 60
        # File is 5 seconds younger than the TTL — should NOT be deleted
        under_ttl_file = _create_file(tmp_path, "under_ttl_audio.mp3", age_seconds=ttl_seconds - 5)

        sleep_call_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        with pytest.raises(asyncio.CancelledError):
            await _media_cleanup_loop(ttl_seconds=ttl_seconds, interval_seconds=1)

        assert under_ttl_file.exists(), "File under TTL should NOT be deleted"

    async def test_empty_directory_does_not_raise(
        self, tmp_path: pathlib.Path, monkeypatch
    ) -> None:
        """Cleanup on an empty directory completes without error.

        Requirements: 4.2
        """
        monkeypatch.setattr(main_module, "_MEDIA_DIR", tmp_path)

        sleep_call_count = 0

        async def mock_sleep(seconds: float) -> None:
            nonlocal sleep_call_count
            sleep_call_count += 1
            if sleep_call_count >= 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)

        # Should not raise any exception other than CancelledError
        with pytest.raises(asyncio.CancelledError):
            await _media_cleanup_loop(ttl_seconds=60, interval_seconds=1)
