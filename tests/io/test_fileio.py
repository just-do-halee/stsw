"""Tests for file I/O module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stsw.io.fileio import (
    FileIOError,
    SafeFileWriter,
    ensure_parent_dir,
    get_file_size,
    pwrite,
    safe_seek,
)


class TestSafeFileWriter:
    """Test SafeFileWriter class."""

    def test_basic_write(self, tmp_path):
        """Test basic file writing."""
        target = tmp_path / "test.txt"

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"Hello, World!")
        writer.close()

        assert target.exists()
        assert target.read_bytes() == b"Hello, World!"

    def test_context_manager(self, tmp_path):
        """Test using SafeFileWriter as context manager."""
        target = tmp_path / "test.txt"

        with SafeFileWriter(target) as writer:
            writer.write(b"Context manager test")

        assert target.exists()
        assert target.read_bytes() == b"Context manager test"

    def test_atomic_write(self, tmp_path):
        """Test atomic write behavior."""
        target = tmp_path / "test.txt"
        temp_path = target.with_suffix(".txt.tmp")

        with SafeFileWriter(target) as writer:
            writer.write(b"Atomic write")
            # Temp file should exist during write
            assert temp_path.exists()

        # Temp file should be gone after close
        assert not temp_path.exists()
        assert target.exists()

    def test_abort(self, tmp_path):
        """Test aborting write."""
        target = tmp_path / "test.txt"
        temp_path = target.with_suffix(".txt.tmp")

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"This will be aborted")
        writer.abort()

        # Neither file should exist
        assert not temp_path.exists()
        assert not target.exists()

    def test_abort_in_context_manager(self, tmp_path):
        """Test abort when exception in context manager."""
        target = tmp_path / "test.txt"

        with pytest.raises(ValueError):
            with SafeFileWriter(target) as writer:
                writer.write(b"This will be aborted")
                raise ValueError("Test error")

        # File should not exist after exception
        assert not target.exists()

    def test_multiple_writes(self, tmp_path):
        """Test multiple write operations."""
        target = tmp_path / "test.txt"

        with SafeFileWriter(target) as writer:
            writer.write(b"First ")
            writer.write(b"Second ")
            writer.write(b"Third")

        assert target.read_bytes() == b"First Second Third"

    def test_large_buffer(self, tmp_path):
        """Test writing with large buffer."""
        target = tmp_path / "test.txt"
        large_data = b"x" * (10 * 1024 * 1024)  # 10MB

        with SafeFileWriter(target, buffer_size=8 * 1024 * 1024) as writer:
            writer.write(large_data)

        assert target.stat().st_size == len(large_data)

    def test_invalid_mode(self, tmp_path):
        """Test invalid file mode."""
        target = tmp_path / "test.txt"

        with pytest.raises(ValueError, match="Mode must be binary write"):
            SafeFileWriter(target, mode="r")

    def test_write_without_open(self, tmp_path):
        """Test writing without explicit open."""
        target = tmp_path / "test.txt"

        writer = SafeFileWriter(target)
        writer.write(b"Auto-open")
        writer.close()

        assert target.read_bytes() == b"Auto-open"

    def test_flush_and_fsync(self, tmp_path):
        """Test flush and fsync operations."""
        target = tmp_path / "test.txt"

        with SafeFileWriter(target) as writer:
            writer.write(b"Test data")
            writer.flush()
            writer.fsync()

        assert target.exists()

    @patch("os.fsync")
    def test_fsync_called(self, mock_fsync, tmp_path):
        """Test that fsync is called."""
        target = tmp_path / "test.txt"

        with SafeFileWriter(target) as writer:
            writer.write(b"Test")
            writer.fsync()

        mock_fsync.assert_called()

    def test_windows_rename(self, tmp_path):
        """Test Windows-specific rename behavior."""
        target = tmp_path / "test.txt"
        # Create existing file
        target.write_bytes(b"Old content")

        with patch("platform.system", return_value="Windows"):
            with SafeFileWriter(target) as writer:
                writer.write(b"New content")

        assert target.read_bytes() == b"New content"

    def test_cleanup_on_exit(self, tmp_path):
        """Test cleanup via atexit handler."""
        target = tmp_path / "test.txt"
        temp_path = target.with_suffix(".txt.tmp")

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"Test")

        # Simulate cleanup
        writer._cleanup()

        assert not temp_path.exists()
        assert not target.exists()

    def test_open_error(self, tmp_path):
        """Test handling open errors."""
        # Use invalid path
        target = tmp_path / "nonexistent" / "test.txt"

        writer = SafeFileWriter(target)
        with pytest.raises(FileIOError, match="Failed to open temporary file"):
            writer.open()

    def test_write_error(self, tmp_path):
        """Test handling write errors."""
        target = tmp_path / "test.txt"

        writer = SafeFileWriter(target)
        writer.open()

        # Mock file to raise error on write
        writer.file = MagicMock()
        writer.file.write.side_effect = OSError("Write failed")

        with pytest.raises(FileIOError, match="Failed to write data"):
            writer.write(b"Test")

    def test_rename_error(self, tmp_path):
        """Test handling rename errors."""
        target = tmp_path / "test.txt"

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"Test")

        # Mock rename to fail
        with patch.object(Path, "rename", side_effect=OSError("Rename failed")):
            with pytest.raises(FileIOError, match="Failed to rename temp file"):
                writer.close()


class TestPwrite:
    """Test pwrite function."""

    def test_pwrite_unix(self, tmp_path):
        """Test pwrite on Unix-like systems."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"0" * 100)

        with open(test_file, "r+b") as f:
            fd = f.fileno()

            # Write at offset 10
            written = pwrite(fd, b"HELLO", 10)

        assert written == 5
        content = test_file.read_bytes()
        assert content[10:15] == b"HELLO"
        assert content[0:10] == b"0" * 10
        assert content[15:20] == b"0" * 5

    @patch("os.pwrite", side_effect=OSError("pwrite failed"))
    def test_pwrite_error(self, mock_pwrite, tmp_path):
        """Test pwrite error handling."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")

        with open(test_file, "r+b") as f:
            with pytest.raises(FileIOError, match="pwrite failed"):
                pwrite(f.fileno(), b"data", 0)

    def test_pwrite_windows_fallback(self, tmp_path):
        """Test pwrite Windows fallback."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"0" * 100)

        # Temporarily remove pwrite to simulate Windows
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                fd = f.fileno()

                # Original position
                os.lseek(fd, 50, os.SEEK_SET)

                # Write at offset 10
                written = pwrite(fd, b"HELLO", 10)

                # Position should be restored
                assert os.lseek(fd, 0, os.SEEK_CUR) == 50
        finally:
            # Restore original pwrite if it existed
            if original_pwrite is not None:
                os.pwrite = original_pwrite

        assert written == 5
        content = test_file.read_bytes()
        assert content[10:15] == b"HELLO"

    @patch("os.lseek", side_effect=OSError("Seek failed"))
    def test_pwrite_windows_error(self, mock_lseek, tmp_path):
        """Test pwrite Windows fallback error."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")

        # Temporarily remove pwrite to simulate Windows
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                with pytest.raises(FileIOError, match="pwrite emulation failed"):
                    pwrite(f.fileno(), b"data", 0)
        finally:
            # Restore original pwrite if it existed
            if original_pwrite is not None:
                os.pwrite = original_pwrite


class TestSafeSeek:
    """Test safe_seek function."""

    def test_seek_set(self, tmp_path):
        """Test seeking from beginning."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"0123456789")

        with open(test_file, "rb") as f:
            pos = safe_seek(f, 5, os.SEEK_SET)
            assert pos == 5
            assert f.read(1) == b"5"

    def test_seek_cur(self, tmp_path):
        """Test seeking from current position."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"0123456789")

        with open(test_file, "rb") as f:
            f.read(3)  # Position at 3
            pos = safe_seek(f, 2, os.SEEK_CUR)
            assert pos == 5
            assert f.read(1) == b"5"

    def test_seek_end(self, tmp_path):
        """Test seeking from end."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"0123456789")

        with open(test_file, "rb") as f:
            pos = safe_seek(f, -3, os.SEEK_END)
            assert pos == 7
            assert f.read(3) == b"789"

    def test_seek_error(self, tmp_path):
        """Test seek error handling."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        with open(test_file, "rb") as f:
            # Mock seek to fail
            f.seek = MagicMock(side_effect=OSError("Seek failed"))

            with pytest.raises(FileIOError, match="Seek failed"):
                safe_seek(f, 0)


class TestGetFileSize:
    """Test get_file_size function."""

    def test_get_size(self, tmp_path):
        """Test getting file size."""
        test_file = tmp_path / "test.txt"
        test_data = b"x" * 1234
        test_file.write_bytes(test_data)

        size = get_file_size(test_file)
        assert size == 1234

    def test_get_size_string_path(self, tmp_path):
        """Test getting file size with string path."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test")

        size = get_file_size(str(test_file))
        assert size == 4

    def test_get_size_nonexistent(self, tmp_path):
        """Test getting size of non-existent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(FileIOError, match="Failed to get file size"):
            get_file_size(test_file)


class TestEnsureParentDir:
    """Test ensure_parent_dir function."""

    def test_create_parent(self, tmp_path):
        """Test creating parent directory."""
        test_file = tmp_path / "subdir" / "test.txt"

        ensure_parent_dir(test_file)

        assert test_file.parent.exists()
        assert test_file.parent.is_dir()

    def test_existing_parent(self, tmp_path):
        """Test with existing parent directory."""
        test_file = tmp_path / "test.txt"

        # Should not raise
        ensure_parent_dir(test_file)

    def test_nested_parents(self, tmp_path):
        """Test creating nested parent directories."""
        test_file = tmp_path / "a" / "b" / "c" / "test.txt"

        ensure_parent_dir(test_file)

        assert test_file.parent.exists()
        assert (tmp_path / "a" / "b").exists()

    @patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))
    def test_create_error(self, mock_mkdir, tmp_path):
        """Test error creating parent directory."""
        test_file = tmp_path / "subdir" / "test.txt"

        with pytest.raises(FileIOError, match="Failed to create parent directory"):
            ensure_parent_dir(test_file)
