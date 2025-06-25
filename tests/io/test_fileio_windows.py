"""Windows-specific tests for file I/O module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from stsw.io.fileio import FileIOError, SafeFileWriter, pwrite


class TestWindowsSpecificBehavior:
    """Test Windows-specific file I/O behavior."""

    def test_windows_atomic_rename_file_exists(self, tmp_path):
        """Test Windows atomic rename when target file exists."""
        target = tmp_path / "existing.txt"
        target.write_bytes(b"old content")

        with patch("platform.system", return_value="Windows"):
            writer = SafeFileWriter(target)
            writer.open()
            writer.write(b"new content")

            # Mock unlink to verify it's called
            with patch.object(Path, "unlink") as mock_unlink:
                writer.close()
                mock_unlink.assert_called_once()

        assert target.read_bytes() == b"new content"

    def test_windows_rename_permission_error(self, tmp_path):
        """Test Windows rename with permission error on unlink."""
        target = tmp_path / "locked.txt"
        target.write_bytes(b"old content")

        with patch("platform.system", return_value="Windows"):
            writer = SafeFileWriter(target)
            writer.open()
            writer.write(b"new content")

            # Mock unlink to raise permission error
            with patch.object(
                Path, "unlink", side_effect=PermissionError("Access denied")
            ):
                with pytest.raises(FileIOError):
                    writer.close()

    def test_pwrite_windows_seek_restore(self, tmp_path):
        """Test pwrite Windows fallback preserves file position."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"0" * 100)

        # Remove pwrite to force Windows fallback
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                fd = f.fileno()

                # Set initial position
                os.lseek(fd, 25, os.SEEK_SET)

                # Write at different position
                written = pwrite(fd, b"HELLO", 50)

                # Check position was restored
                current_pos = os.lseek(fd, 0, os.SEEK_CUR)
                assert current_pos == 25
                assert written == 5

            # Verify data was written at correct position
            content = test_file.read_bytes()
            assert content[50:55] == b"HELLO"
            assert content[25:30] == b"00000"  # Original position unchanged

        finally:
            if original_pwrite is not None:
                os.pwrite = original_pwrite

    def test_pwrite_windows_seek_error(self, tmp_path):
        """Test pwrite Windows fallback with seek error."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")

        # Remove pwrite to force Windows fallback
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                # Mock lseek to fail on first call
                with patch("os.lseek", side_effect=OSError("Seek failed")):
                    with pytest.raises(FileIOError, match="pwrite emulation failed"):
                        pwrite(f.fileno(), b"data", 0)
        finally:
            if original_pwrite is not None:
                os.pwrite = original_pwrite

    def test_pwrite_windows_write_error(self, tmp_path):
        """Test pwrite Windows fallback with write error."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")

        # Remove pwrite to force Windows fallback
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                fd = f.fileno()

                # Mock os.write to fail
                with patch("os.write", side_effect=OSError("Write failed")):
                    with pytest.raises(FileIOError, match="pwrite emulation failed"):
                        pwrite(fd, b"new data", 0)
        finally:
            if original_pwrite is not None:
                os.pwrite = original_pwrite

    def test_pwrite_windows_restore_position_error(self, tmp_path):
        """Test pwrite Windows fallback with error restoring position."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")

        # Remove pwrite to force Windows fallback
        original_pwrite = getattr(os, "pwrite", None)
        if hasattr(os, "pwrite"):
            delattr(os, "pwrite")

        try:
            with open(test_file, "r+b") as f:
                fd = f.fileno()

                # Create a mock that fails on the third call (restore position)
                call_count = 0

                def mock_lseek(fd, offset, whence):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 3:  # Third call is restore
                        raise OSError("Seek failed")
                    elif call_count == 1:  # First call is get current
                        return 0
                    else:  # Second call is seek to write position
                        return offset

                with patch("os.lseek", side_effect=mock_lseek):
                    with patch("os.write", return_value=4):
                        with pytest.raises(
                            FileIOError, match="pwrite emulation failed"
                        ):
                            pwrite(fd, b"data", 0)
        finally:
            if original_pwrite is not None:
                os.pwrite = original_pwrite

    def test_safe_file_writer_windows_temp_cleanup(self, tmp_path):
        """Test SafeFileWriter cleans up temp file on Windows."""
        target = tmp_path / "test.txt"
        temp_path = target.with_suffix(".txt.tmp")

        with patch("platform.system", return_value="Windows"):
            writer = SafeFileWriter(target)
            writer.open()
            writer.write(b"test data")

            # Temp file should exist
            assert temp_path.exists()

            # Abort should clean up
            writer.abort()
            assert not temp_path.exists()

    def test_safe_file_writer_windows_double_abort(self, tmp_path):
        """Test SafeFileWriter double abort on Windows."""
        target = tmp_path / "test.txt"

        with patch("platform.system", return_value="Windows"):
            writer = SafeFileWriter(target)
            writer.open()
            writer.write(b"test")

            # First abort
            writer.abort()

            # Second abort should not error
            writer.abort()

    def test_safe_file_writer_cleanup_error(self, tmp_path):
        """Test SafeFileWriter cleanup when unlink fails."""
        target = tmp_path / "test.txt"
        temp_path = target.with_suffix(".txt.tmp")

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"test")

        # Mock unlink to fail
        with patch.object(Path, "unlink", side_effect=OSError("Access denied")):
            # Should not raise, just silently fail
            writer.abort()

    def test_safe_file_writer_windows_rename_both_fail(self, tmp_path):
        """Test Windows rename when both unlink and rename fail."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"existing")

        with patch("platform.system", return_value="Windows"):
            writer = SafeFileWriter(target)
            writer.open()
            writer.write(b"new content")

            # Mock both operations to fail
            with patch.object(Path, "unlink", side_effect=OSError("Unlink failed")):
                with patch.object(Path, "rename", side_effect=OSError("Rename failed")):
                    with pytest.raises(FileIOError, match="Failed to rename temp file"):
                        writer.close()
