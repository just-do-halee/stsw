"""Final tests to achieve full coverage for fileio.py."""

from pathlib import Path
from unittest.mock import patch

import pytest

from stsw.io.fileio import FileIOError, SafeFileWriter, pwrite


class TestFileIOFinalCoverage:
    """Final tests for complete fileio coverage."""

    def test_safe_file_writer_reopen(self, tmp_path):
        """Test reopening an already open file."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        # Open once
        file1 = writer.open()
        assert file1 is not None

        # Try to open again - should return same file
        file2 = writer.open()
        assert file2 is file1

        writer.close()

    def test_safe_file_writer_close_already_closed(self, tmp_path):
        """Test closing already closed writer."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        writer.open()
        writer.write(b"test")
        writer.close()

        # Close again - should be no-op
        writer.close()

        assert target.exists()

    def test_safe_file_writer_close_aborted(self, tmp_path):
        """Test closing after abort."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        writer.open()
        writer.abort()

        # Close after abort - should be no-op
        writer.close()

        assert not target.exists()

    def test_safe_file_writer_abort_already_closed(self, tmp_path):
        """Test aborting already closed writer."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        writer.open()
        writer.write(b"test")
        writer.close()

        # Abort after close - should be no-op
        writer.abort()

        # File should still exist
        assert target.exists()

    def test_safe_file_writer_flush_no_file(self, tmp_path):
        """Test flush when file is not open."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        # Flush without opening - should be no-op
        writer.flush()

    def test_safe_file_writer_fsync_no_file(self, tmp_path):
        """Test fsync when file is not open."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        # Fsync without opening - should be no-op
        writer.fsync()

    def test_pwrite_unix_error(self, tmp_path):
        """Test pwrite error on Unix systems."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")

        with open(test_file, "r+b") as f:
            fd = f.fileno()

            # Mock os.pwrite to exist but fail
            with patch("os.pwrite", side_effect=OSError("pwrite failed")):
                with pytest.raises(FileIOError, match="pwrite failed"):
                    pwrite(fd, b"new", 0)

    def test_windows_rename_target_unlink_fails(self, tmp_path):
        """Test Windows rename when target exists and unlink fails."""
        target = tmp_path / "test.txt"
        target.write_bytes(b"existing")

        writer = SafeFileWriter(target)
        writer.open()
        writer.write(b"new content")

        with patch("platform.system", return_value="Windows"):
            # Make exists() return True but unlink fail
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "unlink", side_effect=OSError("Access denied")):
                    # Rename should be attempted anyway
                    with patch.object(Path, "rename") as mock_rename:
                        writer.close()
                        mock_rename.assert_called_once()

    def test_abort_no_file(self, tmp_path):
        """Test abort when file was never opened."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        # Abort without opening
        writer.abort()

        # Should not error
        assert not target.exists()

    def test_cleanup_already_aborted(self, tmp_path):
        """Test cleanup when already aborted."""
        target = tmp_path / "test.txt"
        writer = SafeFileWriter(target)

        writer.open()
        writer.abort()

        # Cleanup should be no-op
        writer._cleanup()

    def test_context_manager_no_exception_no_close(self, tmp_path):
        """Test context manager exit without explicit close."""
        target = tmp_path / "test.txt"

        with SafeFileWriter(target) as writer:
            writer.write(b"test data")
            # Don't call close explicitly

        # Should auto-close and create file
        assert target.exists()
        assert target.read_bytes() == b"test data"
