"""Tests for StreamWriter edge cases to improve coverage."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stsw._core.meta import TensorMeta
from stsw.writer.writer import StreamWriter, TensorOrderError


class TestWriterStatsEdgeCases:
    """Test WriterStats edge cases."""

    def test_stats_without_psutil(self, tmp_path):
        """Test stats when psutil is not available."""
        # Save original psutil import
        import sys

        original_psutil = sys.modules.get("psutil")

        try:
            # Remove psutil from modules if it exists
            if "psutil" in sys.modules:
                del sys.modules["psutil"]

            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            # Write some data
            writer.write_block("test", b"x" * 40)

            # Mock psutil import to fail in the stats method
            def mock_import(name, *args, **kwargs):
                if name == "psutil":
                    raise ImportError("No module named 'psutil'")
                # Use the real import for everything else
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                # Get stats - should work without psutil
                stats = writer.stats()
                assert stats.written == 40
                assert stats.total == 40
                assert stats.rss_mb == 0  # Should be 0 when psutil not available

            writer.finalize_tensor("test")
            writer.close()
        finally:
            # Restore original psutil if it existed
            if original_psutil is not None:
                sys.modules["psutil"] = original_psutil

    def test_stats_zero_elapsed_time(self, tmp_path):
        """Test stats calculation with zero elapsed time."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Mock start time to be exactly now
        writer._start_time = time.time()

        # Get stats immediately
        stats = writer.stats()
        assert stats.mb_per_s == 0  # Should handle zero elapsed time
        assert stats.eta_s == 0

        writer.abort()

    def test_stats_zero_speed(self, tmp_path):
        """Test stats ETA calculation with zero speed."""
        meta = TensorMeta("test", "F32", (1000,), 0, 4000)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Don't write any data, so speed is 0
        writer._start_time = time.time() - 10  # 10 seconds ago

        stats = writer.stats()
        assert stats.written == 0
        assert stats.mb_per_s == 0
        assert stats.eta_s == 0  # Should be 0 when speed is 0

        writer.abort()


class TestWriterPaddingEdgeCases:
    """Test padding edge cases."""

    def test_write_exactly_aligned_tensor(self, tmp_path):
        """Test writing tensor that's exactly aligned."""
        # Create tensor that's exactly 64 bytes
        meta = TensorMeta("aligned", "F64", (8,), 0, 64)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], align=64)

        data = np.arange(8, dtype=np.float64).tobytes()
        assert len(data) == 64

        writer.write_block("aligned", data)
        writer.finalize_tensor("aligned")
        writer.close()

        # Check file size to ensure no extra padding
        file_size = (tmp_path / "test.st").stat().st_size
        header_size = writer._header_size
        assert file_size == header_size + 64  # No padding needed

    def test_write_multiple_aligned_tensors(self, tmp_path):
        """Test writing multiple exactly aligned tensors."""
        # All tensors are multiples of 64 bytes
        metas = [
            TensorMeta("tensor1", "F64", (8,), 0, 64),
            TensorMeta("tensor2", "F32", (32,), 64, 192),  # 128 bytes
            TensorMeta("tensor3", "I64", (24,), 192, 384),  # 192 bytes
        ]

        writer = StreamWriter.open(tmp_path / "test.st", metas, align=64)

        # Write all tensors
        writer.write_block("tensor1", b"x" * 64)
        writer.finalize_tensor("tensor1")

        writer.write_block("tensor2", b"y" * 128)
        writer.finalize_tensor("tensor2")

        writer.write_block("tensor3", b"z" * 192)
        writer.finalize_tensor("tensor3")

        writer.close()

        assert (tmp_path / "test.st").exists()

    def test_write_with_custom_alignment(self, tmp_path):
        """Test writing with custom alignment values."""
        # Test with alignment of 128
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], align=128)

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Should write 88 bytes of padding (40 + 88 = 128)
        writer.close()

        assert (tmp_path / "test.st").exists()


class TestWriterCRC32EdgeCases:
    """Test CRC32 edge cases."""

    def test_close_without_crc(self, tmp_path):
        """Test close path when CRC is disabled."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], crc32=False)

        # Verify CRC is not enabled
        assert not writer.enable_crc32
        assert len(writer._crc_calculators) == 0

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Close normally - pwrite will be called internally
        writer.close()

        # Verify file exists and is valid
        assert (tmp_path / "test.st").exists()

    def test_close_with_crc_update(self, tmp_path):
        """Test close with CRC updating header."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], crc32=True)

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Verify tensor has CRC after finalization
        assert writer.tensors[0].crc32 is not None

        # Close normally
        writer.close()

        # Verify file exists
        assert (tmp_path / "test.st").exists()

    def test_finalize_empty_tensor_with_crc(self, tmp_path):
        """Test finalizing empty tensor with CRC enabled."""
        meta = TensorMeta("empty", "F32", (0,), 0, 0)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], crc32=True)

        # Finalize without writing any data
        writer.finalize_tensor("empty")

        # Should have CRC value (0 for empty data)
        assert writer.tensors[0].crc32 == 0

        writer.close()


class TestWriterThreadSafety:
    """Test writer thread safety edge cases."""

    def test_concurrent_stats_access(self, tmp_path):
        """Test concurrent access to stats."""
        meta = TensorMeta("test", "F32", (1000,), 0, 4000)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        stats_results = []
        errors = []

        def get_stats():
            try:
                for _ in range(10):
                    stats = writer.stats()
                    stats_results.append(stats)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        # Start multiple threads accessing stats
        threads = [threading.Thread(target=get_stats) for _ in range(5)]
        for t in threads:
            t.start()

        # Write data concurrently
        for i in range(10):
            writer.write_block("test", b"x" * 400)
            time.sleep(0.001)

        # Wait for threads
        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
        assert len(stats_results) > 0

        writer.finalize_tensor("test")
        writer.close()

    def test_abort_during_write(self, tmp_path):
        """Test aborting while write is in progress."""
        meta = TensorMeta("test", "F32", (10000,), 0, 40000)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        def write_data():
            try:
                for _ in range(100):
                    writer.write_block("test", b"x" * 400)
            except Exception:
                pass  # Expected if aborted

        # Start writing in thread
        write_thread = threading.Thread(target=write_data)
        write_thread.start()

        # Abort while writing
        time.sleep(0.01)
        writer.abort()

        write_thread.join()

        # File should not exist
        assert not (tmp_path / "test.st").exists()


class TestWriterErrorRecovery:
    """Test writer error recovery."""

    def test_write_after_finalize_all(self, tmp_path):
        """Test writing after all tensors finalized."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Try to write more data - should fail
        with pytest.raises(TensorOrderError, match="already finalized"):
            writer.write_block("test", b"more data")

        # But close should still work
        writer.close()
        assert (tmp_path / "test.st").exists()

    def test_partial_close_recovery(self, tmp_path):
        """Test recovery when close partially fails."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Mock file writer close to fail
        writer._file_writer.close = MagicMock(side_effect=OSError("Close failed"))

        # Close should propagate error
        with pytest.raises(OSError, match="Close failed"):
            writer.close()

        # Writer should be in closed state
        assert writer._file_writer is not None  # Failed to clean up

    def test_header_size_calculation(self, tmp_path):
        """Test header size calculation edge cases."""
        # Create many tensors to test large header
        metas = []
        for i in range(100):
            name = f"tensor_{i:03d}"
            metas.append(TensorMeta(name, "F32", (10, 10), i * 512, (i + 1) * 512))

        writer = StreamWriter.open(tmp_path / "test.st", metas)

        # Header should be properly sized
        assert writer._header_size > 0
        assert writer._data_start == writer._header_size

        writer.abort()
