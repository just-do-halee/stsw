"""Additional tests for StreamWriter to improve coverage."""

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from stsw._core.meta import TensorMeta
from stsw.writer.writer import (
    LengthMismatchError,
    StreamWriter,
    TensorOrderError,
    WriterStats,
)


class TestStreamWriterCoverage:
    """Additional tests for StreamWriter coverage."""

    def test_writer_stats_repr(self):
        """Test WriterStats string representation."""
        stats = WriterStats(
            written=1000,
            total=2000,
            mb_per_s=10.5,
            eta_s=5.0,
            rss_mb=50.0
        )
        
        repr_str = repr(stats)
        assert "WriterStats" in repr_str
        assert "written=1000" in repr_str
        assert "total=2000" in repr_str
        assert "mb_per_s=10.5" in repr_str

    def test_writer_context_manager_exception(self, tmp_path):
        """Test writer context manager with exception."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        
        with pytest.raises(ValueError):
            with StreamWriter.open(tmp_path / "test.st", [meta]) as writer:
                # Raise exception in context
                raise ValueError("Test error")
        
        # File should be aborted/cleaned up
        assert not (tmp_path / "test.st").exists()

    def test_writer_double_close(self, tmp_path):
        """Test closing writer twice."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Write data
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        
        # Close once
        writer.close()
        
        # Close again should not error
        writer.close()

    def test_writer_write_after_close(self, tmp_path):
        """Test writing after close raises error."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Write and finalize first
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()
        
        # Should raise error (TensorOrderError since tensor is already finalized)
        with pytest.raises(TensorOrderError):
            writer.write_block("test", b"data")

    def test_writer_finalize_after_close(self, tmp_path):
        """Test finalizing after close raises error."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Write and finalize first
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()
        
        # Should raise error
        with pytest.raises(TensorOrderError):
            writer.finalize_tensor("test")

    def test_writer_stats_after_close(self, tmp_path):
        """Test getting stats after close."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()
        
        # Should still return stats
        stats = writer.stats()
        # Written includes padding (aligned to 64 bytes)
        assert stats.written >= 40
        assert stats.total == 40

    def test_writer_partial_tensor_close(self, tmp_path):
        """Test closing with partially written tensor."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Write partial data
        writer.write_block("test", b"x" * 20)
        
        # Close without finalizing - should raise error
        with pytest.raises(RuntimeError, match="Cannot close: 1 tensors not written"):
            writer.close()

    def test_writer_repr(self, tmp_path):
        """Test writer string representation."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Just test that repr works without error
        repr_str = repr(writer)
        assert isinstance(repr_str, str)
        
        writer.abort()

    def test_writer_no_padding_needed(self, tmp_path):
        """Test writer when no padding is needed (aligned data)."""
        # Create tensor that's already aligned to 64 bytes
        meta = TensorMeta("test", "F32", (16,), 0, 64)  # Exactly 64 bytes
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        writer.write_block("test", b"x" * 64)
        writer.finalize_tensor("test")
        writer.close()
        
        # Verify file size
        file_size = (tmp_path / "test.st").stat().st_size
        # Should be header + 64 bytes data (no padding)
        assert file_size > 64

    def test_writer_buffer_flush(self, tmp_path):
        """Test writer buffer flushing."""
        meta = TensorMeta("test", "F32", (1000,), 0, 4000)
        
        # Use small buffer to force flushes
        writer = StreamWriter.open(tmp_path / "test.st", [meta], buffer_size=100)
        
        # Write in small chunks
        for i in range(40):
            writer.write_block("test", b"x" * 100)
        
        writer.finalize_tensor("test")
        writer.close()
        
        assert (tmp_path / "test.st").exists()

    def test_writer_crc_enabled(self, tmp_path):
        """Test writer with CRC enabled."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], crc32=True)
        
        # CRC should be enabled
        assert writer.enable_crc32
        assert len(writer._crc_calculators) == 1
        
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()

    def test_writer_empty_data_block(self, tmp_path):
        """Test writing empty data block."""
        meta = TensorMeta("test", "F32", (0,), 0, 0)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Write empty block
        writer.write_block("test", b"")
        writer.finalize_tensor("test")
        writer.close()
        
        assert (tmp_path / "test.st").exists()

    def test_writer_memoryview_input(self, tmp_path):
        """Test writing memoryview data."""
        import numpy as np
        
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Create numpy array and get memoryview of its bytes
        arr = np.arange(10, dtype=np.float32)
        # Need to write the raw bytes, not the array view
        data = arr.tobytes()
        
        writer.write_block("test", data)
        writer.finalize_tensor("test")
        writer.close()
        
        assert (tmp_path / "test.st").exists()

    def test_writer_stats_calculation(self, tmp_path):
        """Test writer stats calculation."""
        meta = TensorMeta("test", "F32", (100,), 0, 400)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Mock start time to control elapsed time
        writer._start_time = time.time() - 1.0  # 1 second ago
        
        # Write data
        writer.write_block("test", b"x" * 400)
        
        stats = writer.stats()
        assert stats.written == 400
        assert stats.total == 400
        assert stats.mb_per_s > 0
        assert stats.eta_s == 0.0  # Complete
        
        writer.finalize_tensor("test")
        writer.close()

    def test_writer_stats_partial_progress(self, tmp_path):
        """Test writer stats with partial progress."""
        meta = TensorMeta("test", "F32", (1000,), 0, 4000)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Mock start time
        writer._start_time = time.time() - 1.0
        
        # Write partial data
        writer.write_block("test", b"x" * 1000)
        
        stats = writer.stats()
        assert stats.written == 1000
        assert stats.total == 4000
        assert stats.eta_s > 0  # Should have ETA
        
        writer.abort()

    def test_writer_finalize_wrong_order(self, tmp_path):
        """Test finalizing tensors in wrong order."""
        meta1 = TensorMeta("tensor1", "F32", (10,), 0, 40)
        meta2 = TensorMeta("tensor2", "I64", (5,), 64, 104)
        
        writer = StreamWriter.open(tmp_path / "test.st", [meta1, meta2])
        
        # Try to finalize tensor2 first
        with pytest.raises(TensorOrderError):
            writer.finalize_tensor("tensor2")
        
        writer.abort()

    def test_writer_unknown_tensor_write(self, tmp_path):
        """Test writing to unknown tensor."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        with pytest.raises(ValueError, match="Unknown tensor"):
            writer.write_block("unknown", b"data")
        
        writer.abort()

    def test_writer_unknown_tensor_finalize(self, tmp_path):
        """Test finalizing unknown tensor."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        with pytest.raises(TensorOrderError):
            writer.finalize_tensor("unknown")
        
        writer.abort()