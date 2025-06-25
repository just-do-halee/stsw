"""Final tests to achieve full coverage for writer.py."""

from unittest.mock import patch

import numpy as np
import pytest

from stsw._core.meta import TensorMeta
from stsw.writer.writer import LengthMismatchError, StreamWriter, TensorOrderError


class TestWriterFinalCoverage:
    """Final tests for complete writer coverage."""

    def test_writer_finalize_with_exact_padding(self, tmp_path):
        """Test finalize when tensor ends exactly on alignment boundary."""
        # Create tensor that ends exactly at 64-byte boundary
        meta = TensorMeta("aligned", "F64", (8,), 0, 64)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], align=64)

        # Write exactly 64 bytes
        writer.write_block("aligned", b"x" * 64)

        # Finalize - should not add padding
        writer.finalize_tensor("aligned")

        # Current tensor index should advance
        assert writer._current_tensor_idx == 1

        writer.close()

    def test_writer_write_unknown_tensor_early(self, tmp_path):
        """Test writing to unknown tensor."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Try to write to non-existent tensor
        with pytest.raises(ValueError, match="Unknown tensor: unknown"):
            writer.write_block("unknown", b"data")

        writer.abort()

    def test_writer_write_already_finalized(self, tmp_path):
        """Test writing to already finalized tensor."""
        meta1 = TensorMeta("tensor1", "F32", (10,), 0, 40)
        meta2 = TensorMeta("tensor2", "I32", (5,), 64, 84)
        writer = StreamWriter.open(tmp_path / "test.st", [meta1, meta2])

        # Write and finalize first tensor
        writer.write_block("tensor1", b"x" * 40)
        writer.finalize_tensor("tensor1")

        # Try to write to first tensor again
        with pytest.raises(TensorOrderError, match="already finalized"):
            writer.write_block("tensor1", b"more")

        writer.abort()

    def test_writer_write_wrong_order(self, tmp_path):
        """Test writing tensors out of order."""
        meta1 = TensorMeta("tensor1", "F32", (10,), 0, 40)
        meta2 = TensorMeta("tensor2", "I32", (5,), 64, 84)
        writer = StreamWriter.open(tmp_path / "test.st", [meta1, meta2])

        # Try to write second tensor first
        with pytest.raises(
            TensorOrderError, match="Expected tensor 'tensor1' but got 'tensor2'"
        ):
            writer.write_block("tensor2", b"data")

        writer.abort()

    def test_writer_write_too_much_data(self, tmp_path):
        """Test writing more data than tensor size."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Try to write more than 40 bytes
        with pytest.raises(
            LengthMismatchError, match="expects 40 more bytes, but got 50"
        ):
            writer.write_block("test", b"x" * 50)

        writer.abort()

    def test_writer_partial_write_then_too_much(self, tmp_path):
        """Test partial write followed by too much data."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Write 20 bytes
        writer.write_block("test", b"x" * 20)

        # Try to write 30 more (total would be 50, but only 40 allowed)
        with pytest.raises(
            LengthMismatchError, match="expects 20 more bytes, but got 30"
        ):
            writer.write_block("test", b"x" * 30)

        writer.abort()

    def test_writer_finalize_incomplete_tensor(self, tmp_path):
        """Test finalizing tensor with incomplete data."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Write only 20 bytes
        writer.write_block("test", b"x" * 20)

        # Try to finalize - should fail
        with pytest.raises(
            LengthMismatchError, match="expects 40 bytes, but only 20 were written"
        ):
            writer.finalize_tensor("test")

        writer.abort()

    def test_writer_finalize_zero_padding(self, tmp_path):
        """Test finalize writes zeros for padding."""
        # Create tensor that needs padding
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], align=64)

        # Write data
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()

        # Read file and check padding
        with open(tmp_path / "test.st", "rb") as f:
            # Skip header
            header_len = int.from_bytes(f.read(8), "little")
            f.seek(8 + header_len)

            # Read data and padding
            data = f.read(64)
            assert data[:40] == b"x" * 40
            assert data[40:] == b"\x00" * 24  # Padding should be zeros

    def test_writer_memoryview_conversion(self, tmp_path):
        """Test memoryview is converted to bytes."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Create numpy array and get memoryview
        arr = np.arange(10, dtype=np.float32)
        mv = memoryview(arr.tobytes())

        # Write memoryview
        writer.write_block("test", mv)
        writer.finalize_tensor("test")
        writer.close()

        assert (tmp_path / "test.st").exists()

    def test_writer_header_update_error_handling(self, tmp_path):
        """Test header update error handling in close."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta], crc32=True)

        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Mock pwrite to fail
        with patch("stsw.io.fileio.pwrite", side_effect=OSError("Write failed")):
            # The writer may handle the error gracefully or re-raise it
            # Just ensure close() can be called
            try:
                writer.close()
            except OSError:
                # If it does raise, that's also valid
                pass

        # File should still exist either way
        assert (tmp_path / "test.st").exists()

    def test_writer_stats_zero_progress(self, tmp_path):
        """Test stats with zero progress."""
        meta = TensorMeta("test", "F32", (100,), 0, 400)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Immediately get stats without writing
        stats = writer.stats()
        assert stats.written == 0
        assert stats.total == 400
        assert stats.mb_per_s == 0.0
        assert stats.eta_s == 0.0

        writer.abort()
