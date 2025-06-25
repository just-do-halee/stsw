"""Tests for writer error conditions."""

import pytest

from stsw._core.meta import TensorMeta
from stsw.writer.writer import (
    LengthMismatchError,
    StreamWriter,
    TensorOrderError,
)


class TestWriterErrors:
    """Test writer error handling."""

    def test_write_too_much_data(self, tmp_path):
        """Test writing more data than tensor size."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Try to write more than 40 bytes
        with pytest.raises(LengthMismatchError, match="expects 40 more bytes"):
            writer.write_block("test", b"x" * 50)

        writer.abort()

    def test_finalize_without_all_data(self, tmp_path):
        """Test finalizing with incomplete data."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Write only partial data
        writer.write_block("test", b"x" * 30)

        # Try to finalize
        with pytest.raises(LengthMismatchError, match="expects 40 bytes"):
            writer.finalize_tensor("test")

        writer.abort()

    def test_write_tensor_twice(self, tmp_path):
        """Test writing to already finalized tensor."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Write and finalize
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Try to write again
        with pytest.raises(TensorOrderError, match="already finalized"):
            writer.write_block("test", b"x" * 40)

        writer.close()

    def test_skip_tensor(self, tmp_path):
        """Test trying to write tensor out of order."""
        meta1 = TensorMeta("tensor1", "F32", (10,), 0, 40)
        meta2 = TensorMeta("tensor2", "I64", (5,), 64, 104)

        writer = StreamWriter.open(tmp_path / "test.st", [meta1, meta2])

        # Try to write tensor2 first
        with pytest.raises(TensorOrderError, match="Expected tensor 'tensor1'"):
            writer.write_block("tensor2", b"x" * 40)

        writer.abort()

    def test_finalize_all_tensors_already(self, tmp_path):
        """Test finalizing when all tensors are done."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Write and finalize
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")

        # Try to finalize again
        with pytest.raises(TensorOrderError, match="All tensors already finalized"):
            writer.finalize_tensor("test")

        writer.close()

    def test_multiple_tensor_order(self, tmp_path):
        """Test writing multiple tensors in correct order."""
        metas = [
            TensorMeta("tensor1", "F32", (10,), 0, 40),
            TensorMeta("tensor2", "I64", (5,), 64, 104),
            TensorMeta("tensor3", "F16", (2, 2), 128, 136),
        ]

        writer = StreamWriter.open(tmp_path / "test.st", metas)

        # Write in order
        writer.write_block("tensor1", b"x" * 40)
        writer.finalize_tensor("tensor1")

        writer.write_block("tensor2", b"y" * 40)
        writer.finalize_tensor("tensor2")

        writer.write_block("tensor3", b"z" * 8)
        writer.finalize_tensor("tensor3")

        writer.close()

        assert (tmp_path / "test.st").exists()

    def test_zero_size_tensor(self, tmp_path):
        """Test writing zero-size tensor."""
        meta = TensorMeta("empty", "F32", (0,), 0, 0)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Finalize without writing (zero size)
        writer.finalize_tensor("empty")
        writer.close()

        assert (tmp_path / "test.st").exists()

    def test_repr_str(self, tmp_path):
        """Test string representation."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        repr_str = repr(writer)
        assert "StreamWriter" in repr_str

        str_repr = str(writer)
        assert isinstance(str_repr, str)

        writer.abort()
