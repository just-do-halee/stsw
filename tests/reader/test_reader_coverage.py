"""Additional tests for StreamReader to improve coverage."""

import struct
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stsw._core.header import HeaderError, build_header
from stsw._core.meta import TensorMeta
from stsw.reader.reader import StreamReader


class TestStreamReaderInit:
    """Test StreamReader initialization."""

    def test_file_not_found(self):
        """Test initializing with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            StreamReader("nonexistent.safetensors")

    def test_incomplete_file_rejected(self, tmp_path):
        """Test rejecting incomplete file by default."""
        # Create file with incomplete marker
        test_file = tmp_path / "incomplete.safetensors"

        meta = TensorMeta("test", "F32", (10,), 0, 40)
        header = build_header([meta], incomplete=True)

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"x" * 40)

        with pytest.raises(HeaderError, match="File is marked as incomplete"):
            StreamReader(test_file)

    def test_header_too_small(self, tmp_path):
        """Test file too small for header."""
        test_file = tmp_path / "tiny.safetensors"
        test_file.write_bytes(b"short")

        with pytest.raises(HeaderError, match="File too small"):
            StreamReader(test_file)

    def test_header_parse_error(self, tmp_path):
        """Test header parsing error."""
        test_file = tmp_path / "bad_header.safetensors"

        # Write invalid header length
        with open(test_file, "wb") as f:
            f.write(struct.pack("<Q", 100))  # Header length
            f.write(b"not json")  # Invalid JSON

        with pytest.raises(HeaderError, match="Failed to parse header"):
            StreamReader(test_file)

    def test_init_without_mmap(self, tmp_path):
        """Test initialization without memory mapping."""
        test_file = tmp_path / "test.safetensors"

        meta = TensorMeta("test", "F32", (5,), 0, 20)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(np.arange(5, dtype=np.float32).tobytes())

        reader = StreamReader(test_file, mmap=False)
        assert reader._mmap is None
        reader.close()

    def test_properties(self, tmp_path):
        """Test reader properties."""
        test_file = tmp_path / "test.safetensors"

        meta = TensorMeta("test", "F32", (5,), 0, 20)
        header = build_header([meta], metadata={"key": "value"})

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(np.zeros(5, dtype=np.float32).tobytes())

        with StreamReader(test_file) as reader:
            assert reader.version == "1.0"
            assert reader.metadata == {"key": "value"}
            assert len(reader) == 1


class TestStreamReaderDataAccess:
    """Test data access methods."""

    def setup_test_file(self, tmp_path, with_crc=False):
        """Create a test safetensors file."""
        test_file = tmp_path / "test.safetensors"

        # Create test data
        data1 = np.arange(10, dtype=np.float32)
        data2 = np.arange(20, dtype=np.int64)

        # Create metadata
        metas = [
            TensorMeta("tensor1", "F32", data1.shape, 0, data1.nbytes),
            TensorMeta("tensor2", "I64", data2.shape, 64, 64 + data2.nbytes),
        ]

        # Build header
        header = build_header(metas, align=64)

        # Write file
        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data1.tobytes())
            f.write(b"\x00" * (64 - data1.nbytes))  # Padding
            f.write(data2.tobytes())

        return test_file, data1, data2

    def test_keys_method(self, tmp_path):
        """Test keys() method."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            keys = reader.keys()
            assert keys == ["tensor1", "tensor2"]

    def test_meta_method(self, tmp_path):
        """Test meta() method."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            meta1 = reader.meta("tensor1")
            assert meta1.name == "tensor1"
            assert meta1.dtype == "F32"
            assert meta1.shape == (10,)

            meta2 = reader.meta("tensor2")
            assert meta2.name == "tensor2"
            assert meta2.dtype == "I64"

    def test_meta_not_found(self, tmp_path):
        """Test meta() with non-existent tensor."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            with pytest.raises(KeyError, match="Tensor 'missing' not found"):
                reader.meta("missing")

    def test_get_slice_method(self, tmp_path):
        """Test get_slice() method."""
        test_file, data1, data2 = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            # Get first tensor
            slice1 = reader.get_slice("tensor1")
            assert bytes(slice1) == data1.tobytes()

            # Get second tensor
            slice2 = reader.get_slice("tensor2")
            assert bytes(slice2) == data2.tobytes()

    def test_get_slice_not_found(self, tmp_path):
        """Test get_slice() with non-existent tensor."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            with pytest.raises(KeyError, match="Tensor 'missing' not found"):
                reader.get_slice("missing")

    def test_get_slice_without_mmap(self, tmp_path):
        """Test get_slice() without memory mapping."""
        test_file, data1, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file, mmap=False) as reader:
            # mmap=False means no memory mapping
            assert reader._mmap is None

            # Should still work using direct file read
            slice1 = reader.get_slice("tensor1")
            assert bytes(slice1) == data1.tobytes()

    def test_get_slice_error(self, tmp_path):
        """Test get_slice() with read error."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            # Mock mmap to raise error
            reader._mmap = MagicMock()
            reader._mmap.get_slice.side_effect = Exception("Read error")

            with pytest.raises(Exception, match="Read error"):
                reader.get_slice("tensor1")

    def test_to_numpy_method(self, tmp_path):
        """Test to_numpy() method."""
        test_file, data1, data2 = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            # Load first tensor
            arr1 = reader.to_numpy("tensor1")
            assert isinstance(arr1, np.ndarray)
            assert arr1.dtype == np.float32
            assert arr1.shape == (10,)
            assert np.array_equal(arr1, data1)

            # Load second tensor
            arr2 = reader.to_numpy("tensor2")
            assert arr2.dtype == np.int64
            assert arr2.shape == (20,)
            assert np.array_equal(arr2, data2)

    def test_to_numpy_bf16(self, tmp_path):
        """Test to_numpy() with BF16 dtype warning."""
        # Test that BF16 conversion produces a warning
        from stsw._core.dtype import to_numpy

        with pytest.warns(UserWarning, match="BF16 is not natively supported"):
            dtype = to_numpy("BF16")
            assert dtype == np.dtype("float32")

    @pytest.mark.skipif(not hasattr(np, "__version__"), reason="NumPy not available")
    def test_iterator(self, tmp_path):
        """Test iterating over tensor names."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            names = list(reader)
            assert names == ["tensor1", "tensor2"]

    def test_context_manager(self, tmp_path):
        """Test using reader as context manager."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        with StreamReader(test_file) as reader:
            assert reader._mmap is not None

        # After context, mmap should be closed
        assert reader._mmap is None

    def test_close_method(self, tmp_path):
        """Test explicit close."""
        test_file, _, _ = self.setup_test_file(tmp_path)

        reader = StreamReader(test_file)
        assert reader._mmap is not None

        reader.close()
        assert reader._mmap is None

        # Close again should not error
        reader.close()


class TestStreamReaderCRC:
    """Test CRC verification functionality."""

    def test_verify_crc_success(self, tmp_path):
        """Test successful CRC verification."""
        from stsw._core.crc32 import compute_crc32

        test_file = tmp_path / "crc.safetensors"

        # Create test data
        data = np.arange(10, dtype=np.float32)
        crc = compute_crc32(data.tobytes())

        # Create metadata with CRC
        meta = TensorMeta("tensor", "F32", data.shape, 0, data.nbytes, crc32=crc)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data.tobytes())

        # Read with CRC verification
        with StreamReader(test_file, verify_crc=True) as reader:
            slice_data = reader.get_slice("tensor")
            assert bytes(slice_data) == data.tobytes()

    def test_verify_crc_mismatch(self, tmp_path):
        """Test CRC verification failure."""
        test_file = tmp_path / "bad_crc.safetensors"

        # Create test data
        data = np.arange(10, dtype=np.float32)
        wrong_crc = 12345  # Wrong CRC

        # Create metadata with wrong CRC
        meta = TensorMeta("tensor", "F32", data.shape, 0, data.nbytes, crc32=wrong_crc)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data.tobytes())

        # Read with CRC verification
        with StreamReader(test_file, verify_crc=True) as reader:
            with pytest.raises(ValueError, match="CRC32 mismatch"):
                reader.get_slice("tensor")

    def test_crc_verification_caching(self, tmp_path):
        """Test that CRC is only verified once per tensor."""
        from stsw._core.crc32 import compute_crc32

        test_file = tmp_path / "crc_cache.safetensors"

        data = np.arange(10, dtype=np.float32)
        crc = compute_crc32(data.tobytes())

        meta = TensorMeta("tensor", "F32", data.shape, 0, data.nbytes, crc32=crc)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data.tobytes())

        with StreamReader(test_file, verify_crc=True) as reader:
            # First access verifies CRC
            slice1 = reader.get_slice("tensor")
            assert "tensor" in reader._crc_verified

            # Second access should use cached result
            slice2 = reader.get_slice("tensor")
            assert bytes(slice1) == bytes(slice2)


class TestStreamReaderTorch:
    """Test PyTorch tensor loading."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch not available",
    )
    def test_to_torch_cpu(self, tmp_path):
        """Test loading tensor to CPU."""
        import torch

        test_file = tmp_path / "torch.safetensors"

        data = np.arange(10, dtype=np.float32)
        meta = TensorMeta("tensor", "F32", data.shape, 0, data.nbytes)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data.tobytes())

        with StreamReader(test_file) as reader:
            tensor = reader.to_torch("tensor", device="cpu")
            assert tensor.device.type == "cpu"
            assert tensor.shape == (10,)
            assert torch.allclose(tensor, torch.from_numpy(data))

    def test_to_torch_no_torch_installed(self, tmp_path):
        """Test to_torch when PyTorch is not installed."""
        test_file = tmp_path / "test.safetensors"

        meta = TensorMeta("tensor", "F32", (5,), 0, 20)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(np.zeros(5, dtype=np.float32).tobytes())

        with patch.dict(sys.modules, {"torch": None}):
            with StreamReader(test_file) as reader:
                with pytest.raises(ImportError):
                    reader.to_torch("tensor")
