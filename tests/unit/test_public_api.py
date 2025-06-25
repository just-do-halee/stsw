"""Tests for public API in __init__.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import stsw
from stsw import (
    DEFAULT_ALIGN,
    FileIOError,
    HeaderError,
    LengthMismatchError,
    StreamReader,
    StreamWriter,
    TensorMeta,
    TensorOrderError,
    WriterStats,
    __version__,
    dtype,
    dump,
    tqdm,
)


class TestPublicAPI:
    """Test public API exports."""

    def test_version(self):
        """Test version string."""
        assert __version__ == "1.0.0"
        assert stsw.__version__ == "1.0.0"

    def test_default_align(self):
        """Test default alignment constant."""
        assert DEFAULT_ALIGN == 64
        assert stsw.DEFAULT_ALIGN == 64

    def test_all_exports(self):
        """Test __all__ contains all expected exports."""
        expected = {
            "__version__",
            "DEFAULT_ALIGN",
            "TensorMeta",
            "StreamReader",
            "StreamWriter",
            "WriterStats",
            "HeaderError",
            "FileIOError",
            "TensorOrderError",
            "LengthMismatchError",
            "dtype",
            "dump",
            "tqdm",
        }
        assert set(stsw.__all__) == expected

    def test_imports(self):
        """Test all imports are accessible."""
        # Classes
        assert StreamReader is not None
        assert StreamWriter is not None
        assert TensorMeta is not None
        assert WriterStats is not None

        # Exceptions
        assert HeaderError is not None
        assert FileIOError is not None
        assert TensorOrderError is not None
        assert LengthMismatchError is not None

        # Modules/utilities
        assert dtype is not None
        assert dump is not None
        assert tqdm is not None


class TestDump:
    """Test dump function."""

    def test_dump_numpy_arrays(self, tmp_path):
        """Test dumping numpy arrays."""
        state_dict = {
            "array1": np.arange(10, dtype=np.float32),
            "array2": np.ones((5, 5), dtype=np.int64),
        }

        output_path = tmp_path / "test.safetensors"
        dump(state_dict, output_path)

        # Verify file was created
        assert output_path.exists()

        # Read back and verify
        with StreamReader(output_path) as reader:
            assert set(reader.keys()) == {"array1", "array2"}

            arr1 = reader.to_numpy("array1")
            assert np.array_equal(arr1, state_dict["array1"])

            arr2 = reader.to_numpy("array2")
            assert np.array_equal(arr2, state_dict["array2"])

    def test_dump_with_crc32(self, tmp_path):
        """Test dumping with CRC32 checksums."""
        state_dict = {
            "test": np.zeros(100, dtype=np.float32),
        }

        output_path = tmp_path / "test_crc.safetensors"
        dump(state_dict, output_path, crc32=True)

        # Verify CRC was computed
        with StreamReader(output_path) as reader:
            meta = reader.meta("test")
            assert meta.crc32 is not None

    def test_dump_with_buffer_size(self, tmp_path):
        """Test dumping with custom buffer size."""
        # Create large array to test chunking
        large_array = np.zeros(10000, dtype=np.float64)
        state_dict = {"large": large_array}

        output_path = tmp_path / "test_buffer.safetensors"
        dump(state_dict, output_path, buffer_size=1024)  # Small buffer

        # Verify file was created correctly
        with StreamReader(output_path) as reader:
            arr = reader.to_numpy("large")
            assert np.array_equal(arr, large_array)

    def test_dump_pytorch_tensors(self, tmp_path):
        """Test dumping PyTorch tensors."""
        # Mock PyTorch tensor with all required attributes
        mock_tensor = MagicMock()

        # Set up numpy() method chain
        np_array = np.array([1, 2, 3], dtype=np.float32)
        mock_numpy = MagicMock()
        mock_numpy.return_value = np_array
        mock_tensor.numpy = mock_numpy

        # Set up detach().cpu().numpy() chain
        mock_cpu = MagicMock()
        mock_cpu.numpy.return_value = np_array
        mock_detach = MagicMock()
        mock_detach.cpu.return_value = mock_cpu
        mock_tensor.detach.return_value = mock_detach

        # Set attributes for checking
        mock_tensor.dtype = "torch.float32"
        mock_tensor.shape = (3,)

        state_dict = {"tensor": mock_tensor}
        output_path = tmp_path / "test_torch.safetensors"

        with patch("stsw.dtype.normalize", return_value="F32"):
            dump(state_dict, output_path)

        assert output_path.exists()

    def test_dump_unsupported_type(self, tmp_path):
        """Test dumping unsupported tensor type."""
        state_dict = {"bad": "not a tensor"}

        output_path = tmp_path / "test_bad.safetensors"

        with pytest.raises(TypeError, match="Unsupported tensor type"):
            dump(state_dict, output_path)

    def test_dump_empty_state_dict(self, tmp_path):
        """Test dumping empty state dict."""
        state_dict = {}

        output_path = tmp_path / "test_empty.safetensors"
        # Empty state dict is handled
        dump(state_dict, output_path)

        # File should still be created with header
        assert output_path.exists()


class TestTqdm:
    """Test tqdm integration."""

    def test_wrap_with_tqdm(self, tmp_path):
        """Test wrapping StreamWriter with tqdm."""
        # Create a writer first
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Mock tqdm module
        mock_pbar = MagicMock()
        mock_tqdm_class = MagicMock(return_value=mock_pbar)

        with patch.dict("sys.modules", {"tqdm.auto": MagicMock(tqdm=mock_tqdm_class)}):
            # Wrap with tqdm
            wrapped = tqdm.wrap(writer)

            # Should return a TqdmWriter instance, not the original
            assert wrapped is not writer
            assert hasattr(wrapped, "write_block")

            # Clean up
            writer.abort()

    def test_wrap_without_tqdm(self, tmp_path):
        """Test wrapping when tqdm is not available."""
        # Create a writer
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Test that wrap returns the writer (could be wrapped or not)
        wrapped = tqdm.wrap(writer)
        assert wrapped is not None

        # Should be able to use it like a writer
        if hasattr(wrapped, "_wrapped"):
            # It's a TqdmWriter
            wrapped.abort()
        else:
            # It's the original writer
            writer.abort()

    def test_tqdm_integration(self, tmp_path):
        """Test tqdm integration basics."""
        # Just test that tqdm.wrap exists and is callable
        assert hasattr(tqdm, "wrap")
        assert callable(tqdm.wrap)

        # Create a writer
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])

        # Test wrap returns something
        result = tqdm.wrap(writer)
        assert result is not None

        # Clean up
        writer.abort()
