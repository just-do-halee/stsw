"""Tests for PyTorch-specific paths in __init__.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stsw import dump, tqdm
from stsw._core.meta import TensorMeta
from stsw.writer.writer import StreamWriter


class TestDumpWithPyTorch:
    """Test dump function with PyTorch tensors."""

    def test_dump_with_torch_tensor(self, tmp_path):
        """Test dump with PyTorch tensor."""
        # Mock torch module
        mock_torch = MagicMock()
        mock_tensor = MagicMock()

        # Set up tensor properties
        mock_tensor.shape = (10, 5)
        mock_tensor.dtype = MagicMock()
        mock_tensor.numel.return_value = 50
        mock_tensor.element_size.return_value = 4

        # Create numpy array
        numpy_array = np.random.rand(10, 5).astype(np.float32)
        numpy_bytes = numpy_array.tobytes()

        # Mock tensor to look like PyTorch tensor (has numpy method)
        mock_tensor.numpy = MagicMock()

        # Create mock numpy result that has tobytes method
        mock_numpy_result = MagicMock()
        mock_numpy_result.tobytes.return_value = numpy_bytes
        mock_numpy_result.shape = numpy_array.shape
        mock_numpy_result.nbytes = len(numpy_bytes)

        # Mock detach().cpu().numpy() chain
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = (
            mock_numpy_result
        )

        state_dict = {"tensor": mock_tensor}
        output_path = tmp_path / "test_torch.safetensors"

        # Mock dtype.normalize
        with patch("stsw._core.dtype.normalize", return_value="F32"):
            dump(state_dict, output_path)

        assert output_path.exists()

        # Verify tensor methods were called
        mock_tensor.detach.assert_called()
        mock_tensor.detach.return_value.cpu.assert_called()
        mock_tensor.detach.return_value.cpu.return_value.numpy.assert_called()

    def test_dump_with_unsupported_type(self, tmp_path):
        """Test dump with unsupported tensor type."""
        # Create an object that's not a tensor
        unsupported_obj = {"not": "a tensor"}
        state_dict = {"bad_tensor": unsupported_obj}

        output_path = tmp_path / "test_bad.safetensors"

        with pytest.raises(TypeError, match="Unsupported tensor type for 'bad_tensor'"):
            dump(state_dict, output_path)

    def test_dump_torch_tensor_large_chunks(self, tmp_path):
        """Test dump with PyTorch tensor requiring multiple chunks."""
        # Mock torch tensor
        mock_tensor = MagicMock()

        # Large tensor
        mock_tensor.shape = (1000, 1000)
        mock_tensor.dtype = MagicMock()
        mock_tensor.numel.return_value = 1_000_000
        mock_tensor.element_size.return_value = 4

        # Mock tensor to look like PyTorch tensor
        mock_tensor.numpy = MagicMock()

        # Create mock numpy result
        mock_numpy_result = MagicMock()
        mock_numpy_result.tobytes.return_value = b"x" * (4 * 1_000_000)  # 4MB
        mock_numpy_result.shape = (1000, 1000)
        mock_numpy_result.nbytes = 4 * 1_000_000
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = (
            mock_numpy_result
        )

        state_dict = {"large_tensor": mock_tensor}
        output_path = tmp_path / "test_large_torch.safetensors"

        with patch("stsw._core.dtype.normalize", return_value="F32"):
            # Use small buffer to force chunking
            dump(state_dict, output_path, buffer_size=1024 * 1024)  # 1MB chunks

        assert output_path.exists()


class TestTqdmWithoutPackage:
    """Test tqdm integration when package is not available."""

    def test_tqdm_wrap_without_tqdm(self, tmp_path):
        """Test tqdm.wrap when tqdm is not installed."""
        # Mock ImportError for tqdm
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'tqdm'")
        ):
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            # Should return original writer
            wrapped = tqdm.wrap(writer)
            assert wrapped is writer

            writer.abort()

    def test_tqdm_wrap_with_mock_tqdm(self, tmp_path):
        """Test tqdm.wrap with mocked tqdm to test TqdmWriter methods."""
        # Create mock tqdm
        mock_tqdm_module = MagicMock()
        mock_tqdm_bar = MagicMock()
        mock_tqdm_module.auto.tqdm = mock_tqdm_bar

        # Mock progress bar instance
        mock_pbar = MagicMock()
        mock_tqdm_bar.return_value = mock_pbar

        with patch.dict(
            "sys.modules",
            {"tqdm": mock_tqdm_module, "tqdm.auto": mock_tqdm_module.auto},
        ):
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            wrapped = tqdm.wrap(writer)

            # Test that it's wrapped
            assert wrapped is not writer
            assert hasattr(wrapped, "_pbar")

            # Test that getattr works
            assert wrapped._total_size == writer._total_size

            # Test abort instead to avoid needing to write data
            wrapped.abort()
            mock_pbar.close.assert_called()

    def test_tqdm_wrap_abort_with_tqdm(self, tmp_path):
        """Test TqdmWriter abort method."""
        # Create mock tqdm
        mock_tqdm_module = MagicMock()
        mock_tqdm_bar = MagicMock()
        mock_tqdm_module.auto.tqdm = mock_tqdm_bar

        # Mock progress bar instance
        mock_pbar = MagicMock()
        mock_tqdm_bar.return_value = mock_pbar

        with patch.dict(
            "sys.modules",
            {"tqdm": mock_tqdm_module, "tqdm.auto": mock_tqdm_module.auto},
        ):
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            wrapped = tqdm.wrap(writer)

            # Test abort
            wrapped.abort()
            mock_pbar.close.assert_called()

            # File should not exist
            assert not (tmp_path / "test.st").exists()


class TestDumpMixedTensors:
    """Test dump with mixed tensor types."""

    def test_dump_mixed_torch_numpy(self, tmp_path):
        """Test dump with both PyTorch and numpy tensors."""
        # Mock torch tensor
        mock_torch_tensor = MagicMock()
        mock_torch_tensor.shape = (5,)
        mock_torch_tensor.dtype = MagicMock()
        mock_torch_tensor.numel.return_value = 5
        mock_torch_tensor.element_size.return_value = 4

        mock_numpy_array = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        mock_torch_tensor.numpy = MagicMock()  # Has numpy method

        # Create mock numpy result
        mock_numpy_result = MagicMock()
        mock_numpy_result.tobytes.return_value = mock_numpy_array.tobytes()
        mock_numpy_result.shape = mock_numpy_array.shape
        mock_numpy_result.nbytes = mock_numpy_array.nbytes
        mock_torch_tensor.detach.return_value.cpu.return_value.numpy.return_value = (
            mock_numpy_result
        )

        # Real numpy tensor
        numpy_tensor = np.array([6, 7, 8, 9, 10], dtype=np.int32)

        state_dict = {"torch_tensor": mock_torch_tensor, "numpy_tensor": numpy_tensor}

        output_path = tmp_path / "test_mixed.safetensors"

        with patch(
            "stsw._core.dtype.normalize",
            side_effect=lambda x: "F32" if x == mock_torch_tensor.dtype else "I32",
        ):
            dump(state_dict, output_path)

        assert output_path.exists()
