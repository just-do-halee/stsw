"""Final tests for CLI to improve coverage."""

import argparse
from unittest.mock import MagicMock, patch

import pytest

from stsw.cli.__main__ import cmd_convert, cmd_selftest, main


class TestCLIFinal:
    """Final CLI tests for coverage."""

    def test_convert_non_dict_state(self, tmp_path):
        """Test convert with non-dict state."""
        args = argparse.Namespace(
            input="model.pt",
            output="model.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock torch to return non-dict
        mock_torch = MagicMock()
        mock_torch.load.return_value = "not a dict"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("stsw.cli.__main__.logger") as mock_logger:
                result = cmd_convert(args)

        assert result == 1
        mock_logger.error.assert_called_with("Input file must contain a state dict")

    def test_convert_skip_non_tensor(self, tmp_path):
        """Test convert skipping non-tensor values."""
        args = argparse.Namespace(
            input="model.pt",
            output=tmp_path / "model.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock torch with mixed state dict
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.shape = (10,)
        mock_tensor.numel.return_value = 10
        mock_tensor.element_size.return_value = 4
        mock_tensor.dtype = mock_torch.float32
        mock_tensor.is_contiguous.return_value = True
        mock_tensor.detach().cpu().numpy().tobytes.return_value = b"x" * 40

        # State dict with tensor and non-tensor
        mock_torch.load.return_value = {
            "tensor": mock_tensor,
            "metadata": {"key": "value"},  # Non-tensor
        }
        mock_torch.float32 = "torch.float32"
        mock_torch.Tensor = type(mock_tensor)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("stsw.cli.__main__.normalize", return_value="F32"):
                with patch("stsw.cli.__main__.logger") as mock_logger:
                    result = cmd_convert(args)

        assert result == 0
        # Should warn about skipping non-tensor
        mock_logger.warning.assert_called_once()

    def test_convert_non_contiguous_tensor(self, tmp_path):
        """Test convert with non-contiguous tensor."""
        args = argparse.Namespace(
            input="model.pt",
            output=tmp_path / "model.safetensors",
            crc32=True,  # Test with CRC
            buffer_size=1,  # Small buffer to test chunking
        )

        # Mock torch with non-contiguous tensor
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.shape = (10,)
        mock_tensor.numel.return_value = 10
        mock_tensor.element_size.return_value = 4
        mock_tensor.dtype = mock_torch.float32
        mock_tensor.is_contiguous.return_value = False  # Not contiguous

        # Make contiguous() return a new tensor that's also a mock
        mock_contiguous = MagicMock()
        mock_contiguous.shape = (10,)
        mock_contiguous.numel.return_value = 10
        mock_contiguous.element_size.return_value = 4
        mock_contiguous.dtype = mock_torch.float32
        mock_contiguous.is_contiguous.return_value = True
        mock_contiguous.detach().cpu().numpy().tobytes.return_value = b"x" * 40
        mock_tensor.contiguous.return_value = mock_contiguous

        # Make torch.Tensor a proper type for isinstance check
        mock_torch.Tensor = type(mock_tensor)
        mock_torch.load.return_value = {"tensor": mock_tensor}
        mock_torch.float32 = "torch.float32"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("stsw.cli.__main__.normalize", return_value="F32"):
                result = cmd_convert(args)

        assert result == 0
        # Should have called contiguous()
        mock_tensor.contiguous.assert_called_once()

    def test_selftest_write_error(self, tmp_path):
        """Test selftest with write error."""
        args = argparse.Namespace()

        # Mock numpy
        mock_np = MagicMock()
        mock_np.random.rand.return_value.astype.return_value = MagicMock(
            dtype="float32",
            shape=(1000, 1000),
            nbytes=4000000,
            tobytes=lambda: b"x" * 4000000,
        )

        with patch.dict("sys.modules", {"numpy": mock_np}):
            with patch(
                "stsw.cli.__main__.StreamWriter.open",
                side_effect=Exception("Write error"),
            ):
                with patch("stsw.cli.__main__.logger") as mock_logger:
                    result = cmd_selftest(args)

        assert result == 1
        mock_logger.error.assert_called_once()

    def test_main_invalid_command(self):
        """Test main with invalid command."""
        with patch("sys.argv", ["stsw", "invalid"]):
            # argparse exits with SystemExit(2) for invalid command
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2

    def test_main_verbose_flag(self):
        """Test main with verbose flag."""
        with patch("sys.argv", ["stsw", "-v", "selftest"]):
            with patch("stsw.cli.__main__.cmd_selftest", return_value=0):
                with patch("stsw.cli.__main__.setup_logging") as mock_setup:
                    result = main()

        assert result == 0
        mock_setup.assert_called_once_with(True)
