"""Tests for CLI module."""

import argparse
import sys
from unittest.mock import MagicMock, patch

import pytest

from stsw._core.meta import TensorMeta
from stsw.cli.__main__ import (
    cmd_convert,
    cmd_inspect,
    cmd_selftest,
    cmd_verify,
    main,
)


class TestCmdInspect:
    """Test inspect command."""

    @patch("stsw.cli.__main__.StreamReader")
    def test_inspect_with_rich(self, mock_reader):
        """Test inspect command with rich output."""
        # Mock reader
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = iter(["tensor1", "tensor2"])
        mock_instance.meta.side_effect = [
            TensorMeta("tensor1", "F32", (10, 20), 0, 800),
            TensorMeta("tensor2", "I64", (5, 5), 832, 1032, crc32=12345),
        ]
        mock_instance.version = "1.0"
        mock_instance.metadata = {"key": "value"}
        mock_reader.return_value = mock_instance

        args = argparse.Namespace(file="test.safetensors")

        # Test with rich available
        # Patch the imports inside the function
        mock_console = MagicMock()
        mock_table = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "rich.console": MagicMock(Console=mock_console),
                "rich.table": MagicMock(Table=mock_table),
            },
        ):
            result = cmd_inspect(args)

        assert result == 0
        mock_reader.assert_called_once_with("test.safetensors", mmap=True)

    @patch("stsw.cli.__main__.StreamReader")
    def test_inspect_without_rich(self, mock_reader):
        """Test inspect command with plain output."""
        # Mock reader
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = iter(["tensor1", "tensor2"])
        mock_instance.meta.side_effect = [
            TensorMeta("tensor1", "F32", (10, 20), 0, 800),
            TensorMeta("tensor2", "I64", (5, 5), 832, 1032, crc32=12345),
        ]
        mock_instance.version = "1.0"
        mock_instance.metadata = None
        mock_reader.return_value = mock_instance

        args = argparse.Namespace(file="test.safetensors")

        # Test without rich by making import fail
        with patch.dict(sys.modules, {"rich.console": None, "rich.table": None}):
            result = cmd_inspect(args)

        assert result == 0

    @patch("stsw.cli.__main__.StreamReader")
    def test_inspect_file_not_found(self, mock_reader):
        """Test inspect command with non-existent file."""
        mock_reader.side_effect = FileNotFoundError("File not found")

        args = argparse.Namespace(file="nonexistent.safetensors")

        with patch("stsw.cli.__main__.logger") as mock_logger:
            result = cmd_inspect(args)

        assert result == 1
        mock_logger.error.assert_called_once()

    @patch("stsw.cli.__main__.StreamReader")
    def test_inspect_header_error(self, mock_reader):
        """Test inspect command with header error."""
        from stsw._core.header import HeaderError

        mock_reader.side_effect = HeaderError("Invalid header")

        args = argparse.Namespace(file="corrupt.safetensors")

        with patch("stsw.cli.__main__.logger") as mock_logger:
            result = cmd_inspect(args)

        assert result == 1
        mock_logger.error.assert_called_with("Failed to inspect file: Invalid header")

    @patch("stsw.cli.__main__.StreamReader")
    def test_inspect_general_error(self, mock_reader):
        """Test inspect command with general error."""
        mock_reader.side_effect = Exception("Unknown error")

        args = argparse.Namespace(file="test.safetensors")

        with patch("stsw.cli.__main__.logger") as mock_logger:
            result = cmd_inspect(args)

        assert result == 1
        mock_logger.error.assert_called_with("Failed to inspect file: Unknown error")


class TestCmdVerify:
    """Test verify command."""

    @patch("stsw.cli.__main__.StreamReader")
    def test_verify_success(self, mock_reader):
        """Test successful verification."""
        # Mock reader
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.keys.return_value = ["tensor1", "tensor2"]
        mock_instance.__len__.return_value = 2
        mock_instance.meta.side_effect = [
            TensorMeta("tensor1", "F32", (10, 20), 0, 800, crc32=12345),
            TensorMeta("tensor2", "I64", (5, 5), 832, 1032, crc32=67890),
        ]
        mock_instance.get_slice.return_value = b"dummy_data"
        mock_reader.return_value = mock_instance

        args = argparse.Namespace(file="test.safetensors")

        result = cmd_verify(args)

        assert result == 0
        assert mock_instance.get_slice.call_count == 2

    @patch("stsw.cli.__main__.StreamReader")
    def test_verify_no_crc(self, mock_reader):
        """Test verification with no CRC checksums."""
        # Mock reader
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.__iter__.return_value = iter(["tensor1"])
        mock_instance.__len__.return_value = 1
        mock_instance.meta.return_value = TensorMeta("tensor1", "F32", (10, 20), 0, 800)
        mock_reader.return_value = mock_instance

        args = argparse.Namespace(file="test.safetensors")

        result = cmd_verify(args)

        assert result == 0

    @patch("stsw.cli.__main__.StreamReader")
    def test_verify_crc_mismatch(self, mock_reader):
        """Test verification with CRC mismatch."""
        # Mock reader
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None
        mock_instance.keys.return_value = ["tensor1"]
        mock_instance.__len__.return_value = 1

        # Create meta that will raise CRC error
        meta = TensorMeta("tensor1", "F32", (10, 20), 0, 800, crc32=12345)
        mock_instance.meta.return_value = meta
        mock_instance.get_slice.side_effect = ValueError("CRC32 mismatch")

        mock_reader.return_value = mock_instance

        args = argparse.Namespace(file="test.safetensors")

        result = cmd_verify(args)

        assert result == 1

    @patch("stsw.cli.__main__.StreamReader")
    def test_verify_file_error(self, mock_reader):
        """Test verification with file error."""
        mock_reader.side_effect = Exception("File error")

        args = argparse.Namespace(file="test.safetensors")

        with patch("stsw.cli.__main__.logger") as mock_logger:
            result = cmd_verify(args)

        assert result == 1
        mock_logger.error.assert_called_once()


class TestCmdConvert:
    """Test convert command."""

    def test_convert_success(self):
        """Test successful conversion."""
        args = argparse.Namespace(
            input="model.pt",
            output="model.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock torch
        mock_torch = MagicMock()
        mock_tensor = MagicMock()
        mock_tensor.shape = (10, 20)
        mock_tensor.numel.return_value = 200
        mock_tensor.element_size.return_value = 4
        mock_tensor.dtype = mock_torch.float32
        mock_tensor.is_contiguous.return_value = True
        mock_tensor.detach().cpu().numpy().tobytes.return_value = b"x" * 800
        mock_torch.load.return_value = {"weight": mock_tensor}
        mock_torch.float32 = "torch.float32"
        mock_torch.Tensor = type(mock_tensor)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            with patch("stsw.cli.__main__.normalize", return_value="F32"):
                with patch("stsw.cli.__main__.StreamWriter") as mock_writer:
                    mock_writer_instance = MagicMock()
                    mock_writer.open.return_value.__enter__.return_value = (
                        mock_writer_instance
                    )
                    result = cmd_convert(args)

        assert result == 0

    def test_convert_no_torch(self):
        """Test conversion without torch installed."""
        args = argparse.Namespace(
            input="model.pt",
            output="model.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Make torch import fail
        with patch.dict(sys.modules, {"torch": None}):
            with patch("stsw.cli.__main__.logger") as mock_logger:
                result = cmd_convert(args)

        assert result == 1
        mock_logger.error.assert_called_once()

    def test_convert_error(self):
        """Test conversion with error."""
        args = argparse.Namespace(
            input="model.pt",
            output="model.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.load.side_effect = Exception("Load error")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # The function doesn't catch torch.load errors, so it will raise
            with pytest.raises(Exception, match="Load error"):
                cmd_convert(args)


class TestCmdSelftest:
    """Test selftest command."""

    def test_selftest_success(self, tmp_path):
        """Test successful selftest."""
        args = argparse.Namespace()

        result = cmd_selftest(args)
        assert result == 0

    @patch("stsw.cli.__main__.StreamWriter.open")
    def test_selftest_write_error(self, mock_writer_open):
        """Test selftest with write error."""
        args = argparse.Namespace()

        mock_writer_open.side_effect = Exception("Write error")

        with patch("stsw.cli.__main__.logger") as mock_logger:
            result = cmd_selftest(args)

        assert result == 1
        mock_logger.error.assert_called_once()


class TestMain:
    """Test main entry point."""

    def test_main_inspect(self):
        """Test main with inspect command."""
        with patch("sys.argv", ["stsw", "inspect", "test.safetensors"]):
            with patch("stsw.cli.__main__.cmd_inspect") as mock_inspect:
                mock_inspect.return_value = 0
                result = main()

        assert result == 0
        mock_inspect.assert_called_once()

    def test_main_verify(self):
        """Test main with verify command."""
        with patch("sys.argv", ["stsw", "verify", "test.safetensors"]):
            with patch("stsw.cli.__main__.cmd_verify") as mock_verify:
                mock_verify.return_value = 0
                result = main()

        assert result == 0
        mock_verify.assert_called_once()

    def test_main_convert(self):
        """Test main with convert command."""
        with patch("sys.argv", ["stsw", "convert", "in.pt", "out.st"]):
            with patch("stsw.cli.__main__.cmd_convert") as mock_convert:
                mock_convert.return_value = 0
                result = main()

        assert result == 0
        mock_convert.assert_called_once()

    def test_main_selftest(self):
        """Test main with selftest command."""
        with patch("sys.argv", ["stsw", "selftest"]):
            with patch("stsw.cli.__main__.cmd_selftest") as mock_selftest:
                mock_selftest.return_value = 0
                result = main()

        assert result == 0
        mock_selftest.assert_called_once()

    def test_main_no_args(self):
        """Test main with no arguments."""
        with patch("sys.argv", ["stsw"]):
            result = main()
            assert result == 1

    def test_main_keyboard_interrupt(self):
        """Test main with keyboard interrupt."""
        with patch("sys.argv", ["stsw", "selftest"]):
            with patch("stsw.cli.__main__.cmd_selftest") as mock_selftest:
                mock_selftest.side_effect = KeyboardInterrupt()
                result = main()

        assert result == 130  # Standard exit code for Ctrl+C
