"""Tests for CLI error handling paths."""

import argparse
import builtins
from unittest.mock import MagicMock, patch

from stsw.cli.__main__ import (
    cmd_convert,
    cmd_inspect,
    cmd_selftest,
    cmd_verify,
    main,
    setup_logging,
)


class TestCLIErrorPaths:
    """Test error handling in CLI commands."""

    def test_inspect_file_not_found(self, tmp_path):
        """Test inspect with non-existent file."""
        args = argparse.Namespace(file=tmp_path / "nonexistent.safetensors")

        result = cmd_inspect(args)
        assert result == 1

    def test_inspect_invalid_file(self, tmp_path):
        """Test inspect with invalid safetensors file."""
        test_file = tmp_path / "invalid.safetensors"
        test_file.write_bytes(b"not a valid safetensors file")

        args = argparse.Namespace(file=test_file)

        result = cmd_inspect(args)
        assert result == 1

    def test_inspect_corrupted_header(self, tmp_path):
        """Test inspect with corrupted header."""
        test_file = tmp_path / "corrupted.safetensors"
        # Write invalid header length
        test_file.write_bytes(b"\xFF" * 8 + b"corrupted")

        args = argparse.Namespace(file=test_file)

        result = cmd_inspect(args)
        assert result == 1

    def test_verify_file_not_found(self, tmp_path):
        """Test verify with non-existent file."""
        args = argparse.Namespace(file=tmp_path / "nonexistent.safetensors")

        result = cmd_verify(args)
        assert result == 1

    def test_verify_invalid_file(self, tmp_path):
        """Test verify with invalid file."""
        test_file = tmp_path / "invalid.safetensors"
        test_file.write_bytes(b"invalid content")

        args = argparse.Namespace(file=test_file)

        result = cmd_verify(args)
        assert result == 1

    def test_verify_crc_mismatch(self, tmp_path):
        """Test verify with CRC mismatch."""
        from stsw._core.header import build_header
        from stsw._core.meta import TensorMeta

        # Create file with wrong CRC
        test_file = tmp_path / "wrong_crc.safetensors"
        data = b"test data"
        meta = TensorMeta("tensor", "F32", (2,), 0, len(data), crc32=12345)  # Wrong CRC
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data)

        args = argparse.Namespace(file=test_file)

        # Mock StreamReader to raise ValueError on get_slice
        with patch("stsw.cli.__main__.StreamReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader_class.return_value.__enter__.return_value = mock_reader
            mock_reader.__len__.return_value = 1
            mock_reader.keys.return_value = ["tensor"]
            mock_reader.meta.return_value = meta
            mock_reader.get_slice.side_effect = ValueError("CRC mismatch")

            result = cmd_verify(args)
            assert result == 1

    def test_convert_no_pytorch(self, tmp_path):
        """Test convert when PyTorch is not installed."""
        args = argparse.Namespace(
            input=tmp_path / "input.pt",
            output=tmp_path / "output.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock import error for torch
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'torch'")
        ):
            result = cmd_convert(args)
            assert result == 1

    def test_convert_not_state_dict(self, tmp_path):
        """Test convert with non-dict checkpoint."""
        # Create a dummy input file
        input_file = tmp_path / "input.pt"
        input_file.write_bytes(b"dummy")

        args = argparse.Namespace(
            input=input_file,
            output=tmp_path / "output.safetensors",
            crc32=False,
            buffer_size=8,
        )

        # Mock torch
        mock_torch = MagicMock()
        mock_torch.load.return_value = [
            "not",
            "a",
            "dict",
        ]  # Return list instead of dict

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = cmd_convert(args)
            assert result == 1

    def test_convert_non_tensor_in_dict(self, tmp_path):
        """Test convert with non-tensor values in state dict."""
        input_file = tmp_path / "input.pt"
        input_file.write_bytes(b"dummy")

        args = argparse.Namespace(
            input=input_file,
            output=tmp_path / "output.safetensors",
            crc32=True,
            buffer_size=4,
        )

        # Mock torch module
        mock_torch = MagicMock()
        
        # Create a simple Tensor type
        class TensorType:
            pass
        
        mock_torch.Tensor = TensorType
        
        # Create mock tensors that are instances of TensorType
        mock_tensor1 = MagicMock()
        mock_tensor1.__class__ = TensorType
        mock_tensor1.shape = (10,)
        mock_tensor1.numel.return_value = 10
        mock_tensor1.element_size.return_value = 4
        mock_tensor1.is_contiguous.return_value = True
        mock_tensor1.detach.return_value.cpu.return_value.numpy.return_value.tobytes.return_value = (
            b"x" * 40
        )
        mock_tensor1.dtype = "mock_dtype"
        
        mock_tensor2 = MagicMock()
        mock_tensor2.__class__ = TensorType
        mock_tensor2.shape = (5,)
        mock_tensor2.numel.return_value = 5
        mock_tensor2.element_size.return_value = 4
        mock_tensor2.is_contiguous.return_value = True
        mock_tensor2.detach.return_value.cpu.return_value.numpy.return_value.tobytes.return_value = (
            b"x" * 20
        )
        mock_tensor2.dtype = "mock_dtype"

        # State dict with mixed types - one string will be skipped
        state_dict = {
            "tensor1": mock_tensor1,
            "not_tensor": "string value",  # This should be skipped
            "tensor2": mock_tensor2,
        }
        mock_torch.load.return_value = state_dict

        # Mock normalize to return F32 for any dtype
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("stsw.cli.__main__.normalize", return_value="F32"):
                with patch("stsw.cli.__main__.StreamWriter") as mock_writer_class:
                    # Mock the writer
                    mock_writer = MagicMock()
                    mock_writer_class.open.return_value.__enter__.return_value = mock_writer
                    mock_writer_class.open.return_value.__exit__.return_value = None
                    
                    # Mock logger to check warning
                    with patch("stsw.cli.__main__.logger") as mock_logger:
                        result = cmd_convert(args)
                    
                    assert result == 0
                    # Check that warning was logged about skipping non-tensor
                    mock_logger.warning.assert_called_once()
                    warning_msg = str(mock_logger.warning.call_args)
                    assert "not_tensor" in warning_msg
                    assert "str" in warning_msg

    def test_selftest_failure(self, tmp_path):
        """Test selftest with failure."""
        args = argparse.Namespace()

        # Mock numpy to raise error
        with patch("numpy.random.rand", side_effect=Exception("Test failure")):
            result = cmd_selftest(args)
            assert result == 1

    def test_selftest_verification_failure(self, tmp_path):
        """Test selftest with verification failure."""
        args = argparse.Namespace()

        # Mock array_equal to return False
        with patch("numpy.array_equal", return_value=False):
            result = cmd_selftest(args)
            assert result == 1

    def test_main_no_command(self):
        """Test main function with no command."""
        with patch("sys.argv", ["stsw"]):
            result = main()
            assert result == 1

    def test_main_unknown_command(self):
        """Test main function with unknown command."""
        # Capture output to avoid polluting test output
        with patch("sys.argv", ["stsw", "unknown"]):
            with patch("sys.stderr", MagicMock()):
                try:
                    result = main()
                    assert False, "Should have raised SystemExit"
                except SystemExit as e:
                    assert e.code == 2

    def test_setup_logging_no_rich(self):
        """Test logging setup when rich is not available."""
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'rich'")
        ):
            # Should fall back to standard handler
            setup_logging(verbose=True)
            setup_logging(verbose=False)

    def test_inspect_with_metadata(self, tmp_path):
        """Test inspect with file containing metadata."""
        from stsw._core.header import build_header
        from stsw._core.meta import TensorMeta

        test_file = tmp_path / "with_metadata.safetensors"
        meta = TensorMeta("tensor", "F32", (5,), 0, 20)
        metadata = {"key": "value", "version": "1.0"}
        header = build_header([meta], metadata=metadata)

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 20)

        args = argparse.Namespace(file=test_file)

        # Test with rich not available
        with patch.dict("sys.modules", {"rich": None, "rich.console": None, "rich.table": None}):
            result = cmd_inspect(args)
            assert result == 0

    def test_verify_reader_exception(self, tmp_path):
        """Test verify with reader raising exception."""
        test_file = tmp_path / "test.safetensors"
        test_file.write_bytes(b"dummy")

        args = argparse.Namespace(file=test_file)

        # Mock StreamReader to raise exception
        with patch(
            "stsw.cli.__main__.StreamReader", side_effect=Exception("Reader error")
        ):
            result = cmd_verify(args)
            assert result == 1
