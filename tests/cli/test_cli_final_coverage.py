"""Final tests to achieve full coverage for CLI module."""

import argparse
from unittest.mock import MagicMock, patch

from stsw._core.header import build_header
from stsw._core.meta import TensorMeta
from stsw.cli.__main__ import cmd_convert, cmd_inspect, cmd_verify, main


class TestCLIFinalCoverage:
    """Final tests for complete CLI coverage."""

    def test_inspect_with_rich_and_metadata(self, tmp_path):
        """Test inspect command with rich available and metadata."""
        # Create test file with metadata
        test_file = tmp_path / "test.safetensors"
        meta = TensorMeta("tensor", "F32", (10, 10), 0, 400, crc32=12345)
        metadata = {"model": "test", "version": "1.0"}
        header = build_header([meta], metadata=metadata)

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 400)

        args = argparse.Namespace(file=test_file)

        # Mock rich components
        mock_console_instance = MagicMock()
        mock_console_class = MagicMock(return_value=mock_console_instance)
        mock_table_class = MagicMock()
        mock_table = MagicMock()
        mock_table_class.return_value = mock_table

        # Create mock modules
        mock_rich_console = MagicMock(Console=mock_console_class)
        mock_rich_table = MagicMock(Table=mock_table_class)

        # Patch the imports at the module level before the function runs
        with patch.dict(
            "sys.modules",
            {
                "rich": MagicMock(),
                "rich.console": mock_rich_console,
                "rich.table": mock_rich_table,
            },
        ):
            result = cmd_inspect(args)

        assert result == 0
        # Verify console methods were called
        mock_console_instance.print.assert_called()
        mock_table.add_column.assert_called()
        mock_table.add_row.assert_called()

    def test_inspect_plain_output_with_crc(self, tmp_path):
        """Test inspect plain text output with CRC values."""
        test_file = tmp_path / "test.safetensors"
        meta = TensorMeta("tensor_with_crc", "I64", (5,), 0, 40, crc32=67890)
        header = build_header([meta])

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 40)

        args = argparse.Namespace(file=test_file)

        # Force plain text output
        with patch.dict(
            "sys.modules", {"rich": None, "rich.console": None, "rich.table": None}
        ):
            # Capture output
            with patch("builtins.print") as mock_print:
                result = cmd_inspect(args)

        assert result == 0
        # Check that CRC was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("67890" in call for call in print_calls)

    def test_convert_with_non_contiguous_tensor(self, tmp_path):
        """Test convert with non-contiguous tensor."""
        input_file = tmp_path / "input.pt"
        input_file.write_bytes(b"dummy")

        args = argparse.Namespace(
            input=input_file,
            output=tmp_path / "output.safetensors",
            crc32=True,
            buffer_size=2,
        )

        # Mock torch
        mock_torch = MagicMock()

        # Create a more proper mock for torch.float32
        mock_float32 = MagicMock()
        mock_float32.__str__.return_value = "torch.float32"
        mock_torch.float32 = mock_float32

        mock_tensor = MagicMock()
        mock_tensor.__class__ = type("Tensor", (), {})
        mock_tensor.shape = (10,)
        mock_tensor.dtype = mock_float32
        mock_tensor.numel.return_value = 10
        mock_tensor.element_size.return_value = 4
        mock_tensor.is_contiguous.return_value = False  # Non-contiguous

        # Mock contiguous() to return a contiguous version
        mock_contiguous = MagicMock()
        mock_contiguous.detach.return_value.cpu.return_value.numpy.return_value.tobytes.return_value = (
            b"x" * 40
        )
        mock_tensor.contiguous.return_value = mock_contiguous

        state_dict = {"tensor": mock_tensor}
        mock_torch.load.return_value = state_dict
        mock_torch.Tensor = mock_tensor.__class__

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("stsw.cli.__main__.normalize", return_value="F32"):
                result = cmd_convert(args)

        assert result == 0
        # Verify contiguous() was called
        mock_tensor.contiguous.assert_called()

    def test_main_verbose_flag(self):
        """Test main with --verbose flag."""
        with patch("sys.argv", ["stsw", "--verbose", "selftest"]):
            with patch("stsw.cli.__main__.cmd_selftest", return_value=0) as mock_cmd:
                with patch("stsw.cli.__main__.setup_logging") as mock_setup:
                    result = main()

        assert result == 0
        mock_setup.assert_called_with(True)  # verbose=True
        mock_cmd.assert_called()

    def test_verify_tensor_without_crc(self, tmp_path):
        """Test verify with tensor that has no CRC."""
        from stsw._core.crc32 import compute_crc32

        test_file = tmp_path / "test.safetensors"

        # Create tensors - one with CRC, one without
        # Compute the correct CRC for the first tensor's data
        data1 = b"\x00" * 20
        correct_crc = compute_crc32(data1)

        meta1 = TensorMeta("has_crc", "F32", (5,), 0, 20, crc32=correct_crc)
        meta2 = TensorMeta("no_crc", "F32", (5,), 32, 52)  # No CRC
        header = build_header([meta1, meta2], align=32)

        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 52)

        args = argparse.Namespace(file=test_file)

        # Capture output
        with patch("builtins.print") as mock_print:
            result = cmd_verify(args)

        # Should succeed with 0 errors
        assert result == 0

        # Check output mentions no CRC for the second tensor
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any(
            "no_crc" in call and "No CRC32 stored" in call for call in print_calls
        )
