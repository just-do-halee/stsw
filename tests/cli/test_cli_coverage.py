"""Additional tests for CLI module to improve coverage."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stsw._core.header import build_header
from stsw._core.meta import TensorMeta
from stsw.cli.__main__ import cmd_inspect, cmd_verify, setup_logging


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_verbose(self):
        """Test verbose logging setup."""
        setup_logging(verbose=True)
        # Just verify it doesn't error

    def test_setup_logging_quiet(self):
        """Test quiet logging setup."""
        setup_logging(verbose=False)
        # Just verify it doesn't error

    def test_setup_logging_with_rich(self):
        """Test logging setup with rich available."""
        mock_handler = MagicMock()
        mock_rich_handler = MagicMock(return_value=mock_handler)
        
        with patch.dict("sys.modules", {"rich.logging": MagicMock(RichHandler=mock_rich_handler)}):
            setup_logging(verbose=True)
            # Should use RichHandler
            mock_rich_handler.assert_called_once()


class TestCmdInspect:
    """Additional tests for inspect command."""

    def test_inspect_tensor_with_no_crc(self, tmp_path):
        """Test inspecting tensor without CRC."""
        test_file = tmp_path / "test.safetensors"
        
        # Create file with tensor without CRC
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        header = build_header([meta])
        
        with open(test_file, "wb") as f:
            f.write(header)
            f.write(np.zeros(10, dtype=np.float32).tobytes())
        
        args = argparse.Namespace(file=test_file)
        
        # Should not error
        result = cmd_inspect(args)
        assert result == 0

    def test_inspect_large_tensor(self, tmp_path):
        """Test inspecting large tensor."""
        test_file = tmp_path / "test.safetensors"
        
        # Create file with large tensor
        meta = TensorMeta("large", "F32", (1000, 1000), 0, 4000000)
        header = build_header([meta])
        
        with open(test_file, "wb") as f:
            f.write(header)
            f.write(b"\x00" * 4000000)
        
        args = argparse.Namespace(file=test_file)
        
        result = cmd_inspect(args)
        assert result == 0


class TestCmdVerify:
    """Additional tests for verify command."""

    def test_verify_no_tensors(self, tmp_path):
        """Test verifying file with no tensors."""
        test_file = tmp_path / "empty.safetensors"
        
        # Create empty file - this will fail because header must have content
        # So test error handling instead
        test_file.write_bytes(b"invalid")
        
        args = argparse.Namespace(file=test_file)
        
        result = cmd_verify(args)
        assert result == 1  # Should fail

    def test_verify_mixed_crc(self, tmp_path):
        """Test verifying file with some tensors having CRC."""
        from stsw._core.crc32 import compute_crc32
        
        test_file = tmp_path / "mixed.safetensors"
        
        # Create test data
        data1 = np.arange(10, dtype=np.float32).tobytes()
        data2 = np.ones(5, dtype=np.int64).tobytes()
        
        # One with CRC, one without
        meta1 = TensorMeta("tensor1", "F32", (10,), 0, len(data1), crc32=compute_crc32(data1))
        meta2 = TensorMeta("tensor2", "I64", (5,), 64, 64 + len(data2))  # No CRC
        
        header = build_header([meta1, meta2], align=64)
        
        with open(test_file, "wb") as f:
            f.write(header)
            f.write(data1)
            f.write(b"\x00" * (64 - len(data1)))  # Padding
            f.write(data2)
        
        args = argparse.Namespace(file=test_file)
        
        # Should succeed
        result = cmd_verify(args)
        assert result == 0


class TestCLIMisc:
    """Miscellaneous CLI tests."""

    def test_main_version_flag(self):
        """Test --version flag."""
        from stsw.cli.__main__ import main
        
        with patch("sys.argv", ["stsw", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with 0 for --version
            assert exc_info.value.code == 0

    def test_main_help_flag(self):
        """Test --help flag."""
        from stsw.cli.__main__ import main
        
        with patch("sys.argv", ["stsw", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with 0 for --help
            assert exc_info.value.code == 0