"""Final tests to achieve full coverage for __init__.py."""

from unittest.mock import MagicMock, patch

from stsw import tqdm
from stsw._core.meta import TensorMeta
from stsw.writer.writer import StreamWriter


class TestTqdmFullCoverage:
    """Test tqdm integration with full coverage."""

    def test_tqdm_wrap_full_workflow(self, tmp_path):
        """Test complete tqdm workflow with write_block and close."""
        # Import tqdm to see if it's available
        try:
            __import__("tqdm")  # Check if tqdm is available

            # If tqdm is installed, use it directly
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            wrapped = tqdm.wrap(writer)

            # Write and finalize
            wrapped.write_block("test", b"x" * 40)
            wrapped.finalize_tensor("test")
            wrapped.close()

            assert (tmp_path / "test.st").exists()
        except ImportError:
            # Create mock tqdm that works properly
            mock_tqdm_module = MagicMock()
            mock_pbar = MagicMock()

            # Make tqdm return the mock progress bar
            mock_tqdm_module.auto.tqdm = MagicMock(return_value=mock_pbar)

            with patch.dict(
                "sys.modules",
                {"tqdm": mock_tqdm_module, "tqdm.auto": mock_tqdm_module.auto},
            ):
                # Create writer
                meta = TensorMeta("test", "F32", (10,), 0, 40)
                writer = StreamWriter.open(tmp_path / "test.st", [meta])

                # Wrap with tqdm
                wrapped = tqdm.wrap(writer)

                # Verify wrapped
                assert wrapped is not writer
                assert hasattr(wrapped, "_pbar")
                assert hasattr(wrapped, "_wrapped")

                # Test write_block updates progress bar
                wrapped.write_block("test", b"x" * 40)
                mock_pbar.update.assert_called_with(40)

                # Test finalize_tensor (through __getattr__)
                wrapped.finalize_tensor("test")

                # Test close
                wrapped.close()
                mock_pbar.close.assert_called()

                # Verify file was created
                assert (tmp_path / "test.st").exists()

    def test_tqdm_wrap_getattr_coverage(self, tmp_path):
        """Test __getattr__ method of TqdmWriter."""
        # Create mock tqdm
        mock_tqdm_module = MagicMock()
        mock_pbar = MagicMock()
        mock_tqdm_module.auto.tqdm = MagicMock(return_value=mock_pbar)

        with patch.dict(
            "sys.modules",
            {"tqdm": mock_tqdm_module, "tqdm.auto": mock_tqdm_module.auto},
        ):
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            wrapped = tqdm.wrap(writer)

            # Access various attributes through __getattr__
            assert wrapped.path == writer.path
            assert wrapped.tensors == writer.tensors
            assert wrapped._total_size == writer._total_size
            assert wrapped.enable_crc32 == writer.enable_crc32

            # Test method access
            stats = wrapped.stats()
            assert stats is not None

            wrapped.abort()
            mock_pbar.close.assert_called()

    def test_tqdm_with_actual_tqdm_if_available(self, tmp_path):
        """Test with actual tqdm if it's installed."""
        try:
            __import__("tqdm")  # Check if tqdm is available

            # If tqdm is actually installed, test with it
            meta = TensorMeta("test", "F32", (10,), 0, 40)
            writer = StreamWriter.open(tmp_path / "test.st", [meta])

            from stsw import tqdm

            wrapped = tqdm.wrap(writer)

            # Should get a wrapped writer
            assert hasattr(wrapped, "_pbar") or wrapped is writer

            writer.abort()
        except ImportError:
            # If tqdm is not installed, that's fine
            pass
