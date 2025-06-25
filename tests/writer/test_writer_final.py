"""Final tests for writer to reach 98% coverage."""

from pathlib import Path

import pytest

from stsw._core.meta import TensorMeta
from stsw.writer.writer import StreamWriter


class TestWriterFinalCoverage:
    """Final tests for writer coverage."""

    def test_writer_str_method(self, tmp_path):
        """Test __str__ method of StreamWriter."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Test __str__
        str_repr = str(writer)
        assert isinstance(str_repr, str)
        
        writer.abort()

    def test_writer_with_existing_file(self, tmp_path):
        """Test opening writer with existing file."""
        test_file = tmp_path / "test.st"
        
        # Create file first
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(test_file, [meta])
        writer.write_block("test", b"x" * 40)
        writer.finalize_tensor("test")
        writer.close()
        
        # Open again (should overwrite)
        writer2 = StreamWriter.open(test_file, [meta])
        writer2.write_block("test", b"y" * 40)
        writer2.finalize_tensor("test")
        writer2.close()
        
        assert test_file.exists()

    def test_writer_stats_zero_elapsed(self, tmp_path):
        """Test stats when elapsed time is zero."""
        import time
        
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Mock start time to be current time (zero elapsed)
        writer._start_time = time.time()
        
        stats = writer.stats()
        # Should handle zero elapsed time gracefully
        assert stats.mb_per_s == 0.0
        # When mb_per_s is 0, eta_s is also 0
        assert stats.eta_s == 0
        
        writer.abort()

    def test_writer_finalize_with_padding(self, tmp_path):
        """Test finalizing tensor that needs padding."""
        # Create tensor with size that's not aligned
        meta = TensorMeta("test", "F32", (7,), 0, 28)  # 28 bytes, needs padding to 64
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        writer.write_block("test", b"x" * 28)
        writer.finalize_tensor("test")
        
        # Should have written padding
        assert writer._total_written > 28
        
        writer.close()

    def test_writer_context_manager_with_error(self, tmp_path):
        """Test writer context manager handles errors."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        
        try:
            with StreamWriter.open(tmp_path / "test.st", [meta]) as writer:
                # Write partial data
                writer.write_block("test", b"x" * 20)
                # Raise error before finalizing
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # File should not exist (aborted)
        assert not (tmp_path / "test.st").exists()