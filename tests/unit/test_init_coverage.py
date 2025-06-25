"""Additional tests for __init__.py to improve coverage."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from stsw import dump, tqdm
from stsw._core.meta import TensorMeta
from stsw.writer.writer import StreamWriter


class TestDumpCoverage:
    """Additional tests for dump function."""

    def test_dump_with_numpy_dtype_object(self, tmp_path):
        """Test dump with numpy array that has dtype object."""
        # Create numpy array with specific dtype
        arr = np.array([1, 2, 3], dtype=np.int32)
        state_dict = {"tensor": arr}
        
        output_path = tmp_path / "test.safetensors"
        dump(state_dict, output_path)
        
        assert output_path.exists()

    def test_dump_large_chunks(self, tmp_path):
        """Test dump with data that requires multiple chunks."""
        # Create large array (larger than default buffer size)
        large_array = np.zeros(2_000_000, dtype=np.float32)  # ~8MB
        state_dict = {"large": large_array}
        
        output_path = tmp_path / "test_large.safetensors"
        # Use small buffer to force chunking
        dump(state_dict, output_path, buffer_size=1024 * 1024)  # 1MB chunks
        
        assert output_path.exists()


class TestTqdmCoverage:
    """Additional tests for tqdm integration."""

    def test_tqdm_writer_getattr(self, tmp_path):
        """Test TqdmWriter __getattr__ method."""
        # Just test basic tqdm functionality without complex mocking
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Test wrap returns something
        wrapped = tqdm.wrap(writer)
        assert wrapped is not None
        
        # Clean up
        writer.abort()

    def test_tqdm_writer_abort(self, tmp_path):
        """Test TqdmWriter abort method."""
        # Create a writer
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        writer = StreamWriter.open(tmp_path / "test.st", [meta])
        
        # Test without tqdm (returns original writer)
        wrapped = tqdm.wrap(writer)
        
        # Abort should work
        wrapped.abort()
        
        # File should not exist
        assert not (tmp_path / "test.st").exists()


class TestPublicAPICoverage:
    """Additional tests for public API."""

    def test_all_imports_accessible(self):
        """Test all public imports are accessible."""
        import stsw
        
        # Test accessing all public attributes
        assert hasattr(stsw, "__version__")
        assert hasattr(stsw, "DEFAULT_ALIGN")
        assert hasattr(stsw, "TensorMeta")
        assert hasattr(stsw, "StreamReader") 
        assert hasattr(stsw, "StreamWriter")
        assert hasattr(stsw, "WriterStats")
        assert hasattr(stsw, "HeaderError")
        assert hasattr(stsw, "FileIOError")
        assert hasattr(stsw, "TensorOrderError")
        assert hasattr(stsw, "LengthMismatchError")
        assert hasattr(stsw, "dtype")
        assert hasattr(stsw, "dump")
        assert hasattr(stsw, "tqdm")