"""Tests for memory-mapped file wrapper."""

import mmap
import platform
import warnings
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from stsw.io.fileio import FileIOError
from stsw.io.mmapwrap import MMapWrapper


class TestMMapWrapper:
    """Test MMapWrapper class."""

    def test_basic_mmap(self, tmp_path):
        """Test basic memory mapping."""
        test_file = tmp_path / "test.bin"
        test_data = b"Hello, World! This is a test file."
        test_file.write_bytes(test_data)
        
        with MMapWrapper(test_file) as mmap_obj:
            assert len(mmap_obj) == len(test_data)
            assert mmap_obj[0] == test_data[0]
            assert mmap_obj[:5] == memoryview(test_data[:5])

    def test_mmap_with_offset(self, tmp_path):
        """Test memory mapping with offset."""
        test_file = tmp_path / "test.bin"
        test_data = b"0123456789" * 10
        test_file.write_bytes(test_data)
        
        with MMapWrapper(test_file, offset=10, length=20) as mmap_obj:
            assert len(mmap_obj) == 20
            # Should read from offset 10
            assert bytes(mmap_obj[:10]) == b"0123456789"

    def test_mmap_slice_operations(self, tmp_path):
        """Test slice operations on mmap."""
        test_file = tmp_path / "test.bin"
        test_data = bytes(range(256))
        test_file.write_bytes(test_data)
        
        with MMapWrapper(test_file) as mmap_obj:
            # Test various slices
            assert bytes(mmap_obj[10:20]) == test_data[10:20]
            assert bytes(mmap_obj[:10]) == test_data[:10]
            assert bytes(mmap_obj[-10:]) == test_data[-10:]
            assert bytes(mmap_obj[::2][:10]) == test_data[::2][:10]

    def test_get_slice_method(self, tmp_path):
        """Test get_slice method."""
        test_file = tmp_path / "test.bin"
        test_data = b"0123456789" * 10
        test_file.write_bytes(test_data)
        
        with MMapWrapper(test_file) as mmap_obj:
            # Get slice from start
            assert bytes(mmap_obj.get_slice(0, 10)) == b"0123456789"
            
            # Get slice from middle
            assert bytes(mmap_obj.get_slice(50, 10)) == b"0123456789"
            
            # Get rest of file
            assert bytes(mmap_obj.get_slice(90)) == b"0123456789"

    def test_get_slice_bounds_error(self, tmp_path):
        """Test get_slice with out of bounds."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"short")
        
        with MMapWrapper(test_file) as mmap_obj:
            with pytest.raises(ValueError, match="Slice out of bounds"):
                mmap_obj.get_slice(10, 5)
            
            with pytest.raises(ValueError, match="Slice out of bounds"):
                mmap_obj.get_slice(-1, 5)

    def test_context_manager(self, tmp_path):
        """Test using MMapWrapper as context manager."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        
        with MMapWrapper(test_file) as mmap_obj:
            assert len(mmap_obj) == 9
        
        # mmap should be closed after context

    def test_explicit_close(self, tmp_path):
        """Test explicit close."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test data")
        
        mmap_obj = MMapWrapper(test_file)
        assert mmap_obj._mmap is not None
        
        mmap_obj.close()
        assert mmap_obj._mmap is None
        assert mmap_obj._file is None

    def test_nonexistent_file(self, tmp_path):
        """Test with non-existent file."""
        test_file = tmp_path / "nonexistent.bin"
        
        with pytest.raises(FileIOError, match="Failed to stat file"):
            MMapWrapper(test_file)

    def test_empty_file(self, tmp_path):
        """Test with empty file."""
        test_file = tmp_path / "empty.bin"
        test_file.write_bytes(b"")
        
        with pytest.raises(FileIOError, match="Invalid mapping length"):
            MMapWrapper(test_file)

    def test_read_access(self, tmp_path):
        """Test read-only access mode."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"readonly")
        
        with MMapWrapper(test_file, access=mmap.ACCESS_READ) as mmap_obj:
            assert bytes(mmap_obj[:]) == b"readonly"

    def test_write_access(self, tmp_path):
        """Test write access mode."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"original")
        
        with MMapWrapper(test_file, access=mmap.ACCESS_WRITE) as mmap_obj:
            # In real mmap, we could write here
            # But our wrapper returns memoryviews which are read-only
            assert len(mmap_obj) == 8

    def test_copy_access(self, tmp_path):
        """Test copy-on-write access mode."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"original")
        
        with MMapWrapper(test_file, access=mmap.ACCESS_COPY) as mmap_obj:
            assert len(mmap_obj) == 8

    def test_large_file(self, tmp_path):
        """Test with larger file."""
        test_file = tmp_path / "large.bin"
        # Create 1MB file
        test_data = b"x" * (1024 * 1024)
        test_file.write_bytes(test_data)
        
        with MMapWrapper(test_file) as mmap_obj:
            assert len(mmap_obj) == len(test_data)
            # Read a chunk from middle
            assert bytes(mmap_obj.get_slice(500000, 1000)) == b"x" * 1000

    def test_windows_offset_alignment(self, tmp_path):
        """Test Windows-specific offset alignment."""
        test_file = tmp_path / "test.bin"
        test_data = b"0" * 10000
        test_file.write_bytes(test_data)
        
        # Mock Windows platform
        with patch("platform.system", return_value="Windows"):
            # Use offset that's not aligned to allocation granularity
            with patch("mmap.ALLOCATIONGRANULARITY", 4096):
                with MMapWrapper(test_file, offset=1000, length=100) as mmap_obj:
                    assert len(mmap_obj) == 100
                    assert mmap_obj._view_offset == 1000 % 4096

    def test_windows_no_offset(self, tmp_path):
        """Test Windows with no offset."""
        test_file = tmp_path / "test.bin"
        test_data = b"test data"
        test_file.write_bytes(test_data)
        
        with patch("platform.system", return_value="Windows"):
            with MMapWrapper(test_file) as mmap_obj:
                assert mmap_obj._view_offset == 0
                assert len(mmap_obj) == len(test_data)

    def test_unix_offset_handling(self, tmp_path):
        """Test Unix offset handling."""
        test_file = tmp_path / "test.bin"
        test_data = b"0" * 1000
        test_file.write_bytes(test_data)
        
        with patch("platform.system", return_value="Linux"):
            with MMapWrapper(test_file, offset=100, length=200) as mmap_obj:
                # On Unix, _view_offset is not used
                assert len(mmap_obj) == 200

    def test_fallback_mode(self, tmp_path):
        """Test fallback to regular file read."""
        test_file = tmp_path / "test.bin"
        test_data = b"fallback test data"
        test_file.write_bytes(test_data)
        
        # Force mmap to fail
        with patch("mmap.mmap", side_effect=OSError("mmap failed")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                with MMapWrapper(test_file) as mmap_obj:
                    assert mmap_obj._mmap is None
                    assert mmap_obj._fallback_data == test_data
                    assert len(mmap_obj) == len(test_data)
                    assert mmap_obj[5] == test_data[5]
                    assert bytes(mmap_obj[:10]) == test_data[:10]
                
                # Check warning was issued
                assert len(w) == 1
                assert "mmap failed" in str(w[0].message)

    def test_fallback_get_slice(self, tmp_path):
        """Test get_slice in fallback mode."""
        test_file = tmp_path / "test.bin"
        test_data = b"0123456789" * 10
        test_file.write_bytes(test_data)
        
        with patch("mmap.mmap", side_effect=ValueError("mmap failed")):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                
                with MMapWrapper(test_file) as mmap_obj:
                    slice_data = mmap_obj.get_slice(10, 20)
                    assert bytes(slice_data) == test_data[10:30]

    def test_fallback_with_offset(self, tmp_path):
        """Test fallback mode with offset."""
        test_file = tmp_path / "test.bin"
        test_data = b"0123456789" * 10
        test_file.write_bytes(test_data)
        
        with patch("mmap.mmap", side_effect=OSError("mmap failed")):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                
                with MMapWrapper(test_file, offset=20, length=30) as mmap_obj:
                    assert len(mmap_obj) == 30
                    assert bytes(mmap_obj[:]) == test_data[20:50]

    def test_file_size_calculation(self, tmp_path):
        """Test file size calculation with offset."""
        test_file = tmp_path / "test.bin"
        test_data = b"x" * 1000
        test_file.write_bytes(test_data)
        
        # Test with length exceeding file size
        with MMapWrapper(test_file, offset=900, length=200) as mmap_obj:
            # Should be truncated to available size
            assert len(mmap_obj) == 100

    def test_not_initialized_error(self, tmp_path):
        """Test accessing uninitialized mmap."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")
        
        mmap_obj = MMapWrapper(test_file)
        mmap_obj._mmap = None
        mmap_obj._fallback_data = None
        
        with pytest.raises(RuntimeError, match="MMap not initialized"):
            _ = mmap_obj[0]
        
        with pytest.raises(RuntimeError, match="MMap not initialized"):
            mmap_obj.get_slice(0, 1)

    def test_stat_error(self, tmp_path):
        """Test handling stat errors."""
        test_file = tmp_path / "test.bin"
        
        with patch.object(Path, "stat", side_effect=OSError("Stat failed")):
            with pytest.raises(FileIOError, match="Failed to stat file"):
                MMapWrapper(test_file)

    def test_del_cleanup(self, tmp_path):
        """Test cleanup via __del__."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"test")
        
        mmap_obj = MMapWrapper(test_file)
        # Simulate deletion
        mmap_obj.__del__()
        
        assert mmap_obj._mmap is None

    def test_windows_slice_adjustment(self, tmp_path):
        """Test slice adjustment for Windows offset alignment."""
        test_file = tmp_path / "test.bin"
        test_data = bytes(range(256)) * 100
        test_file.write_bytes(test_data)
        
        with patch("platform.system", return_value="Windows"):
            with patch("mmap.ALLOCATIONGRANULARITY", 64):
                # Offset 100 will be aligned to 64, with view_offset = 36
                with MMapWrapper(test_file, offset=100, length=100) as mmap_obj:
                    # Test slice access
                    result = mmap_obj[10:20]
                    expected_start = 100 + 10
                    assert bytes(result) == test_data[expected_start:expected_start + 10]