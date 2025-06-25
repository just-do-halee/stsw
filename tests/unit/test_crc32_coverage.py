"""Additional tests for crc32 module to improve coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from stsw._core.crc32 import StreamingCRC32, compute_crc32


class TestStreamingCRC32Coverage:
    """Additional tests for StreamingCRC32."""

    def test_streaming_basic(self):
        """Test basic StreamingCRC32 functionality."""
        crc = StreamingCRC32()
        crc.update(b"test data")
        result = crc.digest()
        
        # Should return a valid CRC32 value
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF

    def test_streaming_empty(self):
        """Test StreamingCRC32 when no data added."""
        crc = StreamingCRC32()
        result = crc.digest()
        
        # Should return 0 when no data
        assert result == 0

    def test_streaming_without_xxhash(self):
        """Test StreamingCRC32 falls back to zlib when xxhash not available."""
        with patch("stsw._core.crc32._xxhash_available", False):
            crc = StreamingCRC32()
            crc.update(b"test")
            crc.update(b" data")
            
            result = crc.digest()
            
            # Should match zlib.crc32
            import zlib
            expected = zlib.crc32(b"test data") & 0xFFFFFFFF
            assert result == expected

    def test_compute_crc32_basic(self):
        """Test compute_crc32 basic functionality."""
        result = compute_crc32(b"test data")
        
        # Should return consistent CRC32
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF
        
        # Should be consistent
        assert compute_crc32(b"test data") == result

    def test_compute_crc32_without_xxhash(self):
        """Test compute_crc32 falls back to zlib."""
        with patch("stsw._core.crc32._xxhash_available", False):
            result = compute_crc32(b"test data")
            
            # Should match zlib.crc32
            import zlib
            expected = zlib.crc32(b"test data") & 0xFFFFFFFF
            assert result == expected

    def test_verify_crc32_with_memoryview(self):
        """Test verify_crc32 with memoryview input."""
        from stsw._core.crc32 import verify_crc32
        
        data = memoryview(b"test data")
        crc = compute_crc32(data)
        
        # Should verify correctly
        assert verify_crc32(data, crc) is True
        
        # Should fail with wrong CRC
        assert verify_crc32(data, crc + 1) is False

    def test_crc32_consistency(self):
        """Test CRC32 consistency across methods."""
        data = b"test data for crc32"
        
        # Compute using direct method
        direct_crc = compute_crc32(data)
        
        # Compute using streaming
        streaming = StreamingCRC32()
        streaming.update(data)
        streaming_crc = streaming.digest()
        
        # Should be the same
        assert direct_crc == streaming_crc

    def test_crc32_edge_cases(self):
        """Test CRC32 with edge case inputs."""
        # Empty data
        assert compute_crc32(b"") == 0
        
        # Single byte
        assert compute_crc32(b"\x00") > 0
        
        # Large data
        large_data = b"x" * 1_000_000
        crc = compute_crc32(large_data)
        assert crc > 0
        assert crc <= 0xFFFFFFFF

    def test_streaming_crc32_multiple_updates(self):
        """Test StreamingCRC32 with multiple updates."""
        crc = StreamingCRC32()
        
        # Update in chunks
        chunks = [b"chunk1", b"chunk2", b"chunk3"]
        for chunk in chunks:
            crc.update(chunk)
        
        result = crc.digest()
        
        # Should match single update
        expected = compute_crc32(b"chunk1chunk2chunk3")
        assert result == expected

    def test_streaming_crc32_with_bytes_and_memoryview(self):
        """Test StreamingCRC32 with mixed input types."""
        crc = StreamingCRC32()
        
        # Mix bytes and memoryview
        crc.update(b"bytes")
        crc.update(memoryview(b"view"))
        
        result = crc.digest()
        expected = compute_crc32(b"bytesview")
        assert result == expected