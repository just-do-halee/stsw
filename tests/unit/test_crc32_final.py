"""Final tests to achieve full coverage for crc32.py."""

from unittest.mock import MagicMock, patch

from stsw._core.crc32 import StreamingCRC32


class TestCRC32FinalCoverage:
    """Final tests for complete CRC32 coverage."""

    def test_streaming_reset_zlib_path(self):
        """Test reset() when using zlib (no xxhash)."""
        with patch("stsw._core.crc32._xxhash_available", False):
            crc = StreamingCRC32()

            # Add some data
            crc.update(b"test data")
            assert crc._crc != 0

            # Reset
            crc.reset()
            assert crc._crc == 0

            # Digest should be 0 after reset
            assert crc.digest() == 0

    def test_streaming_copy_zlib_path(self):
        """Test copy() when using zlib (no xxhash)."""
        with patch("stsw._core.crc32._xxhash_available", False):
            crc = StreamingCRC32()

            # Add data
            crc.update(b"test data")
            original_crc = crc._crc

            # Copy
            crc_copy = crc.copy()

            # Verify copy
            assert crc_copy._crc == original_crc
            assert crc_copy._use_xxhash is False
            assert crc_copy.digest() == crc.digest()

            # Verify independence
            crc.update(b"more data")
            assert crc_copy._crc == original_crc  # Copy unchanged

    def test_streaming_memoryview_zlib_path(self):
        """Test memoryview conversion in zlib path."""
        with patch("stsw._core.crc32._xxhash_available", False):
            crc = StreamingCRC32()

            # Update with memoryview
            data = memoryview(b"test data")
            crc.update(data)

            # Should work correctly
            result = crc.digest()

            # Compare with direct bytes
            crc2 = StreamingCRC32()
            crc2.update(b"test data")
            assert result == crc2.digest()

    def test_xxhash_fallback_import_order(self):
        """Test xxhash import handling."""
        # Save original state
        import sys

        original_xxhash = sys.modules.get("xxhash")

        try:
            # Remove xxhash if present
            if "xxhash" in sys.modules:
                del sys.modules["xxhash"]

            # Reload module to test import
            import importlib

            import stsw._core.crc32

            importlib.reload(stsw._core.crc32)

            # Should work without xxhash
            from stsw._core.crc32 import StreamingCRC32

            crc = StreamingCRC32()
            crc.update(b"test")
            assert crc.digest() > 0

        finally:
            # Restore original state
            if original_xxhash is not None:
                sys.modules["xxhash"] = original_xxhash
            # Reload again to restore
            importlib.reload(stsw._core.crc32)

    def test_edge_case_empty_update(self):
        """Test updating with empty data."""
        # With zlib
        with patch("stsw._core.crc32._xxhash_available", False):
            crc = StreamingCRC32()
            crc.update(b"")
            assert crc.digest() == 0

            crc.update(b"data")
            assert crc.digest() > 0

    def test_has_data_attribute_edge_cases(self):
        """Test _has_data attribute edge cases with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0x12345678

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Update with empty data
                    crc.update(b"")
                    # Should not set _has_data
                    assert not hasattr(crc, "_has_data")

                    # Digest should still be 0
                    assert crc.digest() == 0

    def test_xxhash_reset_delattr_branch(self):
        """Test reset() delattr branch with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Reset without _has_data attribute
                    crc.reset()
                    # Should not error
                    assert crc._crc == 0
