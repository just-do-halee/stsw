"""Tests for CRC32 module with xxhash available."""

from unittest.mock import MagicMock, patch

from stsw._core.crc32 import StreamingCRC32, compute_crc32, verify_crc32


class TestCRC32WithXXHash:
    """Test CRC32 functionality when xxhash is available."""

    def test_streaming_with_xxhash_available(self):
        """Test StreamingCRC32 when xxhash is available."""
        # Mock xxhash module
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0x12345678

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Should use xxhash
                    assert crc._use_xxhash is True
                    mock_xxhash.xxh32.assert_called_once_with(seed=0)

                    # Update with data
                    crc.update(b"test data")
                    mock_hasher.update.assert_called_once_with(b"test data")

                    # Get digest
                    result = crc.digest()
                    assert result == 0x12345678

    def test_streaming_xxhash_empty_data(self):
        """Test StreamingCRC32 with xxhash on empty data."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Get digest without any data - should return 0
                    result = crc.digest()
                    assert result == 0

                    # Now add empty data
                    crc.update(b"")
                    result = crc.digest()
                    # Still should be 0 since no actual data
                    assert result == 0

    def test_streaming_xxhash_has_data_tracking(self):
        """Test _has_data attribute tracking with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0xABCDEF00

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Initially no _has_data attribute
                    assert not hasattr(crc, "_has_data")

                    # Update with non-empty data
                    crc.update(b"data")

                    # Now should have _has_data
                    assert hasattr(crc, "_has_data")
                    assert crc._has_data is True

                    # Digest should return hasher value
                    result = crc.digest()
                    assert result == 0xABCDEF00

    def test_streaming_xxhash_reset(self):
        """Test reset functionality with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Add data
                    crc.update(b"test")
                    assert hasattr(crc, "_has_data")

                    # Reset
                    crc.reset()

                    # Should reset hasher and remove _has_data
                    mock_hasher.reset.assert_called_once()
                    assert not hasattr(crc, "_has_data")
                    assert crc._crc == 0

    def test_streaming_xxhash_copy(self):
        """Test copy functionality with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_hasher_copy = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.copy.return_value = mock_hasher_copy

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Add data to set _has_data
                    crc.update(b"test")

                    # Copy
                    crc_copy = crc.copy()

                    # Should copy hasher
                    mock_hasher.copy.assert_called_once()
                    assert crc_copy._hasher is mock_hasher_copy
                    assert crc_copy._use_xxhash is True
                    assert hasattr(crc_copy, "_has_data")
                    assert crc_copy._has_data is True

    def test_streaming_xxhash_copy_without_data(self):
        """Test copy functionality with xxhash when no data added."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_hasher_copy = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.copy.return_value = mock_hasher_copy

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Copy without adding data
                    crc_copy = crc.copy()

                    # Should not have _has_data
                    assert not hasattr(crc_copy, "_has_data")

    def test_streaming_xxhash_memoryview(self):
        """Test StreamingCRC32 with memoryview input when xxhash is available."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Update with memoryview
                    data = memoryview(b"test data")
                    crc.update(data)

                    # Should pass memoryview directly to xxhash
                    mock_hasher.update.assert_called_once_with(data)

    def test_compute_crc32_with_xxhash(self):
        """Test compute_crc32 when xxhash is available."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0xDEADBEEF

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    result = compute_crc32(b"test data")
                    assert result == 0xDEADBEEF

    def test_verify_crc32_with_xxhash(self):
        """Test verify_crc32 when xxhash is available."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0xCAFEBABE

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    # Correct CRC
                    assert verify_crc32(b"test", 0xCAFEBABE) is True

                    # Wrong CRC
                    assert verify_crc32(b"test", 0x12345678) is False

    def test_streaming_multiple_updates_xxhash(self):
        """Test multiple updates with xxhash."""
        mock_xxhash = MagicMock()
        mock_hasher = MagicMock()
        mock_xxhash.xxh32.return_value = mock_hasher
        mock_hasher.intdigest.return_value = 0x11223344

        with patch.dict("sys.modules", {"xxhash": mock_xxhash}):
            with patch("stsw._core.crc32._xxhash_available", True):
                with patch("stsw._core.crc32.xxhash", mock_xxhash):
                    crc = StreamingCRC32()

                    # Multiple updates
                    crc.update(b"chunk1")
                    crc.update(b"chunk2")
                    crc.update(b"chunk3")

                    # All updates should be passed to hasher
                    assert mock_hasher.update.call_count == 3
                    calls = mock_hasher.update.call_args_list
                    assert calls[0][0][0] == b"chunk1"
                    assert calls[1][0][0] == b"chunk2"
                    assert calls[2][0][0] == b"chunk3"

                    result = crc.digest()
                    assert result == 0x11223344
