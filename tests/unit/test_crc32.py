"""Unit tests for CRC32 module."""

from stsw._core.crc32 import StreamingCRC32, compute_crc32, verify_crc32


class TestStreamingCRC32:
    """Test StreamingCRC32 class."""

    def test_empty_crc(self):
        """Test CRC of empty data."""
        crc = StreamingCRC32()
        assert crc.digest() == 0

    def test_single_update(self):
        """Test single update."""
        crc = StreamingCRC32()
        crc.update(b"hello")
        result = crc.digest()

        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_multiple_updates(self):
        """Test multiple updates give same result as single."""
        # Single update
        crc1 = StreamingCRC32()
        crc1.update(b"hello world")

        # Multiple updates
        crc2 = StreamingCRC32()
        crc2.update(b"hello")
        crc2.update(b" ")
        crc2.update(b"world")

        assert crc1.digest() == crc2.digest()

    def test_memoryview_update(self):
        """Test updating with memoryview."""
        data = bytearray(b"test data")
        view = memoryview(data)

        crc = StreamingCRC32()
        crc.update(view)
        result = crc.digest()

        # Should work without error
        assert isinstance(result, int)

    def test_reset(self):
        """Test resetting CRC calculator."""
        crc = StreamingCRC32()
        crc.update(b"some data")
        first_result = crc.digest()

        crc.reset()
        assert crc.digest() == 0

        crc.update(b"some data")
        assert crc.digest() == first_result

    def test_copy(self):
        """Test copying CRC state."""
        crc1 = StreamingCRC32()
        crc1.update(b"partial")

        # Copy the state
        crc2 = crc1.copy()

        # Continue updating separately
        crc1.update(b" data 1")
        crc2.update(b" data 2")

        # Results should be different
        assert crc1.digest() != crc2.digest()

        # But copying again should preserve state
        crc3 = crc1.copy()
        assert crc3.digest() == crc1.digest()

    def test_large_data(self):
        """Test CRC with large data."""
        # 1 MB of data
        data = b"x" * (1024 * 1024)

        crc = StreamingCRC32()
        crc.update(data)
        result = crc.digest()

        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_chunked_large_data(self):
        """Test CRC with large data in chunks."""
        # Generate test data
        total_size = 1024 * 1024  # 1 MB
        chunk_size = 8192  # 8 KB chunks

        # Single update
        crc1 = StreamingCRC32()
        full_data = b"a" * total_size
        crc1.update(full_data)

        # Chunked updates
        crc2 = StreamingCRC32()
        for i in range(0, total_size, chunk_size):
            chunk = full_data[i : i + chunk_size]
            crc2.update(chunk)

        assert crc1.digest() == crc2.digest()


class TestComputeCRC32:
    """Test compute_crc32 function."""

    def test_compute_bytes(self):
        """Test computing CRC of bytes."""
        result = compute_crc32(b"test data")
        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_compute_memoryview(self):
        """Test computing CRC of memoryview."""
        data = bytearray(b"test data")
        view = memoryview(data)

        result = compute_crc32(view)
        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_compute_empty(self):
        """Test computing CRC of empty data."""
        assert compute_crc32(b"") == 0

    def test_compute_consistency(self):
        """Test compute gives same result for same data."""
        data = b"consistent data"
        result1 = compute_crc32(data)
        result2 = compute_crc32(data)

        assert result1 == result2


class TestVerifyCRC32:
    """Test verify_crc32 function."""

    def test_verify_correct(self):
        """Test verifying correct CRC."""
        data = b"test data"
        crc = compute_crc32(data)

        assert verify_crc32(data, crc) is True

    def test_verify_incorrect(self):
        """Test verifying incorrect CRC."""
        data = b"test data"
        crc = compute_crc32(data)

        # Modify CRC
        wrong_crc = (crc + 1) % (2**32)
        assert verify_crc32(data, wrong_crc) is False

    def test_verify_corrupted_data(self):
        """Test verifying corrupted data."""
        data = b"original data"
        crc = compute_crc32(data)

        # Corrupt the data
        corrupted = b"modified data"
        assert verify_crc32(corrupted, crc) is False

    def test_verify_empty_data(self):
        """Test verifying empty data."""
        assert verify_crc32(b"", 0) is True
        assert verify_crc32(b"", 12345) is False


class TestCRC32Properties:
    """Property-based tests for CRC32."""

    def test_different_data_different_crc(self):
        """Test different data produces different CRCs (mostly)."""
        # Generate some test cases
        test_data = [
            b"",
            b"a",
            b"aa",
            b"aaa",
            b"abc",
            b"xyz",
            b"hello world",
            b"Hello World",  # Case sensitive
            b"\x00\x01\x02\x03",
            b"\xff\xfe\xfd\xfc",
        ]

        crcs = [compute_crc32(data) for data in test_data]

        # Most should be unique (CRC32 can have collisions but unlikely for small test set)
        unique_crcs = set(crcs)
        assert len(unique_crcs) >= len(crcs) * 0.9  # Allow up to 10% collisions

    def test_append_property(self):
        """Test appending data maintains CRC relationship."""
        base = b"base data"
        suffix1 = b" suffix 1"
        suffix2 = b" suffix 2"

        # CRC of base + suffix1
        crc1 = compute_crc32(base + suffix1)

        # CRC of base + suffix2
        crc2 = compute_crc32(base + suffix2)

        # Should be different
        assert crc1 != crc2

        # But streaming should match non-streaming
        crc_stream = StreamingCRC32()
        crc_stream.update(base)
        crc_stream.update(suffix1)

        assert crc_stream.digest() == crc1
