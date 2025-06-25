"""Unit tests for header module."""

import json
import struct

import pytest

from stsw._core.header import (
    HEADER_SIZE_LIMIT,
    SAFETENSORS_VERSION,
    HeaderError,
    build_header,
    extract_tensor_metas,
    parse_header,
    validate_header,
    validate_tensor_entry,
)
from stsw._core.meta import TensorMeta


class TestBuildHeader:
    """Test header building functionality."""

    def test_basic_header(self):
        """Test building a basic header."""
        tensors = [
            TensorMeta("tensor1", "F32", (10, 20), 0, 800),
            TensorMeta("tensor2", "I64", (5, 5), 832, 1032),
        ]

        header_bytes = build_header(tensors, align=64)

        # Check structure
        assert len(header_bytes) >= 8
        header_len = struct.unpack("<Q", header_bytes[:8])[0]
        assert len(header_bytes) == 8 + header_len

        # Check JSON content
        json_bytes = header_bytes[8:]
        header_dict = json.loads(json_bytes.decode("utf-8"))

        assert header_dict["__version__"] == SAFETENSORS_VERSION
        assert "tensor1" in header_dict
        assert "tensor2" in header_dict

        # Check tensor data
        assert header_dict["tensor1"]["dtype"] == "F32"
        assert header_dict["tensor1"]["shape"] == [10, 20]
        assert header_dict["tensor1"]["data_offsets"] == [0, 800]

    def test_header_with_metadata(self):
        """Test header with user metadata."""
        tensors = [TensorMeta("test", "F32", (10,), 0, 40)]
        metadata = {"model": "test_model", "version": 1}

        header_bytes = build_header(tensors, metadata=metadata)

        json_bytes = header_bytes[8:]
        header_dict = json.loads(json_bytes.decode("utf-8"))

        assert header_dict["__metadata__"] == metadata

    def test_header_padding(self):
        """Test header is properly padded."""
        tensors = [TensorMeta("t", "F32", (1,), 0, 4)]

        header_bytes = build_header(tensors, align=64)

        # Total size should be aligned
        assert len(header_bytes) % 64 == 0

        # Check padding is spaces
        json_end = header_bytes.rstrip(b" ")
        padding = header_bytes[len(json_end) :]
        assert all(b == ord(" ") for b in padding)

    def test_header_with_crc32(self):
        """Test header with CRC32 values."""
        tensors = [
            TensorMeta("test", "F32", (10,), 0, 40, crc32=12345),
        ]

        header_bytes = build_header(tensors)
        json_bytes = header_bytes[8:]
        header_dict = json.loads(json_bytes.decode("utf-8"))

        assert header_dict["test"]["crc32"] == 12345

    def test_header_size_limit(self):
        """Test header size limit is enforced."""
        # Create a tensor with a very long name that would exceed limit
        huge_name = "a" * (HEADER_SIZE_LIMIT + 1000)
        tensors = [TensorMeta(huge_name[:300], "F32", (1,), 0, 4)]  # Max name length

        # This should work (name is truncated to valid length)
        build_header(tensors)

        # But creating many tensors to exceed limit should fail
        many_tensors = []
        # Each tensor entry is about 80 bytes in JSON, so we need more than 1.25M tensors
        for i in range(HEADER_SIZE_LIMIT // 50):  # Create enough to exceed limit
            many_tensors.append(
                TensorMeta(f"tensor_{i:06d}", "F32", (1,), i * 64, i * 64 + 4)
            )

        with pytest.raises(HeaderError, match="exceeds limit"):
            build_header(many_tensors)


class TestParseHeader:
    """Test header parsing functionality."""

    def test_parse_valid_header(self):
        """Test parsing a valid header."""
        # Build a header first
        tensors = [
            TensorMeta("test", "F32", (10, 20), 0, 800),
        ]
        header_bytes = build_header(tensors)

        # Parse it back
        header_dict, total_size = parse_header(header_bytes)

        assert total_size == len(header_bytes)
        assert header_dict["__version__"] == SAFETENSORS_VERSION
        assert "test" in header_dict
        assert header_dict["test"]["dtype"] == "F32"
        assert header_dict["test"]["shape"] == [10, 20]
        assert header_dict["test"]["data_offsets"] == [0, 800]

    def test_parse_header_too_short(self):
        """Test parsing header that's too short."""
        with pytest.raises(HeaderError, match="too short"):
            parse_header(b"short")

    def test_parse_header_invalid_length(self):
        """Test parsing header with invalid length."""
        # Length that exceeds limit
        bad_length = struct.pack("<Q", HEADER_SIZE_LIMIT + 1)
        with pytest.raises(HeaderError, match="exceeds limit"):
            parse_header(bad_length + b"dummy")

    def test_parse_incomplete_header(self):
        """Test parsing incomplete header."""
        # Valid length but not enough data
        length_bytes = struct.pack("<Q", 1000)
        with pytest.raises(HeaderError, match="Incomplete header"):
            parse_header(length_bytes + b"not enough data")

    def test_parse_invalid_json(self):
        """Test parsing header with invalid JSON."""
        length_bytes = struct.pack("<Q", 10)
        invalid_json = b"not json!!"

        with pytest.raises(HeaderError, match="Failed to parse header JSON"):
            parse_header(length_bytes + invalid_json)

    def test_parse_invalid_utf8(self):
        """Test parsing header with invalid UTF-8."""
        length_bytes = struct.pack("<Q", 4)
        invalid_utf8 = b"\xff\xfe\xfd\xfc"

        with pytest.raises(HeaderError, match="Failed to parse header JSON"):
            parse_header(length_bytes + invalid_utf8)


class TestValidateHeader:
    """Test header validation."""

    def test_validate_valid_header(self):
        """Test validating a valid header."""
        header = {
            "__version__": SAFETENSORS_VERSION,
            "tensor1": {
                "dtype": "F32",
                "shape": [10, 20],
                "data_offsets": [0, 800],
            },
        }

        # Should not raise
        validate_header(header)

    def test_validate_not_dict(self):
        """Test header must be a dictionary."""
        with pytest.raises(HeaderError, match="must be a dictionary"):
            validate_header([])  # type: ignore

    def test_validate_empty_header(self):
        """Test empty header is invalid."""
        with pytest.raises(HeaderError, match="at least one tensor or metadata"):
            validate_header({})

    def test_validate_metadata_only(self):
        """Test header with only metadata is valid."""
        header = {
            "__metadata__": {"key": "value"},
        }
        validate_header(header)  # Should not raise

    def test_validate_incomplete_not_bool(self):
        """Test __incomplete__ must be boolean."""
        header = {
            "__incomplete__": "true",  # String instead of bool
            "tensor": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 40],
            },
        }

        with pytest.raises(HeaderError, match="__incomplete__ must be a boolean"):
            validate_header(header)

    def test_validate_metadata_not_dict(self):
        """Test __metadata__ must be dictionary."""
        header = {
            "__metadata__": "not a dict",
            "tensor": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 40],
            },
        }

        with pytest.raises(HeaderError, match="__metadata__ must be a dictionary"):
            validate_header(header)

    def test_validate_deep_nesting(self):
        """Test deep nesting detection."""
        # Create deeply nested metadata
        metadata = {}
        current = metadata
        for _ in range(100):  # Exceed depth limit
            current["nested"] = {}
            current = current["nested"]

        header = {
            "__metadata__": metadata,
            "tensor": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 40],
            },
        }

        with pytest.raises(HeaderError, match="JSON depth exceeds limit"):
            validate_header(header)


class TestValidateTensorEntry:
    """Test tensor entry validation."""

    def test_valid_entry(self):
        """Test validating a valid tensor entry."""
        entry = {
            "dtype": "F32",
            "shape": [10, 20, 30],
            "data_offsets": [0, 2400],
        }

        # Should not raise
        validate_tensor_entry("test", entry)

    def test_entry_not_dict(self):
        """Test tensor entry must be dict."""
        with pytest.raises(HeaderError, match="must be a dictionary"):
            validate_tensor_entry("test", "not a dict")

    def test_missing_required_keys(self):
        """Test missing required keys."""
        entry = {
            "dtype": "F32",
            "shape": [10],
            # Missing data_offsets
        }

        with pytest.raises(HeaderError, match="missing required keys"):
            validate_tensor_entry("test", entry)

    def test_shape_not_list(self):
        """Test shape must be list."""
        entry = {
            "dtype": "F32",
            "shape": (10, 20),  # Tuple instead of list
            "data_offsets": [0, 800],
        }

        with pytest.raises(HeaderError, match="shape must be a list"):
            validate_tensor_entry("test", entry)

    def test_negative_shape(self):
        """Test negative shape dimensions."""
        entry = {
            "dtype": "F32",
            "shape": [10, -1],
            "data_offsets": [0, 40],
        }

        with pytest.raises(HeaderError, match="non-negative integers"):
            validate_tensor_entry("test", entry)

    def test_invalid_offsets_format(self):
        """Test invalid offset format."""
        entry = {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [0, 40, 80],  # Too many
        }

        with pytest.raises(HeaderError, match="list of 2 integers"):
            validate_tensor_entry("test", entry)

    def test_negative_offsets(self):
        """Test negative offsets."""
        entry = {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [-10, 40],
        }

        with pytest.raises(HeaderError, match="non-negative integers"):
            validate_tensor_entry("test", entry)

    def test_invalid_offset_order(self):
        """Test end < begin offsets."""
        entry = {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [100, 50],
        }

        with pytest.raises(HeaderError, match="invalid offsets"):
            validate_tensor_entry("test", entry)


class TestExtractTensorMetas:
    """Test extracting TensorMeta objects from header."""

    def test_extract_single_tensor(self):
        """Test extracting single tensor."""
        header = {
            "__version__": SAFETENSORS_VERSION,
            "test": {
                "dtype": "F32",
                "shape": [10, 20],
                "data_offsets": [0, 800],
            },
        }

        metas = extract_tensor_metas(header)

        assert len(metas) == 1
        assert metas[0].name == "test"
        assert metas[0].dtype == "F32"
        assert metas[0].shape == (10, 20)
        assert metas[0].offset_begin == 0
        assert metas[0].offset_end == 800

    def test_extract_multiple_tensors_sorted(self):
        """Test tensors are sorted by offset."""
        header = {
            "b": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [64, 104],
            },
            "a": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 40],
            },
            "c": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [128, 168],
            },
        }

        metas = extract_tensor_metas(header)

        assert len(metas) == 3
        assert [m.name for m in metas] == ["a", "b", "c"]
        assert [m.offset_begin for m in metas] == [0, 64, 128]

    def test_extract_ignores_reserved_keys(self):
        """Test reserved keys are ignored."""
        header = {
            "__version__": SAFETENSORS_VERSION,
            "__metadata__": {"key": "value"},
            "__incomplete__": True,
            "tensor": {
                "dtype": "F32",
                "shape": [10],
                "data_offsets": [0, 40],
            },
        }

        metas = extract_tensor_metas(header)

        assert len(metas) == 1
        assert metas[0].name == "tensor"
