"""Unit tests for meta module."""

import pytest

from stsw._core.meta import (
    TensorMeta,
    build_aligned_offsets,
    validate_tensor_order,
)


class TestTensorMeta:
    """Test TensorMeta dataclass."""

    def test_valid_tensor_meta(self):
        """Test creating valid TensorMeta."""
        meta = TensorMeta(
            name="test_tensor",
            dtype="F32",
            shape=(10, 20),
            offset_begin=0,
            offset_end=800,
            crc32=None,
        )

        assert meta.name == "test_tensor"
        assert meta.dtype == "F32"
        assert meta.shape == (10, 20)
        assert meta.offset_begin == 0
        assert meta.offset_end == 800
        assert meta.nbytes == 800
        assert meta.crc32 is None

    def test_invalid_name_pattern(self):
        """Test invalid tensor names are rejected."""
        with pytest.raises(ValueError, match="Invalid tensor name"):
            TensorMeta(
                name="invalid name!",  # Contains space and !
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
            )

        with pytest.raises(ValueError, match="Invalid tensor name"):
            TensorMeta(
                name="",  # Empty name
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
            )

        with pytest.raises(ValueError, match="Invalid tensor name"):
            TensorMeta(
                name="a" * 301,  # Too long
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
            )

    def test_valid_names(self):
        """Test various valid tensor names."""
        valid_names = [
            "simple",
            "with_underscore",
            "with-dash",
            "with.dot",
            "MixedCase",
            "with123numbers",
            "0starts_with_number",
            "_starts_with_underscore",
        ]

        for name in valid_names:
            meta = TensorMeta(
                name=name,
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
            )
            assert meta.name == name

    def test_invalid_dtype(self):
        """Test invalid dtype is rejected."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            TensorMeta(
                name="test",
                dtype="invalid",  # type: ignore
                shape=(10,),
                offset_begin=0,
                offset_end=40,
            )

    def test_invalid_offsets(self):
        """Test invalid offsets are rejected."""
        # Negative offset_begin
        with pytest.raises(ValueError, match="offset_begin must be non-negative"):
            TensorMeta(
                name="test",
                dtype="F32",
                shape=(10,),
                offset_begin=-1,
                offset_end=40,
            )

        # offset_end < offset_begin
        with pytest.raises(ValueError, match="offset_end .* must be >= offset_begin"):
            TensorMeta(
                name="test",
                dtype="F32",
                shape=(10,),
                offset_begin=100,
                offset_end=50,
            )

    def test_invalid_shape(self):
        """Test invalid shapes are rejected."""
        with pytest.raises(
            ValueError, match="All shape dimensions must be non-negative"
        ):
            TensorMeta(
                name="test",
                dtype="F32",
                shape=(10, -1),
                offset_begin=0,
                offset_end=40,
            )

    def test_invalid_crc32(self):
        """Test invalid CRC32 values are rejected."""
        with pytest.raises(ValueError, match="CRC32 must be a 32-bit unsigned integer"):
            TensorMeta(
                name="test",
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
                crc32=-1,
            )

        with pytest.raises(ValueError, match="CRC32 must be a 32-bit unsigned integer"):
            TensorMeta(
                name="test",
                dtype="F32",
                shape=(10,),
                offset_begin=0,
                offset_end=40,
                crc32=2**32,
            )

    def test_to_dict(self):
        """Test conversion to dict format."""
        meta = TensorMeta(
            name="test",
            dtype="F32",
            shape=(10, 20),
            offset_begin=64,
            offset_end=864,
            crc32=12345,
        )

        d = meta.to_dict()
        assert d == {
            "dtype": "F32",
            "shape": [10, 20],
            "data_offsets": [64, 864],
            "crc32": 12345,
        }

    def test_to_dict_no_crc(self):
        """Test conversion to dict without CRC."""
        meta = TensorMeta(
            name="test",
            dtype="F32",
            shape=(10,),
            offset_begin=0,
            offset_end=40,
            crc32=None,
        )

        d = meta.to_dict()
        assert d == {
            "dtype": "F32",
            "shape": [10],
            "data_offsets": [0, 40],
        }
        assert "crc32" not in d

    def test_from_dict(self):
        """Test creation from dict format."""
        data = {
            "dtype": "I64",
            "shape": [100, 200],
            "data_offsets": [128, 160128],
            "crc32": 98765,
        }

        meta = TensorMeta.from_dict("test_tensor", data)

        assert meta.name == "test_tensor"
        assert meta.dtype == "I64"
        assert meta.shape == (100, 200)
        assert meta.offset_begin == 128
        assert meta.offset_end == 160128
        assert meta.crc32 == 98765

    def test_from_dict_missing_offsets(self):
        """Test from_dict with missing data_offsets."""
        with pytest.raises(ValueError, match="Missing 'data_offsets'"):
            TensorMeta.from_dict("test", {"dtype": "F32", "shape": [10]})

    def test_from_dict_invalid_offsets(self):
        """Test from_dict with invalid data_offsets."""
        with pytest.raises(
            ValueError, match="data_offsets must have exactly 2 elements"
        ):
            TensorMeta.from_dict(
                "test",
                {
                    "dtype": "F32",
                    "shape": [10],
                    "data_offsets": [0, 40, 80],  # Too many
                },
            )


class TestValidateTensorOrder:
    """Test validate_tensor_order function."""

    def test_valid_order(self):
        """Test valid tensor ordering passes."""
        metas = [
            TensorMeta("a", "F32", (10,), 0, 40),
            TensorMeta("b", "F32", (10,), 64, 104),
            TensorMeta("c", "F32", (10,), 128, 168),
        ]

        # Should not raise
        validate_tensor_order(metas, align=64)

    def test_duplicate_names(self):
        """Test duplicate names are rejected."""
        metas = [
            TensorMeta("tensor", "F32", (10,), 0, 40),
            TensorMeta("tensor", "F32", (10,), 64, 104),  # Duplicate
        ]

        with pytest.raises(ValueError, match="Duplicate tensor name"):
            validate_tensor_order(metas)

    def test_overlapping_offsets(self):
        """Test overlapping offsets are rejected."""
        metas = [
            TensorMeta("a", "F32", (10,), 0, 40),
            TensorMeta("b", "F32", (10,), 30, 70),  # Overlaps with a
        ]

        with pytest.raises(ValueError, match="overlapping offsets"):
            validate_tensor_order(metas)

    def test_misaligned_tensor(self):
        """Test misaligned tensors are rejected."""
        metas = [
            TensorMeta("a", "F32", (10,), 0, 40),
            TensorMeta("b", "F32", (10,), 50, 90),  # Not aligned to 64
        ]

        with pytest.raises(ValueError, match="is not aligned"):
            validate_tensor_order(metas, align=64)

    def test_empty_list(self):
        """Test empty list is valid."""
        validate_tensor_order([])  # Should not raise


class TestBuildAlignedOffsets:
    """Test build_aligned_offsets function."""

    def test_single_tensor(self):
        """Test building offsets for single tensor."""
        tensors = [("test", "F32", (10, 10), 400)]
        metas = build_aligned_offsets(tensors, align=64)

        assert len(metas) == 1
        assert metas[0].name == "test"
        assert metas[0].offset_begin == 0
        assert metas[0].offset_end == 400

    def test_multiple_tensors_with_padding(self):
        """Test padding is added between tensors."""
        tensors = [
            ("a", "F32", (10,), 40),  # Needs padding to 64
            ("b", "F32", (20,), 80),  # Needs padding to 128
            ("c", "F32", (30,), 120),  # Needs padding to 192
        ]

        metas = build_aligned_offsets(tensors, align=64)

        assert len(metas) == 3

        assert metas[0].name == "a"
        assert metas[0].offset_begin == 0
        assert metas[0].offset_end == 40

        assert metas[1].name == "b"
        assert metas[1].offset_begin == 64  # Aligned
        assert metas[1].offset_end == 144

        assert metas[2].name == "c"
        assert metas[2].offset_begin == 192  # Aligned
        assert metas[2].offset_end == 312

    def test_already_aligned(self):
        """Test tensors that are already aligned."""
        tensors = [
            ("a", "F32", (16,), 64),
            ("b", "F32", (32,), 128),
        ]

        metas = build_aligned_offsets(tensors, align=64)

        assert metas[0].offset_begin == 0
        assert metas[0].offset_end == 64
        assert metas[1].offset_begin == 64
        assert metas[1].offset_end == 192
