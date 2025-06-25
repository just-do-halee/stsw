"""Unit tests for dtype module."""

import pytest

from stsw._core import dtype
from stsw._core.meta import VALID_DTYPES


class TestDTypeConversions:
    """Test dtype conversion functions."""

    def test_normalize_string_dtypes(self):
        """Test normalizing string dtypes."""
        assert dtype.normalize("F32") == "F32"
        assert dtype.normalize("f32") == "F32"
        assert dtype.normalize("I64") == "I64"
        assert dtype.normalize("i64") == "I64"
        assert dtype.normalize("BF16") == "BF16"
        assert dtype.normalize("bf16") == "BF16"

    def test_normalize_invalid_string(self):
        """Test invalid string dtype raises error."""
        with pytest.raises(ValueError, match="Unknown string dtype"):
            dtype.normalize("invalid")

        with pytest.raises(ValueError, match="Unknown string dtype"):
            dtype.normalize("float32")  # Wrong format

    def test_normalize_numpy_dtypes(self):
        """Test normalizing numpy dtypes."""
        import numpy as np

        assert dtype.normalize(np.dtype("float16")) == "F16"
        assert dtype.normalize(np.dtype("float32")) == "F32"
        assert dtype.normalize(np.dtype("float64")) == "F64"
        assert dtype.normalize(np.dtype("int8")) == "I8"
        assert dtype.normalize(np.dtype("int16")) == "I16"
        assert dtype.normalize(np.dtype("int32")) == "I32"
        assert dtype.normalize(np.dtype("int64")) == "I64"

    def test_normalize_unsupported_numpy(self):
        """Test unsupported numpy dtype raises error."""
        import numpy as np

        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            dtype.normalize(np.dtype("complex64"))

    @pytest.mark.skipif(
        not pytest.importorskip("torch"), reason="PyTorch not installed"
    )
    def test_normalize_torch_dtypes(self):
        """Test normalizing PyTorch dtypes."""
        import torch

        assert dtype.normalize(torch.float16) == "F16"
        assert dtype.normalize(torch.float32) == "F32"
        assert dtype.normalize(torch.float64) == "F64"
        assert dtype.normalize(torch.int8) == "I8"
        assert dtype.normalize(torch.int16) == "I16"
        assert dtype.normalize(torch.int32) == "I32"
        assert dtype.normalize(torch.int64) == "I64"
        assert dtype.normalize(torch.bfloat16) == "BF16"

    @pytest.mark.skipif(
        not pytest.importorskip("torch"), reason="PyTorch not installed"
    )
    def test_to_torch(self):
        """Test converting to PyTorch dtypes."""
        import torch

        assert dtype.to_torch("F16") == torch.float16
        assert dtype.to_torch("F32") == torch.float32
        assert dtype.to_torch("F64") == torch.float64
        assert dtype.to_torch("I8") == torch.int8
        assert dtype.to_torch("I16") == torch.int16
        assert dtype.to_torch("I32") == torch.int32
        assert dtype.to_torch("I64") == torch.int64
        assert dtype.to_torch("BF16") == torch.bfloat16

    def test_to_numpy(self):
        """Test converting to numpy dtypes."""
        import numpy as np

        assert dtype.to_numpy("F16") == np.dtype("float16")
        assert dtype.to_numpy("F32") == np.dtype("float32")
        assert dtype.to_numpy("F64") == np.dtype("float64")
        assert dtype.to_numpy("I8") == np.dtype("int8")
        assert dtype.to_numpy("I16") == np.dtype("int16")
        assert dtype.to_numpy("I32") == np.dtype("int32")
        assert dtype.to_numpy("I64") == np.dtype("int64")

    def test_to_numpy_bf16_warning(self):
        """Test BF16 conversion to numpy warns."""
        with pytest.warns(UserWarning, match="BF16 is not natively supported"):
            result = dtype.to_numpy("BF16")
            assert result == pytest.importorskip("numpy").dtype("float32")

    def test_get_itemsize(self):
        """Test getting item size for dtypes."""
        assert dtype.get_itemsize("F16") == 2
        assert dtype.get_itemsize("F32") == 4
        assert dtype.get_itemsize("F64") == 8
        assert dtype.get_itemsize("I8") == 1
        assert dtype.get_itemsize("I16") == 2
        assert dtype.get_itemsize("I32") == 4
        assert dtype.get_itemsize("I64") == 8
        assert dtype.get_itemsize("BF16") == 2

    def test_get_itemsize_invalid(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError, match="Invalid safetensors dtype"):
            dtype.get_itemsize("invalid")

    def test_all_valid_dtypes_have_sizes(self):
        """Test all valid dtypes have defined sizes."""
        for valid_dtype in VALID_DTYPES:
            size = dtype.get_itemsize(valid_dtype)
            assert size > 0
            assert size <= 8  # Max 64 bits
