"""Additional tests for dtype module to improve coverage."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from stsw._core.dtype import (
    DTYPE_TO_BYTES,
    DTYPE_TO_NUMPY,
    NUMPY_TO_DTYPE,
    get_itemsize,
    normalize,
    to_numpy,
)


class TestNormalize:
    """Test dtype normalization."""

    def test_normalize_valid_string(self):
        """Test normalizing valid dtype strings."""
        assert normalize("F32") == "F32"
        assert normalize("I64") == "I64"
        assert normalize("BF16") == "BF16"

    def test_normalize_lowercase_string(self):
        """Test normalizing lowercase dtype strings."""
        assert normalize("f32") == "F32"
        assert normalize("i64") == "I64"
        assert normalize("bf16") == "BF16"

    def test_normalize_invalid_string(self):
        """Test normalizing invalid dtype strings."""
        with pytest.raises(ValueError, match="Unknown string dtype"):
            normalize("invalid")

    def test_normalize_numpy_dtype(self):
        """Test normalizing numpy dtypes."""
        import numpy as np

        assert normalize(np.dtype("float32")) == "F32"
        assert normalize(np.dtype("int64")) == "I64"
        assert normalize(np.dtype("float16")) == "F16"

    def test_normalize_numpy_unsupported(self):
        """Test normalizing unsupported numpy dtype."""
        import numpy as np

        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            normalize(np.dtype("complex64"))

    @patch("sys.modules", {"torch": MagicMock()})
    def test_normalize_torch_dtype(self):
        """Test normalizing torch dtypes."""
        # Mock torch module
        mock_torch = sys.modules["torch"]
        mock_dtype = MagicMock()
        mock_dtype.__str__ = lambda self: "torch.float32"

        with patch("stsw._core.dtype.normalize_torch", return_value="F32"):
            result = normalize(mock_dtype)
            assert result == "F32"

    def test_normalize_unknown_object(self):
        """Test normalizing unknown object."""
        with pytest.raises(ValueError, match="Cannot normalize dtype"):
            normalize(object())


class TestNormalizeTorch:
    """Test PyTorch dtype normalization."""

    def test_normalize_torch_dtypes(self):
        """Test normalizing all supported torch dtypes."""
        # Create mock torch module
        mock_torch = MagicMock()

        # Create dtype mocks
        dtypes = {
            "float16": "F16",
            "float32": "F32",
            "float64": "F64",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "int64": "I64",
            "bfloat16": "BF16",
        }

        for torch_name, expected in dtypes.items():
            dtype_mock = MagicMock()
            setattr(mock_torch, torch_name, dtype_mock)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            # Import inside patch context
            from stsw._core.dtype import normalize_torch

            # Test each dtype
            assert normalize_torch(mock_torch.float16) == "F16"
            assert normalize_torch(mock_torch.float32) == "F32"
            assert normalize_torch(mock_torch.float64) == "F64"
            assert normalize_torch(mock_torch.int8) == "I8"
            assert normalize_torch(mock_torch.int16) == "I16"
            assert normalize_torch(mock_torch.int32) == "I32"
            assert normalize_torch(mock_torch.int64) == "I64"
            assert normalize_torch(mock_torch.bfloat16) == "BF16"

    def test_normalize_torch_unsupported(self):
        """Test normalizing unsupported torch dtype."""
        mock_torch = MagicMock()
        unsupported_dtype = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from stsw._core.dtype import normalize_torch

            with pytest.raises(ValueError, match="Unsupported torch dtype"):
                normalize_torch(unsupported_dtype)


class TestToTorch:
    """Test conversion to PyTorch dtypes."""

    def test_to_torch_all_dtypes(self):
        """Test converting all safetensors dtypes to torch."""
        # Create mock torch module
        mock_torch = MagicMock()

        # Create dtype attributes
        dtype_map = {
            "float16": "F16",
            "float32": "F32",
            "float64": "F64",
            "int8": "I8",
            "int16": "I16",
            "int32": "I32",
            "int64": "I64",
            "bfloat16": "BF16",
        }

        for torch_name in dtype_map:
            setattr(mock_torch, torch_name, f"torch.{torch_name}")

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from stsw._core.dtype import to_torch

            assert to_torch("F16") == "torch.float16"
            assert to_torch("F32") == "torch.float32"
            assert to_torch("F64") == "torch.float64"
            assert to_torch("I8") == "torch.int8"
            assert to_torch("I16") == "torch.int16"
            assert to_torch("I32") == "torch.int32"
            assert to_torch("I64") == "torch.int64"
            assert to_torch("BF16") == "torch.bfloat16"

    def test_to_torch_invalid(self):
        """Test converting invalid dtype to torch."""
        mock_torch = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            from stsw._core.dtype import to_torch

            with pytest.raises(ValueError, match="Invalid safetensors dtype"):
                to_torch("INVALID")


class TestToNumpy:
    """Test conversion to numpy dtypes."""

    def test_to_numpy_all_dtypes(self):
        """Test converting all safetensors dtypes to numpy."""
        import numpy as np

        assert to_numpy("F16") == np.dtype("float16")
        assert to_numpy("F32") == np.dtype("float32")
        assert to_numpy("F64") == np.dtype("float64")
        assert to_numpy("I8") == np.dtype("int8")
        assert to_numpy("I16") == np.dtype("int16")
        assert to_numpy("I32") == np.dtype("int32")
        assert to_numpy("I64") == np.dtype("int64")

    def test_to_numpy_bf16_warning(self):
        """Test BF16 conversion warning."""
        import numpy as np

        with pytest.warns(UserWarning, match="BF16 is not natively supported"):
            dtype = to_numpy("BF16")
            assert dtype == np.dtype("float32")

    def test_to_numpy_invalid(self):
        """Test converting invalid dtype to numpy."""
        with pytest.raises(ValueError, match="Invalid safetensors dtype"):
            to_numpy("INVALID")


class TestGetItemsize:
    """Test get_itemsize function."""

    def test_get_itemsize_all_dtypes(self):
        """Test getting itemsize for all dtypes."""
        assert get_itemsize("F16") == 2
        assert get_itemsize("F32") == 4
        assert get_itemsize("F64") == 8
        assert get_itemsize("I8") == 1
        assert get_itemsize("I16") == 2
        assert get_itemsize("I32") == 4
        assert get_itemsize("I64") == 8
        assert get_itemsize("BF16") == 2

    def test_get_itemsize_invalid(self):
        """Test getting itemsize for invalid dtype."""
        with pytest.raises(ValueError, match="Invalid safetensors dtype"):
            get_itemsize("INVALID")


class TestConstants:
    """Test module constants."""

    def test_dtype_to_bytes(self):
        """Test DTYPE_TO_BYTES mapping."""
        assert len(DTYPE_TO_BYTES) == 9
        assert all(isinstance(v, int) for v in DTYPE_TO_BYTES.values())

    def test_numpy_to_dtype(self):
        """Test NUMPY_TO_DTYPE mapping."""
        assert len(NUMPY_TO_DTYPE) == 8  # No BF16
        assert all(v in DTYPE_TO_BYTES for v in NUMPY_TO_DTYPE.values())

    def test_dtype_to_numpy(self):
        """Test DTYPE_TO_NUMPY mapping."""
        assert len(DTYPE_TO_NUMPY) == 9
        assert DTYPE_TO_NUMPY["BF16"] == "float32"  # Special case
