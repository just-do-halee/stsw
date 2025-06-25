"""Data type conversions between safetensors, numpy, and torch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from stsw._core.meta import VALID_DTYPES, DType

if TYPE_CHECKING:
    import numpy as np
    import torch


DTYPE_TO_BYTES: dict[DType, int] = {
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BF16": 2,
    "U8": 1,
}


NUMPY_TO_DTYPE: dict[str, DType] = {
    "float16": "F16",
    "float32": "F32",
    "float64": "F64",
    "int8": "I8",
    "int16": "I16",
    "int32": "I32",
    "int64": "I64",
    "uint8": "U8",
}


DTYPE_TO_NUMPY: dict[DType, str] = {
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "BF16": "float32",  # NumPy doesn't have native bfloat16
    "U8": "uint8",
}


def normalize(dtype: str | Any) -> DType:
    """Normalize various dtype representations to safetensors format.

    Args:
        dtype: String dtype or torch/numpy dtype object

    Returns:
        Normalized dtype string

    Raises:
        ValueError: If dtype is not supported
    """
    if isinstance(dtype, str):
        if dtype in VALID_DTYPES:
            return dtype  # type: ignore
        dtype_upper = dtype.upper()
        if dtype_upper in VALID_DTYPES:
            return dtype_upper  # type: ignore
        raise ValueError(f"Unknown string dtype: {dtype}")

    dtype_str = str(dtype)

    if "torch" in dtype_str:
        return normalize_torch(dtype)

    if hasattr(dtype, "name"):
        numpy_name = dtype.name
        if numpy_name in NUMPY_TO_DTYPE:
            return NUMPY_TO_DTYPE[numpy_name]
        raise ValueError(f"Unsupported numpy dtype: {numpy_name}")

    raise ValueError(f"Cannot normalize dtype: {dtype}")


def normalize_torch(dtype: torch.dtype) -> DType:
    """Convert PyTorch dtype to safetensors format.

    Args:
        dtype: PyTorch dtype

    Returns:
        Safetensors dtype string

    Raises:
        ValueError: If dtype is not supported
    """
    import torch

    torch_to_st: dict[torch.dtype, DType] = {
        torch.float16: "F16",
        torch.float32: "F32",
        torch.float64: "F64",
        torch.int8: "I8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.bfloat16: "BF16",
        torch.uint8: "U8",
    }

    if dtype not in torch_to_st:
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    return torch_to_st[dtype]


def to_torch(dtype: DType) -> torch.dtype:
    """Convert safetensors dtype to PyTorch dtype.

    Args:
        dtype: Safetensors dtype string

    Returns:
        PyTorch dtype

    Raises:
        ValueError: If dtype is not valid
    """
    import torch

    if dtype not in VALID_DTYPES:
        raise ValueError(f"Invalid safetensors dtype: {dtype}")

    st_to_torch: dict[DType, torch.dtype] = {
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BF16": torch.bfloat16,
        "U8": torch.uint8,
    }

    return st_to_torch[dtype]


def to_numpy(dtype: DType) -> np.dtype[Any]:
    """Convert safetensors dtype to numpy dtype.

    Args:
        dtype: Safetensors dtype string

    Returns:
        NumPy dtype

    Raises:
        ValueError: If dtype is not valid
    """
    import numpy as np

    if dtype not in VALID_DTYPES:
        raise ValueError(f"Invalid safetensors dtype: {dtype}")

    if dtype == "BF16":
        import warnings

        warnings.warn(
            "BF16 is not natively supported by NumPy, using float32 instead",
            UserWarning,
            stacklevel=2,
        )

    return np.dtype(DTYPE_TO_NUMPY[dtype])


def get_itemsize(dtype: DType) -> int:
    """Get the size in bytes of a single element of the given dtype.

    Args:
        dtype: Safetensors dtype string

    Returns:
        Size in bytes

    Raises:
        ValueError: If dtype is not valid
    """
    if dtype not in DTYPE_TO_BYTES:
        raise ValueError(f"Invalid safetensors dtype: {dtype}")

    return DTYPE_TO_BYTES[dtype]
