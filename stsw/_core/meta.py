"""TensorMeta dataclass and marshal helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

DType = Literal["F16", "F32", "F64", "I8", "I16", "I32", "I64", "BF16", "U8"]

VALID_DTYPES = {"F16", "F32", "F64", "I8", "I16", "I32", "I64", "BF16", "U8"}
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,300}$")


@dataclass(frozen=True, repr=True, eq=True)
class TensorMeta:
    """Metadata for a single tensor in the safetensors file.

    Attributes:
        name: Tensor name (must match ^[A-Za-z0-9_.-]{1,300}$)
        dtype: Data type literal
        shape: Tensor shape tuple
        offset_begin: Byte offset from start of data region
        offset_end: Byte offset from start of data region (exclusive)
        crc32: Optional CRC32 checksum
    """

    name: str
    dtype: DType
    shape: tuple[int, ...]
    offset_begin: int
    offset_end: int
    crc32: int | None = None

    def __post_init__(self) -> None:
        """Validate field constraints."""
        # Use object.__setattr__ for frozen dataclass
        if not NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Invalid tensor name '{self.name}': must match {NAME_PATTERN.pattern}"
            )

        if self.dtype not in VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}': must be one of {VALID_DTYPES}"
            )

        if self.offset_begin < 0:
            raise ValueError(
                f"offset_begin must be non-negative, got {self.offset_begin}"
            )

        if self.offset_end < self.offset_begin:
            raise ValueError(
                f"offset_end ({self.offset_end}) must be >= offset_begin ({self.offset_begin})"
            )

        if any(dim < 0 for dim in self.shape):
            raise ValueError(
                f"All shape dimensions must be non-negative, got {self.shape}"
            )

        if self.crc32 is not None and not (0 <= self.crc32 < 2**32):
            raise ValueError(
                f"CRC32 must be a 32-bit unsigned integer, got {self.crc32}"
            )

    @property
    def nbytes(self) -> int:
        """Total bytes for this tensor's data."""
        return self.offset_end - self.offset_begin

    def to_dict(self) -> dict[str, Any]:
        """Convert to safetensors header format."""
        result: dict[str, Any] = {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offsets": [self.offset_begin, self.offset_end],
        }
        if self.crc32 is not None:
            result["crc32"] = self.crc32
        return result

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> TensorMeta:
        """Create from safetensors header format."""
        if "data_offsets" not in data:
            raise ValueError(f"Missing 'data_offsets' for tensor '{name}'")

        offsets = data["data_offsets"]
        if len(offsets) != 2:
            raise ValueError(
                f"data_offsets must have exactly 2 elements, got {len(offsets)}"
            )

        return cls(
            name,  # positional args
            data["dtype"],
            tuple(data["shape"]),
            offsets[0],
            offsets[1],
            data.get("crc32"),
        )


def validate_tensor_order(metas: list[TensorMeta], align: int = 64) -> None:
    """Validate that tensors are properly ordered and aligned.

    Args:
        metas: List of tensor metadata
        align: Alignment requirement in bytes

    Raises:
        ValueError: If validation fails
    """
    if not metas:
        return

    names_seen = set()
    prev_end = 0

    for i, meta in enumerate(metas):
        if meta.name in names_seen:
            raise ValueError(f"Duplicate tensor name: '{meta.name}'")
        names_seen.add(meta.name)

        if meta.offset_begin < prev_end:
            raise ValueError(
                f"Tensor '{meta.name}' at index {i} has overlapping offsets: "
                f"begins at {meta.offset_begin} but previous tensor ends at {prev_end}"
            )

        if i > 0 and meta.offset_begin % align != 0:
            raise ValueError(
                f"Tensor '{meta.name}' at index {i} is not aligned: "
                f"offset {meta.offset_begin} is not divisible by {align}"
            )

        prev_end = meta.offset_end


def build_aligned_offsets(
    tensors: list[tuple[str, DType, tuple[int, ...], int]], align: int = 64
) -> list[TensorMeta]:
    """Build TensorMeta list with proper alignment.

    Args:
        tensors: List of (name, dtype, shape, nbytes) tuples
        align: Alignment requirement in bytes

    Returns:
        List of TensorMeta with computed offsets
    """
    metas = []
    offset = 0

    for name, dtype, shape, nbytes in tensors:
        meta = TensorMeta(
            name,
            dtype,
            shape,
            offset,
            offset + nbytes,
        )
        metas.append(meta)

        offset += nbytes
        if offset % align != 0:
            offset = ((offset + align - 1) // align) * align

    return metas
