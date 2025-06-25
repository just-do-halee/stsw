"""Header building, parsing, and validation for safetensors format."""

from __future__ import annotations

import json
import struct
from typing import Any

from stsw._core.meta import TensorMeta

HEADER_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB
JSON_DEPTH_LIMIT = 64
SAFETENSORS_VERSION = "1.0"


class HeaderError(Exception):
    """Raised when header parsing or validation fails."""

    pass


def build_header(
    tensors: list[TensorMeta],
    metadata: dict[str, Any] | None = None,
    align: int = 64,
    incomplete: bool = False,
) -> bytes:
    """Build a safetensors header from tensor metadata.

    Args:
        tensors: List of tensor metadata
        metadata: Optional user metadata dict
        align: Alignment for header padding
        incomplete: Whether to mark file as incomplete

    Returns:
        Header bytes (8-byte length prefix + padded JSON)

    Raises:
        HeaderError: If header would exceed size limit
    """
    header_dict: dict[str, Any] = {}

    if metadata:
        header_dict["__metadata__"] = metadata

    if incomplete:
        header_dict["__incomplete__"] = True

    header_dict["__version__"] = SAFETENSORS_VERSION

    for tensor in tensors:
        header_dict[tensor.name] = tensor.to_dict()

    json_bytes = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    json_len = len(json_bytes)

    if json_len > HEADER_SIZE_LIMIT:
        raise HeaderError(
            f"Header size {json_len} bytes exceeds limit of {HEADER_SIZE_LIMIT} bytes"
        )

    padding_needed = 0
    total_size = 8 + json_len
    if total_size % align != 0:
        padding_needed = align - (total_size % align)

    padded_json = json_bytes + b" " * padding_needed

    length_bytes = struct.pack("<Q", len(padded_json))

    return length_bytes + padded_json


def parse_header(data: bytes) -> tuple[dict[str, Any], int]:
    """Parse a safetensors header.

    Args:
        data: Raw bytes containing at least the header

    Returns:
        Tuple of (header dict, total header size including 8-byte prefix)

    Raises:
        HeaderError: If parsing fails
    """
    if len(data) < 8:
        raise HeaderError("Header too short: need at least 8 bytes for length prefix")

    header_len = struct.unpack("<Q", data[:8])[0]

    if header_len > HEADER_SIZE_LIMIT:
        raise HeaderError(
            f"Header size {header_len} bytes exceeds limit of {HEADER_SIZE_LIMIT} bytes"
        )

    if len(data) < 8 + header_len:
        raise HeaderError(
            f"Incomplete header: expected {header_len} bytes, got {len(data) - 8}"
        )

    json_bytes = data[8 : 8 + header_len]

    try:
        header_dict = json.loads(json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise HeaderError(f"Failed to parse header JSON: {e}") from e

    validate_header(header_dict)

    return header_dict, 8 + header_len


def validate_header(header: dict[str, Any], check_depth: int = 0) -> None:
    """Validate a parsed header dictionary.

    Args:
        header: Parsed header dictionary
        check_depth: Current recursion depth (for nested validation)

    Raises:
        HeaderError: If validation fails
    """
    if check_depth > JSON_DEPTH_LIMIT:
        raise HeaderError(f"JSON depth exceeds limit of {JSON_DEPTH_LIMIT}")

    if not isinstance(header, dict):  # type: ignore[reportUnnecessaryIsInstance]
        raise HeaderError("Header must be a dictionary")

    reserved_keys = {"__metadata__", "__version__", "__incomplete__"}
    tensor_names = []

    for key, value in header.items():
        if key in reserved_keys:
            if key == "__version__" and value != SAFETENSORS_VERSION:
                pass  # Allow future versions
            elif key == "__incomplete__" and not isinstance(value, bool):
                raise HeaderError("__incomplete__ must be a boolean")
            elif key == "__metadata__":
                if not isinstance(value, dict):
                    raise HeaderError("__metadata__ must be a dictionary")
                # Metadata values are arbitrary, just check depth
                _check_json_depth(value, check_depth + 1)
        else:
            tensor_names.append(key)
            validate_tensor_entry(key, value)

    if not tensor_names and "__metadata__" not in header:
        raise HeaderError("Header must contain at least one tensor or metadata")


def validate_tensor_entry(name: str, entry: Any) -> None:
    """Validate a single tensor entry in the header.

    Args:
        name: Tensor name
        entry: Tensor metadata dict

    Raises:
        HeaderError: If validation fails
    """
    if not isinstance(entry, dict):
        raise HeaderError(f"Tensor entry '{name}' must be a dictionary")

    required_keys = {"dtype", "shape", "data_offsets"}
    missing_keys = required_keys - set(entry.keys())
    if missing_keys:
        raise HeaderError(f"Tensor '{name}' missing required keys: {missing_keys}")

    if not isinstance(entry["shape"], list):
        raise HeaderError(f"Tensor '{name}' shape must be a list")

    if not all(isinstance(dim, int) and dim >= 0 for dim in entry["shape"]):
        raise HeaderError(
            f"Tensor '{name}' shape must contain only non-negative integers"
        )

    offsets = entry["data_offsets"]
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise HeaderError(f"Tensor '{name}' data_offsets must be a list of 2 integers")

    if not all(isinstance(off, int) and off >= 0 for off in offsets):
        raise HeaderError(f"Tensor '{name}' data_offsets must be non-negative integers")

    if offsets[1] < offsets[0]:
        raise HeaderError(
            f"Tensor '{name}' has invalid offsets: end ({offsets[1]}) < begin ({offsets[0]})"
        )


def _check_json_depth(obj: Any, depth: int) -> None:
    """Recursively check JSON depth limit.

    Args:
        obj: JSON object to check
        depth: Current recursion depth

    Raises:
        HeaderError: If depth exceeds limit
    """
    if depth > JSON_DEPTH_LIMIT:
        raise HeaderError(f"JSON depth exceeds limit of {JSON_DEPTH_LIMIT}")

    if isinstance(obj, dict):
        for value in obj.values():
            _check_json_depth(value, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_json_depth(item, depth + 1)


def extract_tensor_metas(header: dict[str, Any]) -> list[TensorMeta]:
    """Extract TensorMeta objects from a parsed header.

    Args:
        header: Parsed header dictionary

    Returns:
        List of TensorMeta objects, sorted by offset
    """
    metas = []

    for key, value in header.items():
        if not key.startswith("__"):
            meta = TensorMeta.from_dict(key, value)
            metas.append(meta)

    metas.sort(key=lambda m: m.offset_begin)  # type: ignore[no-any-return]

    return metas
