"""StreamReader implementation for safetensors format."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from stsw._core.crc32 import verify_crc32
from stsw._core.dtype import to_numpy, to_torch
from stsw._core.header import HeaderError, extract_tensor_metas, parse_header
from stsw._core.meta import TensorMeta
from stsw.io.fileio import PathLike, get_file_size
from stsw.io.mmapwrap import MMapWrapper

if TYPE_CHECKING:
    import numpy as np
    import torch  # type: ignore[import]

logger = logging.getLogger("stsw")


class StreamReader:
    """Zero-copy reader for safetensors files.

    Uses memory mapping for constant memory usage regardless of file size.
    Supports lazy loading and CRC verification.
    """

    def __init__(
        self,
        path: PathLike,
        *,
        mmap: bool = True,
        verify_crc: bool = False,
    ) -> None:
        """Initialize StreamReader.

        Args:
            path: Path to safetensors file
            mmap: Whether to use memory mapping (recommended)
            verify_crc: Whether to verify CRC32 checksums on first access

        Raises:
            FileNotFoundError: If file doesn't exist
            HeaderError: If header is invalid
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        self.use_mmap = mmap
        self.verify_crc = verify_crc
        self.allow_partial = False  # Always False for v1.0 API compliance

        # Read and parse header
        self._header_dict: dict[str, Any] = {}
        self._header_size: int = 0
        self._data_start: int = 0
        self._tensors: list[TensorMeta] = []
        self._tensor_map: dict[str, TensorMeta] = {}
        self._mmap: MMapWrapper | None = None
        self._crc_verified: dict[str, bool] = {}

        self._load_header()

        # Open mmap if requested
        if self.use_mmap:
            self._open_mmap()

    def _load_header(self) -> None:
        """Load and parse the safetensors header."""
        # Read header size first
        with open(self.path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8:
                raise HeaderError("File too small to contain header")

            import struct

            header_len = struct.unpack("<Q", header_size_bytes)[0]

            # Read full header
            header_data = header_size_bytes + f.read(header_len)

        # Parse header
        try:
            self._header_dict, self._header_size = parse_header(header_data)
        except HeaderError as e:
            raise HeaderError(f"Failed to parse header: {e}") from e

        self._data_start = self._header_size

        # Check for incomplete file
        if self._header_dict.get("__incomplete__", False):
            if not self.allow_partial:
                raise HeaderError(
                    "File is marked as incomplete. "
                    "Set allow_partial=True to read anyway."
                )
            else:
                warnings.warn(
                    "Reading incomplete file - some tensors may be missing",
                    UserWarning,
                    stacklevel=2,
                )

        # Extract tensor metadata
        self._tensors = extract_tensor_metas(self._header_dict)
        self._tensor_map = {t.name: t for t in self._tensors}

        # Validate file size
        if self._tensors:
            expected_size = self._data_start + max(t.offset_end for t in self._tensors)
            actual_size = get_file_size(self.path)

            if actual_size < expected_size:
                if not self.allow_partial:
                    raise HeaderError(
                        f"File too small: expected {expected_size} bytes, "
                        f"got {actual_size} bytes"
                    )

    def _open_mmap(self) -> None:
        """Open memory-mapped file."""
        file_size = get_file_size(self.path)
        if file_size <= self._data_start:
            return  # No data to map

        self._mmap = MMapWrapper(
            self.path,
            length=file_size - self._data_start,
            offset=self._data_start,
        )

    def keys(self) -> list[str]:
        """Get list of tensor names in the file.

        Returns:
            List of tensor names in file order
        """
        return [t.name for t in self._tensors]

    def meta(self, name: str) -> TensorMeta:
        """Get metadata for a specific tensor.

        Args:
            name: Tensor name

        Returns:
            TensorMeta object

        Raises:
            KeyError: If tensor not found
        """
        if name not in self._tensor_map:
            raise KeyError(f"Tensor '{name}' not found in file")
        return self._tensor_map[name]

    def get_slice(self, name: str) -> memoryview:
        """Get raw tensor data as memoryview (zero-copy).

        Args:
            name: Tensor name

        Returns:
            Memoryview of tensor data

        Raises:
            KeyError: If tensor not found
            RuntimeError: If mmap not available
        """
        meta = self.meta(name)

        if self._mmap is None:
            raise RuntimeError("Memory mapping not available")

        # Get slice
        start = meta.offset_begin
        length = meta.nbytes
        data = self._mmap.get_slice(start, length)

        # Verify CRC if requested and not already done
        if self.verify_crc and meta.crc32 is not None:
            if name not in self._crc_verified:
                if not verify_crc32(data, meta.crc32):
                    raise ValueError(f"CRC32 mismatch for tensor '{name}'")
                self._crc_verified[name] = True

        return data

    def to_numpy(self, name: str) -> "np.ndarray[Any, Any]":
        """Load tensor as numpy array.

        Args:
            name: Tensor name

        Returns:
            NumPy array with correct shape and dtype
        """
        import numpy as np

        meta = self.meta(name)
        data = self.get_slice(name)

        # Create numpy array from memoryview
        dtype = to_numpy(meta.dtype)
        flat_array = np.frombuffer(data, dtype=dtype)

        # Reshape to correct shape
        if meta.shape:
            return flat_array.reshape(meta.shape)
        else:
            return flat_array

    def to_torch(
        self, name: str, *, device: Union[str, "torch.device"] = "cpu"
    ) -> "torch.Tensor":
        """Load tensor as PyTorch tensor.

        Args:
            name: Tensor name
            device: Target device (default: "cpu")

        Returns:
            PyTorch tensor with correct shape and dtype
        """
        import torch  # type: ignore[import]

        meta = self.meta(name)

        if str(device) == "cpu":
            # Use numpy intermediate for CPU tensors
            numpy_array = self.to_numpy(name)
            tensor = torch.from_numpy(numpy_array)

            # Convert dtype if needed (e.g., for BF16)
            target_dtype = to_torch(meta.dtype)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)

            return tensor
        else:
            # For GPU, create on CPU first then move
            data = self.get_slice(name)
            dtype = to_torch(meta.dtype)

            # Create flat tensor
            flat_tensor = torch.frombuffer(
                memoryview(data).cast("B"),
                dtype=dtype,
            )

            # Reshape and move to device
            if meta.shape:
                tensor = flat_tensor.reshape(meta.shape)
            else:
                tensor = flat_tensor

            return tensor.to(device)

    def __iter__(self) -> Iterator[str]:
        """Iterate over tensor names in file order."""
        return iter(self.keys())

    def close(self) -> None:
        """Close the reader and release resources."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def __enter__(self) -> "StreamReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @property
    def version(self) -> str:
        """Get safetensors format version."""
        return self._header_dict.get("__version__", "1.0")

    @property  
    def metadata(self) -> dict[str, Any] | None:
        """Get user metadata from header."""
        return self._header_dict.get("__metadata__")

    def __len__(self) -> int:
        """Return number of tensors in the file."""
        return len(self._tensors)
