"""Safe memory-mapped file wrapper with Windows fallback."""

from __future__ import annotations

import mmap
import platform
from pathlib import Path
from typing import Any, BinaryIO

from stsw.io.fileio import FileIOError, PathLike


class MMapWrapper:
    """Cross-platform memory-mapped file wrapper.

    Provides a consistent interface for memory-mapped files with
    graceful fallback for systems that don't support mmap well.
    """

    def __init__(
        self,
        path: PathLike,
        length: int | None = None,
        offset: int = 0,
        access: int = mmap.ACCESS_READ,
    ) -> None:
        """Initialize memory-mapped file.

        Args:
            path: File path to map
            length: Number of bytes to map (None for entire file)
            offset: Offset in file to start mapping
            access: Access mode (ACCESS_READ, ACCESS_WRITE, ACCESS_COPY)

        Raises:
            FileIOError: If mapping fails
        """
        self.path = Path(path).resolve()
        self.offset = offset
        self.access = access
        self._mmap: mmap.mmap | None = None
        self._file: BinaryIO | None = None
        self._fallback_data: bytes | None = None

        # Get file size
        try:
            file_size = self.path.stat().st_size
        except OSError as e:
            raise FileIOError(f"Failed to stat file: {e}") from e

        # Calculate mapping length
        if length is None:
            self.length = file_size - offset
        else:
            self.length = min(length, file_size - offset)

        if self.length <= 0:
            raise FileIOError("Invalid mapping length")

        # Try to create mmap
        self._create_mmap()

    def _create_mmap(self) -> None:
        """Create the memory mapping."""
        # Determine file open mode
        if self.access == mmap.ACCESS_READ:
            mode = "rb"
        elif self.access == mmap.ACCESS_WRITE:
            mode = "r+b"
        else:  # ACCESS_COPY
            mode = "rb"

        try:
            self._file = open(self.path, mode)

            # Platform-specific mmap creation
            if platform.system() == "Windows":
                # Windows mmap requires special handling for offset
                if self.offset > 0:
                    # Windows requires offset to be multiple of allocation granularity
                    alloc_granularity = mmap.ALLOCATIONGRANULARITY
                    aligned_offset = (
                        self.offset // alloc_granularity
                    ) * alloc_granularity
                    extra_offset = self.offset - aligned_offset

                    self._mmap = mmap.mmap(
                        self._file.fileno(),
                        self.length + extra_offset,
                        access=self.access,
                        offset=aligned_offset,
                    )
                    # Create a memoryview to handle the extra offset
                    self._view_offset = extra_offset
                else:
                    self._mmap = mmap.mmap(
                        self._file.fileno(), self.length, access=self.access
                    )
                    self._view_offset = 0
            else:
                # Unix-like systems handle offset normally
                self._mmap = mmap.mmap(
                    self._file.fileno(),
                    self.length,
                    access=self.access,
                    offset=self.offset,
                )
                self._view_offset = 0

        except (OSError, ValueError) as e:
            # Fallback to regular file read
            self._fallback_mode(e)

    def _fallback_mode(self, error: Exception) -> None:
        """Fall back to regular file reading."""
        import warnings

        warnings.warn(
            f"mmap failed ({error}), falling back to regular file read",
            UserWarning,
            stacklevel=3,
        )

        if self._file is not None:
            self._file.close()
            self._file = None

        # Read the requested portion into memory
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            self._fallback_data = f.read(self.length)

    def __getitem__(self, key: int | slice) -> int | memoryview | bytes:
        """Get data by index or slice."""
        if self._fallback_data is not None:
            return self._fallback_data[key]
        elif self._mmap is not None:
            if self._view_offset > 0:
                # Adjust for Windows offset alignment
                if isinstance(key, slice):
                    start = (key.start or 0) + self._view_offset
                    stop = (
                        key.stop if key.stop is not None else self.length
                    ) + self._view_offset
                    adjusted_slice = slice(start, stop, key.step)
                    return self._mmap[adjusted_slice]
                else:
                    result = self._mmap[key + self._view_offset]
                    return result if isinstance(key, int) else memoryview(result)  # type: ignore[arg-type]
            else:
                result = self._mmap[key]
                return result if isinstance(key, int) else memoryview(result)  # type: ignore[arg-type]
        else:
            raise RuntimeError("MMap not initialized")

    def get_slice(self, start: int = 0, length: int | None = None) -> memoryview:
        """Get a memoryview slice of the mapped data.

        Args:
            start: Start offset within the mapping
            length: Number of bytes (None for rest of mapping)

        Returns:
            Memoryview of the requested slice
        """
        if length is None:
            length = self.length - start

        if start < 0 or start + length > self.length:
            raise ValueError("Slice out of bounds")

        if self._fallback_data is not None:
            return memoryview(self._fallback_data)[start : start + length]
        elif self._mmap is not None:
            if self._view_offset > 0:
                adjusted_start = start + self._view_offset
                return memoryview(self._mmap)[adjusted_start : adjusted_start + length]
            else:
                return memoryview(self._mmap)[start : start + length]
        else:
            raise RuntimeError("MMap not initialized")

    def close(self) -> None:
        """Close the memory mapping and file."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

        if self._file is not None:
            self._file.close()
            self._file = None

        self._fallback_data = None

    def __enter__(self) -> "MMapWrapper":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    def __len__(self) -> int:
        """Return the length of the mapping."""
        return self.length

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.close()
