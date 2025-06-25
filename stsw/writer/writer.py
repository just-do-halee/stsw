"""StreamWriter implementation for safetensors format."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

if TYPE_CHECKING:
    import psutil  # pyright: ignore[reportUnusedImport]

from stsw._core.crc32 import StreamingCRC32
from stsw._core.header import build_header
from stsw._core.meta import TensorMeta, validate_tensor_order
from stsw.io.fileio import PathLike, SafeFileWriter, pwrite

logger = logging.getLogger("stsw")


class TensorOrderError(Exception):
    """Raised when tensors are written out of order."""

    pass


class LengthMismatchError(Exception):
    """Raised when tensor data length doesn't match metadata."""

    pass


@dataclass(frozen=True)
class WriterStats:
    """Live telemetry for StreamWriter."""

    written: int
    total: int
    mb_per_s: float
    eta_s: float
    rss_mb: float


class StreamWriter:
    """Streaming writer for safetensors format.

    Thread-safe implementation that writes tensors in forward-only stream
    with optional CRC32 checksums and atomic file operations.
    """

    def __init__(
        self,
        path: PathLike,
        tensors: Sequence[TensorMeta],
        *,
        align: int = 64,
        buffer_size: int = 4 << 20,  # 4 MiB
        crc32: bool = False,
        resume: bool = False,
    ) -> None:
        """Private constructor. Use StreamWriter.open() instead."""
        self.path = Path(path)
        self.tensors = list(tensors)
        self.align = align
        self.buffer_size = buffer_size
        self.enable_crc32 = crc32

        # Validate tensors
        validate_tensor_order(self.tensors, align)

        # Build tensor lookup
        self._tensor_map: dict[str, TensorMeta] = {t.name: t for t in self.tensors}
        self._tensor_index: dict[str, int] = {
            t.name: i for i, t in enumerate(self.tensors)
        }

        # Initialize state
        self._lock = threading.Lock()
        self._file_writer: SafeFileWriter | None = None
        self._file: BinaryIO | None = None
        self._header_size: int = 0
        self._data_start: int = 0
        self._current_tensor_idx: int = 0
        self._current_tensor_written: int = 0
        self._total_written: int = 0
        self._start_time = time.time()

        # CRC32 state
        self._crc_calculators: dict[str, StreamingCRC32] = {}
        if self.enable_crc32:
            for tensor in self.tensors:
                self._crc_calculators[tensor.name] = StreamingCRC32()

        # Calculate total size
        self._total_size = sum(t.nbytes for t in self.tensors)

        # Open file and write header
        self._initialize_file(resume)

    @classmethod
    def open(
        cls,
        path: PathLike,
        tensors: Sequence[TensorMeta],
        *,
        align: int = 64,
        buffer_size: int = 4 << 20,  # 4 MiB as per spec
        crc32: bool = False,
    ) -> StreamWriter:
        """Open a new StreamWriter.

        Args:
            path: Output file path
            tensors: Sequence of tensor metadata
            align: Alignment for tensor data (must be power of 2)
            buffer_size: I/O buffer size in bytes
            crc32: Whether to compute CRC32 checksums

        Returns:
            StreamWriter instance
        """
        return cls(
            path=path,
            tensors=tensors,
            align=align,
            buffer_size=buffer_size,
            crc32=crc32,
            resume=False,  # Always False for v1.0 API compliance
        )

    def _initialize_file(self, resume: bool) -> None:
        """Initialize file and write header."""
        # Build header (with incomplete marker)
        header_bytes = build_header(
            self.tensors,
            metadata=None,
            align=self.align,
            incomplete=True,
        )
        self._header_size = len(header_bytes)
        self._data_start = self._header_size

        # Create file writer
        self._file_writer = SafeFileWriter(self.path, buffer_size=self.buffer_size)
        self._file = self._file_writer.open()

        # Write header
        self._file_writer.write(header_bytes)
        self._file_writer.flush()

    def write_block(self, name: str, data: bytes | memoryview) -> None:
        """Write a block of data for the specified tensor.

        Args:
            name: Tensor name
            data: Data block to write

        Raises:
            TensorOrderError: If tensor is written out of order
            LengthMismatchError: If total data exceeds tensor size
        """
        with self._lock:
            # Validate tensor exists
            if name not in self._tensor_map:
                raise ValueError(f"Unknown tensor: {name}")

            # Check order
            expected_idx = self._current_tensor_idx
            actual_idx = self._tensor_index[name]

            if actual_idx < expected_idx:
                raise TensorOrderError(f"Tensor '{name}' already finalized")
            elif actual_idx > expected_idx:
                expected_name = self.tensors[expected_idx].name
                raise TensorOrderError(
                    f"Expected tensor '{expected_name}' but got '{name}'"
                )

            # Check length
            tensor = self._tensor_map[name]
            remaining = tensor.nbytes - self._current_tensor_written
            data_len = len(data)

            if data_len > remaining:
                raise LengthMismatchError(
                    f"Tensor '{name}' expects {remaining} more bytes, "
                    f"but got {data_len}"
                )

            # Convert memoryview to bytes if needed
            if isinstance(data, memoryview):
                data_bytes = bytes(data)
            else:
                data_bytes = data

            # Write data
            assert self._file_writer is not None
            self._file_writer.write(data_bytes)

            # Update CRC
            if self.enable_crc32:
                self._crc_calculators[name].update(data_bytes)

            # Update counters
            self._current_tensor_written += data_len
            self._total_written += data_len

    def finalize_tensor(self, name: str) -> None:
        """Finalize the current tensor and prepare for the next one.

        Args:
            name: Tensor name to finalize

        Raises:
            TensorOrderError: If wrong tensor is finalized
            LengthMismatchError: If tensor data is incomplete
        """
        with self._lock:
            # Validate it's the current tensor
            if self._current_tensor_idx >= len(self.tensors):
                raise TensorOrderError("All tensors already finalized")

            current_tensor = self.tensors[self._current_tensor_idx]
            if current_tensor.name != name:
                raise TensorOrderError(
                    f"Expected to finalize '{current_tensor.name}' but got '{name}'"
                )

            # Check all data was written
            if self._current_tensor_written != current_tensor.nbytes:
                raise LengthMismatchError(
                    f"Tensor '{name}' expects {current_tensor.nbytes} bytes, "
                    f"but only {self._current_tensor_written} were written"
                )

            # Add alignment padding
            padding_needed = 0
            next_offset = current_tensor.offset_end

            if next_offset % self.align != 0:
                padding_needed = self.align - (next_offset % self.align)
                padding = b"\x00" * padding_needed

                assert self._file_writer is not None
                self._file_writer.write(padding)
                self._total_written += padding_needed

            # Store CRC if enabled
            if self.enable_crc32:
                crc_value = self._crc_calculators[name].digest()
                # Update tensor metadata with CRC
                self.tensors[self._current_tensor_idx] = TensorMeta(
                    current_tensor.name,
                    current_tensor.dtype,
                    current_tensor.shape,
                    current_tensor.offset_begin,
                    current_tensor.offset_end,
                    crc_value,
                )

            # Move to next tensor
            self._current_tensor_idx += 1
            self._current_tensor_written = 0

            # Log progress
            logger.debug(
                f"Finalized tensor '{name}' "
                f"({self._current_tensor_idx}/{len(self.tensors)})"
            )

    def close(self) -> None:
        """Close the writer and finalize the file."""
        with self._lock:
            if self._file_writer is None:
                return

            # Verify all tensors were written
            if self._current_tensor_idx < len(self.tensors):
                remaining = [t.name for t in self.tensors[self._current_tensor_idx :]]
                raise RuntimeError(
                    f"Cannot close: {len(remaining)} tensors not written: {remaining}"
                )

            # Update header with CRC values if enabled
            if self.enable_crc32:
                final_header = build_header(
                    self.tensors,
                    metadata=None,
                    align=self.align,
                    incomplete=False,
                )

                # Use pwrite to update header without seeking
                assert self._file is not None
                pwrite(self._file.fileno(), final_header[8:], 8)
            else:
                # Just remove incomplete marker
                header_dict = {"__version__": "1.0"}
                for tensor in self.tensors:
                    header_dict[tensor.name] = tensor.to_dict()  # type: ignore[assignment]

                import json

                json_bytes = json.dumps(header_dict, separators=(",", ":")).encode(
                    "utf-8"
                )
                padding_needed = self._header_size - 8 - len(json_bytes)
                padded_json = json_bytes + b" " * padding_needed

                assert self._file is not None
                pwrite(self._file.fileno(), padded_json, 8)

            # Close file atomically
            self._file_writer.close()
            self._file_writer = None
            self._file = None

            logger.info(f"Successfully wrote {self.path}")

    def abort(self) -> None:
        """Abort writing and clean up temporary file."""
        with self._lock:
            if self._file_writer is not None:
                self._file_writer.abort()
                self._file_writer = None
                self._file = None

    def stats(self) -> WriterStats:
        """Get current writer statistics."""
        with self._lock:
            elapsed = time.time() - self._start_time
            mb_written = self._total_written / (1024 * 1024)
            mb_per_s = mb_written / elapsed if elapsed > 0 else 0

            remaining = self._total_size - self._total_written
            eta_s = remaining / (mb_per_s * 1024 * 1024) if mb_per_s > 0 else 0

            # Get RSS in MB
            try:
                import psutil

                rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except ImportError:
                rss_mb = 0

            return WriterStats(
                written=self._total_written,
                total=self._total_size,
                mb_per_s=mb_per_s,
                eta_s=eta_s,
                rss_mb=rss_mb,
            )

    def __enter__(self) -> StreamWriter:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.abort()
        else:
            self.close()
