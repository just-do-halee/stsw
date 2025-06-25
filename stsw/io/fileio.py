"""Cross-platform file I/O helpers with robust error handling."""

from __future__ import annotations

import atexit
import os
import platform
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, Union, cast

PathLike = Union[str, os.PathLike[str]]


class FileIOError(Exception):
    """Raised when file operations fail."""

    pass


class SafeFileWriter:
    """Atomic file writer with temporary file and cleanup."""

    def __init__(
        self,
        path: PathLike,
        mode: str = "wb",
        buffer_size: int = 4 * 1024 * 1024,  # 4 MB
    ) -> None:
        """Initialize atomic file writer.

        Args:
            path: Target file path
            mode: File open mode (must be binary write mode)
            buffer_size: I/O buffer size in bytes
        """
        if "b" not in mode or "r" in mode:
            raise ValueError("Mode must be binary write mode (e.g., 'wb')")

        self.path = Path(path).resolve()
        self.mode = mode
        self.buffer_size = buffer_size

        self.temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.file: BinaryIO | None = None
        self._closed = False
        self._aborted = False

        # Register cleanup handler
        atexit.register(self._cleanup)

    def open(self) -> BinaryIO:
        """Open the temporary file for writing.

        Returns:
            File handle

        Raises:
            FileIOError: If file cannot be opened
        """
        if self.file is not None:
            return self.file

        try:
            file = open(
                self.temp_path, self.mode, buffering=self.buffer_size
            )  # noqa: SIM115
            self.file = file  # type: ignore[assignment]
            return file  # type: ignore[return-value]
        except OSError as e:
            raise FileIOError(f"Failed to open temporary file: {e}") from e

    def write(self, data: bytes) -> int:
        """Write data to the temporary file.

        Args:
            data: Bytes to write

        Returns:
            Number of bytes written

        Raises:
            FileIOError: If write fails
        """
        if self.file is None:
            self.open()

        try:
            assert self.file is not None
            return self.file.write(data)
        except OSError as e:
            raise FileIOError(f"Failed to write data: {e}") from e

    def flush(self) -> None:
        """Flush file buffers to disk."""
        if self.file is not None:
            self.file.flush()

    def fsync(self) -> None:
        """Force write of file to disk (flush OS buffers)."""
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self) -> None:
        """Close and atomically rename temp file to target."""
        if self._closed or self._aborted:
            return

        self._closed = True

        if self.file is not None:
            self.fsync()
            self.file.close()
            self.file = None

        # Atomic rename
        try:
            # Windows doesn't support atomic rename if target exists
            if platform.system() == "Windows" and self.path.exists():
                self.path.unlink()
            self.temp_path.rename(self.path)
        except OSError as e:
            raise FileIOError(f"Failed to rename temp file: {e}") from e

    def abort(self) -> None:
        """Abort write and clean up temp file."""
        self._aborted = True

        if self.file is not None:
            self.file.close()
            self.file = None

        if self.temp_path.exists():
            with suppress(OSError):
                self.temp_path.unlink()

    def _cleanup(self) -> None:
        """Emergency cleanup on exit."""
        if not self._closed and not self._aborted:
            self.abort()

    def __enter__(self) -> SafeFileWriter:
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.abort()
        else:
            self.close()


def pwrite(fd: int, data: bytes, offset: int) -> int:
    """Write data at specific offset without changing file position.

    Args:
        fd: File descriptor
        data: Data to write
        offset: Offset in file

    Returns:
        Number of bytes written

    Raises:
        FileIOError: If pwrite fails
    """
    if hasattr(os, "pwrite"):
        # Unix-like systems
        try:
            return cast(Any, os).pwrite(fd, data, offset)
        except OSError as e:
            raise FileIOError(f"pwrite failed: {e}") from e
    else:
        # Windows fallback
        try:
            original_pos = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, offset, os.SEEK_SET)
            written = os.write(fd, data)
            os.lseek(fd, original_pos, os.SEEK_SET)
            return written
        except OSError as e:
            raise FileIOError(f"pwrite emulation failed: {e}") from e


def safe_seek(file: BinaryIO, offset: int, whence: int = os.SEEK_SET) -> int:
    """Safely seek in a file with error handling.

    Args:
        file: File handle
        offset: Seek offset
        whence: Seek mode (SEEK_SET, SEEK_CUR, SEEK_END)

    Returns:
        New file position

    Raises:
        FileIOError: If seek fails
    """
    try:
        return file.seek(offset, whence)
    except OSError as e:
        raise FileIOError(f"Seek failed: {e}") from e


def get_file_size(path: PathLike) -> int:
    """Get file size in bytes.

    Args:
        path: File path

    Returns:
        Size in bytes

    Raises:
        FileIOError: If stat fails
    """
    try:
        return Path(path).stat().st_size
    except OSError as e:
        raise FileIOError(f"Failed to get file size: {e}") from e


def ensure_parent_dir(path: PathLike) -> None:
    """Ensure parent directory exists.

    Args:
        path: File path

    Raises:
        FileIOError: If directory creation fails
    """
    parent = Path(path).parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise FileIOError(f"Failed to create parent directory: {e}") from e
