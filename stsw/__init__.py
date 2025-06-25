"""stsw - The Last-Word Safe-Tensor Stream Suite.

Perfectionist-grade Stream Writer & Stream Reader, designed once
so no-one ever has to rewrite them.
"""

from __future__ import annotations

import os
from typing import Any

from stsw._core import dtype
from stsw._core.header import HeaderError
from stsw._core.meta import TensorMeta
from stsw.io.fileio import FileIOError
from stsw.reader.reader import StreamReader
from stsw.writer.writer import (
    LengthMismatchError,
    StreamWriter,
    TensorOrderError,
    WriterStats,
)

__version__ = "1.0.0"

# Public constants
DEFAULT_ALIGN = 64


def dump(
    state_dict: dict[str, Any],
    path: str | os.PathLike[str],
    *,
    workers: int = 1,
    crc32: bool = False,
    buffer_size: int = 4 << 20,  # 4 MiB to match StreamWriter default
) -> None:
    """High-level function to dump a state dict to safetensors format.

    Args:
        state_dict: Dictionary of tensors (torch or numpy)
        path: Output file path
        workers: Number of worker threads (not implemented yet)
        crc32: Whether to compute CRC32 checksums
        buffer_size: I/O buffer size in bytes
    """
    from stsw._core.meta import build_aligned_offsets

    # Build tensor metadata
    tensor_info = []
    for name, tensor in state_dict.items():
        if hasattr(tensor, "numpy"):  # PyTorch tensor
            tensor_np = tensor.detach().cpu().numpy()
            dtype_str = dtype.normalize(tensor.dtype)
        elif hasattr(tensor, "shape"):  # NumPy array
            tensor_np = tensor
            dtype_str = dtype.normalize(tensor.dtype)
        else:
            raise TypeError(f"Unsupported tensor type for '{name}': {type(tensor)}")

        shape = tuple(tensor_np.shape)
        nbytes = tensor_np.nbytes

        tensor_info.append((name, dtype_str, shape, nbytes))

    # Build aligned metadata
    metas = build_aligned_offsets(tensor_info, align=DEFAULT_ALIGN)

    # Write file
    with StreamWriter.open(path, metas, crc32=crc32, buffer_size=buffer_size) as writer:
        for name, tensor in state_dict.items():
            if hasattr(tensor, "numpy"):  # PyTorch tensor
                data = tensor.detach().cpu().numpy().tobytes()
            else:  # NumPy array
                data = tensor.tobytes()

            # Write in chunks
            chunk_size = buffer_size
            for offset in range(0, len(data), chunk_size):
                chunk = data[offset : offset + chunk_size]
                writer.write_block(name, chunk)

            writer.finalize_tensor(name)


# Helper for tqdm integration
class tqdm:
    """Namespace for tqdm integration helpers."""

    @staticmethod
    def wrap(writer: StreamWriter) -> StreamWriter:
        """Wrap a StreamWriter with tqdm progress bar.

        Args:
            writer: StreamWriter instance

        Returns:
            Writer with progress bar support
        """
        try:
            from tqdm.auto import tqdm as tqdm_bar

            class TqdmWriter:
                def __init__(self, wrapped: StreamWriter) -> None:
                    self._wrapped = wrapped
                    self._pbar = tqdm_bar(
                        total=wrapped._total_size,  # pyright: ignore[reportPrivateUsage]
                        unit="B",
                        unit_scale=True,
                        desc="Writing",
                    )

                def __getattr__(self, name: str) -> Any:
                    return getattr(self._wrapped, name)

                def write_block(
                    self, name: str, data: bytes | memoryview
                ) -> None:
                    result = self._wrapped.write_block(name, data)
                    self._pbar.update(len(data))
                    return result

                def close(self) -> None:
                    self._pbar.close()
                    return self._wrapped.close()

                def abort(self) -> None:
                    self._pbar.close()
                    return self._wrapped.abort()

            return TqdmWriter(writer)  # pyright: ignore[reportReturnType]

        except ImportError:
            # No tqdm available, return unwrapped
            return writer


# All public exports
__all__ = [
    # Version
    "__version__",
    # Constants
    "DEFAULT_ALIGN",
    # Core types
    "TensorMeta",
    # Main classes
    "StreamReader",
    "StreamWriter",
    "WriterStats",
    # Errors
    "HeaderError",
    "FileIOError",
    "TensorOrderError",
    "LengthMismatchError",
    # Utilities
    "dtype",
    "dump",
    "tqdm",
]
