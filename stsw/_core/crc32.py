"""Streaming CRC32 implementation with xxhash fallback."""

from __future__ import annotations

import zlib

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xxhash

_xxhash_available = False
try:
    import xxhash

    _xxhash_available = True
except ImportError:
    xxhash = None


CRC32_POLY = 0x1EDC6F41  # Castagnoli polynomial


class StreamingCRC32:
    """Streaming CRC32-C (Castagnoli) calculator.

    Uses zlib.crc32 for performance, with xxhash fallback if available.
    """

    def __init__(self) -> None:
        """Initialize a new CRC32 calculator."""
        self._crc: int = 0
        self._use_xxhash = _xxhash_available

        if self._use_xxhash:
            assert xxhash is not None  # for type checker
            self._hasher = xxhash.xxh32(seed=0)

    def update(self, data: bytes | memoryview) -> None:
        """Update the CRC32 with new data.

        Args:
            data: Bytes to add to the checksum
        """
        if self._use_xxhash:
            self._hasher.update(data)
            if len(data) > 0:
                self._has_data = True
        else:
            if isinstance(data, memoryview):
                data = bytes(data)
            self._crc = zlib.crc32(data, self._crc)

    def digest(self) -> int:
        """Get the current CRC32 value.

        Returns:
            32-bit unsigned integer checksum
        """
        if self._use_xxhash:
            # XXHash has a different initial state, but we want CRC32 of empty data to be 0
            if not hasattr(self, '_has_data'):
                return 0
            return self._hasher.intdigest() & 0xFFFFFFFF
        else:
            return self._crc & 0xFFFFFFFF

    def reset(self) -> None:
        """Reset the CRC32 calculator to initial state."""
        self._crc = 0
        if self._use_xxhash:
            self._hasher.reset()
            if hasattr(self, '_has_data'):
                delattr(self, '_has_data')

    def copy(self) -> "StreamingCRC32":
        """Create a copy of the current CRC32 state.

        Returns:
            New StreamingCRC32 instance with same state
        """
        new_crc = StreamingCRC32()
        new_crc._crc = self._crc
        new_crc._use_xxhash = self._use_xxhash

        if self._use_xxhash:
            new_crc._hasher = self._hasher.copy()
            if hasattr(self, '_has_data'):
                new_crc._has_data = True

        return new_crc


def compute_crc32(data: bytes | memoryview) -> int:
    """Compute CRC32-C checksum of data in one shot.

    Args:
        data: Data to checksum

    Returns:
        32-bit unsigned integer checksum
    """
    crc = StreamingCRC32()
    crc.update(data)
    return crc.digest()


def verify_crc32(data: bytes | memoryview, expected: int) -> bool:
    """Verify data against expected CRC32 checksum.

    Args:
        data: Data to verify
        expected: Expected CRC32 value

    Returns:
        True if checksum matches
    """
    return compute_crc32(data) == expected
