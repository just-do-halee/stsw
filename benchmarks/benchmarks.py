"""ASV benchmarks for stsw."""

import tempfile
from pathlib import Path

import numpy as np

from stsw import StreamReader, StreamWriter, TensorMeta


class WriteBenchmarks:
    """Benchmarks for StreamWriter performance."""

    params = [
        [1, 10, 100, 1000],  # Size in MB
        [1, 4, 8],  # Buffer size in MB
    ]
    param_names = ["size_mb", "buffer_mb"]

    def setup(self, size_mb, buffer_mb):
        """Setup benchmark data."""
        self.tmpdir = tempfile.mkdtemp()
        self.output_path = Path(self.tmpdir) / "benchmark.st"

        # Create test data
        self.size_bytes = size_mb * 1024 * 1024
        self.buffer_size = buffer_mb * 1024 * 1024
        self.data = np.random.rand(self.size_bytes // 8).astype(np.float64)

        # Create metadata
        self.meta = TensorMeta(
            name="benchmark_tensor",
            dtype="F64",
            shape=(self.size_bytes // 8,),
            offset_begin=0,
            offset_end=self.size_bytes,
        )

    def teardown(self, size_mb, buffer_mb):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def time_write_throughput(self, size_mb, buffer_mb):
        """Benchmark write throughput."""
        writer = StreamWriter.open(
            self.output_path, [self.meta], buffer_size=self.buffer_size
        )

        # Write data
        writer.write_block("benchmark_tensor", self.data.tobytes())
        writer.finalize_tensor("benchmark_tensor")
        writer.close()

    def peakmem_write_rss(self, size_mb, buffer_mb):
        """Benchmark peak RSS memory during write."""
        writer = StreamWriter.open(
            self.output_path, [self.meta], buffer_size=self.buffer_size
        )

        # Write data in chunks
        chunk_size = min(self.buffer_size, self.size_bytes)
        data_bytes = self.data.tobytes()

        for i in range(0, self.size_bytes, chunk_size):
            chunk = data_bytes[i : i + chunk_size]
            writer.write_block("benchmark_tensor", chunk)

        writer.finalize_tensor("benchmark_tensor")
        writer.close()


class ReadBenchmarks:
    """Benchmarks for StreamReader performance."""

    params = [[1, 10, 100, 1000]]  # Size in MB
    param_names = ["size_mb"]

    def setup(self, size_mb):
        """Create test file for reading."""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "benchmark.st"

        # Create test data
        size_bytes = size_mb * 1024 * 1024
        data = np.random.rand(size_bytes // 8).astype(np.float64)

        # Write test file
        meta = TensorMeta(
            name="benchmark_tensor",
            dtype="F64",
            shape=(size_bytes // 8,),
            offset_begin=0,
            offset_end=size_bytes,
        )

        writer = StreamWriter.open(self.test_file, [meta])
        writer.write_block("benchmark_tensor", data.tobytes())
        writer.finalize_tensor("benchmark_tensor")
        writer.close()

    def teardown(self, size_mb):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def time_read_mmap(self, size_mb):
        """Benchmark mmap read performance."""
        with StreamReader(self.test_file, mmap=True) as reader:
            data = reader.get_slice("benchmark_tensor")
            # Force data access
            _ = data[0]

    def time_read_to_numpy(self, size_mb):
        """Benchmark reading to numpy array."""
        with StreamReader(self.test_file) as reader:
            arr = reader.to_numpy("benchmark_tensor")
            # Force computation
            _ = arr.sum()


class CRC32Benchmarks:
    """Benchmarks for CRC32 computation."""

    params = [[1, 10, 100]]  # Size in MB
    param_names = ["size_mb"]

    def setup(self, size_mb):
        """Setup benchmark data."""
        self.data = b"x" * (size_mb * 1024 * 1024)

    def time_crc32_compute(self, size_mb):
        """Benchmark CRC32 computation."""
        from stsw._core.crc32 import compute_crc32

        compute_crc32(self.data)

    def time_crc32_streaming(self, size_mb):
        """Benchmark streaming CRC32 computation."""
        from stsw._core.crc32 import StreamingCRC32

        crc = StreamingCRC32()
        chunk_size = 1024 * 1024  # 1MB chunks

        for i in range(0, len(self.data), chunk_size):
            crc.update(self.data[i : i + chunk_size])

        _ = crc.digest()
