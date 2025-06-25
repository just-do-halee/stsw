#!/usr/bin/env python3
"""Run performance benchmarks for stsw."""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stsw import StreamReader, StreamWriter, TensorMeta


def benchmark_write_throughput(size_gb=1):
    """Benchmark write throughput."""
    print(f"\n=== Write Throughput Benchmark ({size_gb} GB) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "benchmark.st"

        # Create test data
        size_bytes = size_gb * 1024 * 1024 * 1024
        num_elements = size_bytes // 4  # float32

        # Create metadata
        meta = TensorMeta(
            name="benchmark_tensor",
            dtype="F32",
            shape=(num_elements,),
            offset_begin=0,
            offset_end=size_bytes,
        )

        # Write in chunks to avoid memory issues
        chunk_size = 100 * 1024 * 1024  # 100 MB chunks
        chunks_written = 0

        start_time = time.time()

        writer = StreamWriter.open(output_path, [meta], buffer_size=8 * 1024 * 1024)

        while chunks_written < size_bytes:
            chunk_elements = min(chunk_size // 4, num_elements - (chunks_written // 4))
            chunk_data = np.random.rand(int(chunk_elements)).astype(np.float32)
            writer.write_block("benchmark_tensor", chunk_data.tobytes())
            chunks_written += chunk_elements * 4

        writer.finalize_tensor("benchmark_tensor")

        # Get stats before closing
        stats = writer.stats()
        peak_rss = stats.rss_mb

        writer.close()

        elapsed = time.time() - start_time
        throughput_mb = (size_bytes / (1024 * 1024)) / elapsed
        throughput_gb = throughput_mb / 1024

        print(f"Written: {size_gb} GB")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput_mb:.2f} MB/s ({throughput_gb:.2f} GB/s)")
        print(f"Peak RSS: {peak_rss:.2f} MB")

        return throughput_gb, peak_rss


def benchmark_read_throughput(size_gb=1):
    """Benchmark read throughput."""
    print(f"\n=== Read Throughput Benchmark ({size_gb} GB) ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "benchmark.st"

        # First create the file
        size_bytes = size_gb * 1024 * 1024 * 1024
        num_elements = size_bytes // 4  # float32

        meta = TensorMeta(
            name="benchmark_tensor",
            dtype="F32",
            shape=(num_elements,),
            offset_begin=0,
            offset_end=size_bytes,
        )

        # Write test file
        writer = StreamWriter.open(test_file, [meta])

        chunk_size = 100 * 1024 * 1024  # 100 MB chunks
        chunks_written = 0

        while chunks_written < size_bytes:
            chunk_elements = min(chunk_size // 4, num_elements - (chunks_written // 4))
            chunk_data = np.zeros(int(chunk_elements), dtype=np.float32)
            writer.write_block("benchmark_tensor", chunk_data.tobytes())
            chunks_written += chunk_elements * 4

        writer.finalize_tensor("benchmark_tensor")
        writer.close()

        # Now benchmark reading
        start_time = time.time()

        with StreamReader(test_file, mmap=True) as reader:
            data = reader.get_slice("benchmark_tensor")
            # Force data access by reading some values
            for i in range(0, len(data), 1024 * 1024):
                _ = data[i]

        elapsed = time.time() - start_time
        throughput_mb = (size_bytes / (1024 * 1024)) / elapsed
        throughput_gb = throughput_mb / 1024

        print(f"Read: {size_gb} GB")
        print(f"Time: {elapsed:.2f} seconds")
        print(f"Throughput: {throughput_mb:.2f} MB/s ({throughput_gb:.2f} GB/s)")

        return throughput_gb


def benchmark_crc32_speed():
    """Benchmark CRC32 computation speed."""
    print("\n=== CRC32 Benchmark ===")

    from stsw._core.crc32 import compute_crc32

    # Test different sizes
    sizes = [1, 10, 100]  # MB

    for size_mb in sizes:
        data = b"x" * (size_mb * 1024 * 1024)

        start_time = time.time()
        crc = compute_crc32(data)
        elapsed = time.time() - start_time

        throughput = size_mb / elapsed
        print(f"{size_mb} MB: {elapsed:.3f}s ({throughput:.2f} MB/s)")


def main():
    """Run all benchmarks."""
    print("stsw Performance Benchmarks")
    print("=" * 50)

    # Run write benchmark
    write_throughput, peak_rss = benchmark_write_throughput(
        0.1
    )  # 100 MB for quick test

    # Run read benchmark
    read_throughput = benchmark_read_throughput(0.1)  # 100 MB for quick test

    # Run CRC32 benchmark
    benchmark_crc32_speed()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Write throughput: {write_throughput:.2f} GB/s")
    print(f"Write peak RSS: {peak_rss:.2f} MB")
    print(f"Read throughput: {read_throughput:.2f} GB/s")

    # Check against targets
    print("\nPerformance Targets:")
    print(
        f"✓ Write throughput ≥ 1.7 GB/s: {'PASS' if write_throughput >= 1.7 else 'FAIL'}"
    )
    print(f"✓ Write RSS ≤ 100 MB: {'PASS' if peak_rss <= 100 else 'FAIL'}")
    print(f"✓ Read throughput ≥ 6 GB/s: {'PASS' if read_throughput >= 6 else 'FAIL'}")


if __name__ == "__main__":
    main()
