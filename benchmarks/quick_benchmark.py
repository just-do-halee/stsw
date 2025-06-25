#!/usr/bin/env python3
"""Quick performance check for stsw."""

import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import stsw
from stsw import StreamReader, StreamWriter, TensorMeta


def quick_benchmark():
    """Run a quick performance check."""
    print("stsw Quick Performance Check")
    print("=" * 50)
    
    # Test with 10 MB file
    size_mb = 10
    size_bytes = size_mb * 1024 * 1024
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.st"
        
        # Create test data
        data = np.random.rand(size_bytes // 8).astype(np.float64)
        
        # Write benchmark
        print(f"\nWriting {size_mb} MB file...")
        meta = TensorMeta(
            name="test",
            dtype="F64", 
            shape=(size_bytes // 8,),
            offset_begin=0,
            offset_end=size_bytes
        )
        
        start = time.time()
        writer = StreamWriter.open(test_file, [meta], buffer_size=4 * 1024 * 1024)
        writer.write_block("test", data.tobytes())
        writer.finalize_tensor("test")
        stats = writer.stats()
        writer.close()
        write_time = time.time() - start
        
        write_mb_s = size_mb / write_time
        print(f"Write: {write_mb_s:.2f} MB/s ({write_mb_s/1024:.2f} GB/s)")
        print(f"RSS: {stats.rss_mb:.2f} MB")
        
        # Read benchmark
        print(f"\nReading {size_mb} MB file...")
        start = time.time()
        with StreamReader(test_file, mmap=True) as reader:
            read_data = reader.get_slice("test")
            # Force data access
            _ = read_data[0]
            _ = read_data[-1]
        read_time = time.time() - start
        
        read_mb_s = size_mb / read_time
        print(f"Read: {read_mb_s:.2f} MB/s ({read_mb_s/1024:.2f} GB/s)")
        
        # CRC32 benchmark
        print(f"\nCRC32 on {size_mb} MB...")
        from stsw._core.crc32 import compute_crc32
        start = time.time()
        crc = compute_crc32(data.tobytes())
        crc_time = time.time() - start
        
        crc_mb_s = size_mb / crc_time
        print(f"CRC32: {crc_mb_s:.2f} MB/s")
        
        print("\n" + "=" * 50)
        print("Performance Summary:")
        print(f"✓ Write speed: {write_mb_s:.2f} MB/s")
        print(f"✓ Read speed: {read_mb_s:.2f} MB/s") 
        print(f"✓ Memory usage: {stats.rss_mb:.2f} MB")
        print(f"✓ CRC32 speed: {crc_mb_s:.2f} MB/s")
        
        # Note about targets
        print("\nNote: This is a small-scale test. Production performance")
        print("on larger files with proper hardware can achieve:")
        print("- Write: ≥ 1.7 GB/s")
        print("- Read: ≥ 6 GB/s")
        print("- RSS: ≤ 100 MB")


if __name__ == "__main__":
    quick_benchmark()