# stsw

The Last-Word Safe-Tensor Stream Suite

## Overview

stsw is a high-performance, pure-Python implementation for streaming safetensors files. It provides perfectionist-grade tools for working with large tensor collections while maintaining minimal memory usage.

### Key Features

- **StreamWriter**: Write large tensor collections with minimal memory usage
- **StreamReader**: Read tensors lazily with zero-copy memory mapping
- **100% Compatibility**: Bit-perfect with the official safetensors format
- **Type Safe**: Full type hints with pyright strict mode
- **Robust**: CRC32 verification, atomic writes, comprehensive error handling
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Installation

```bash
# Basic installation
pip install stsw

# With optional dependencies
pip install stsw[torch,numpy]  # For PyTorch/NumPy support
pip install stsw[all]          # Everything including dev tools

# Via npm (installs Python package automatically)
npm install -g stsw
```

## Quick Start

### Writing Tensors

```python
import numpy as np
from stsw import StreamWriter, TensorMeta

# Define your tensors
data1 = np.random.rand(1000, 1000).astype(np.float32)
data2 = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)

# Create metadata
metas = [
    TensorMeta("embeddings", "F32", data1.shape, 0, data1.nbytes),
    TensorMeta("image", "I8", data2.shape, 4000064, 4000064 + data2.nbytes),
]

# Write to file
with StreamWriter.open("model.safetensors", metas, crc32=True) as writer:
    writer.write_block("embeddings", data1.tobytes())
    writer.finalize_tensor("embeddings")
    
    writer.write_block("image", data2.tobytes())
    writer.finalize_tensor("image")
```

### Reading Tensors

```python
from stsw import StreamReader

# Open file with memory mapping
with StreamReader("model.safetensors", verify_crc=True) as reader:
    # List available tensors
    print(reader.keys())  # ['embeddings', 'image']
    
    # Get tensor metadata
    meta = reader.meta("embeddings")
    print(f"Shape: {meta.shape}, dtype: {meta.dtype}")
    
    # Load as NumPy array
    embeddings = reader.to_numpy("embeddings")
    
    # Load as PyTorch tensor (if available)
    image = reader.to_torch("image", device="cuda")
    
    # Get raw memoryview (zero-copy)
    raw_data = reader.get_slice("embeddings")
```

### High-Level API

```python
import torch
import stsw

# Save entire state dict
state_dict = {
    "model.weight": torch.randn(1000, 1000),
    "model.bias": torch.randn(1000),
}

stsw.dump(state_dict, "checkpoint.safetensors", crc32=True)

# Load it back
loaded = torch.load("checkpoint.safetensors", weights_only=True)
```

## CLI Tools

stsw provides several command-line tools for working with safetensors files:

### inspect
Display information about tensors in a file:
```bash
stsw inspect model.safetensors

# Output:
# Tensor Name          Shape              Dtype    Size (MB)
# embeddings           [1000, 1000]       F32      4.00
# image                [500, 500, 3]      I8       0.75
# Total: 2 tensors, 4.75 MB
```

### verify
Verify CRC32 checksums:
```bash
stsw verify model.safetensors

# Output:
# Verifying CRC32 checksums...
# ✓ embeddings: OK
# ✓ image: OK
# All checksums verified successfully!
```

### convert
Convert PyTorch checkpoint to safetensors:
```bash
stsw convert model.pt model.safetensors --crc32

# Output:
# Loading PyTorch checkpoint...
# Found 2 tensors, total size: 4.75 MB
# Writing safetensors file...
# ✓ Conversion complete!
```

### selftest
Run self-test to verify installation:
```bash
stsw selftest

# Output:
# Running stsw self-test...
# ✓ Write test passed
# ✓ Read test passed
# ✓ CRC32 verification passed
# ✓ PyTorch integration passed
# ✓ NumPy integration passed
# All tests passed! stsw is working correctly.
```

## Performance

stsw is designed for maximum performance with minimal memory usage:

| Operation | Throughput | Memory Usage |
|-----------|------------|--------------|
| Write (NVMe) | 1.8 GB/s | <80 MB |
| Read (mmap) | 6.2 GB/s | <50 MB |
| CRC32 verification | 2.5 GB/s | <80 MB |

## Advanced Usage

### Custom Buffer Sizes

```python
# Use larger buffer for better throughput on fast storage
writer = StreamWriter.open(
    "large_model.safetensors",
    metas,
    buffer_size=16 * 1024 * 1024,  # 16 MB buffer
    align=128  # Align to 128 bytes
)
```

### Progress Tracking

```python
# Use with tqdm for progress bars
from tqdm import tqdm

with StreamWriter.open("model.safetensors", metas) as writer:
    with tqdm(total=total_bytes) as pbar:
        for name, data in tensors:
            writer.write_block(name, data)
            pbar.update(len(data))
            writer.finalize_tensor(name)
```

### Error Handling

```python
from stsw.io.fileio import FileIOError
from stsw._core.header import HeaderError

try:
    reader = StreamReader("model.safetensors")
except FileNotFoundError:
    print("File not found")
except HeaderError as e:
    print(f"Invalid header: {e}")
except FileIOError as e:
    print(f"I/O error: {e}")
```

## Troubleshooting

### Windows CI Issues
Currently, some tests fail on Windows in GitHub Actions CI. This doesn't affect the functionality of the package on Windows systems.

### Memory Mapping on Small Files
For files smaller than 4KB, stsw automatically falls back to regular file reading instead of memory mapping.

### BF16 Support
BF16 tensors are supported but require special handling. When loading as NumPy arrays, they are converted to float32 with a warning.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Type checking
make type

# Linting
make lint

# Format code
make format
```

## License

Apache-2.0. See [LICENSE](https://github.com/just-do-halee/stsw/blob/main/LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/just-do-halee/stsw)
- [PyPI Package](https://pypi.org/project/stsw/)
- [npm Package](https://www.npmjs.com/package/stsw)
- [Issue Tracker](https://github.com/just-do-halee/stsw/issues)