# stsw

The Last-Word Safe-Tensor Stream Suite

## Overview

stsw is a high-performance, pure-Python implementation for streaming safetensors files. It provides:

- **StreamWriter**: Write large tensor collections with minimal memory usage
- **StreamReader**: Read tensors lazily with zero-copy memory mapping
- **100% Compatibility**: Bit-perfect with the official safetensors format

## Installation

```bash
pip install stsw
```

## Quick Example

```python
import stsw

# Write tensors
writer = stsw.StreamWriter.open('model.st', metadata)
writer.write_block('weights', tensor_data)
writer.close()

# Read tensors
with stsw.StreamReader('model.st') as reader:
    tensor = reader.to_torch('weights')
```
