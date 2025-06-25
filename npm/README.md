# stsw - npm package

The Last-Word Safe-Tensor Stream Suite - CLI tools for streaming safetensors files.

This is the npm distribution of [stsw](https://github.com/just-do-halee/stsw), providing easy installation of the CLI tools via npm.

## Prerequisites

- Node.js 14+ and npm 6+
- Python 3.9+ with pip

## Installation

```bash
npm install -g stsw
```

This will:
1. Install the npm package
2. Automatically install the Python stsw package via pip

## Usage

After installation, the `stsw` command will be available globally:

```bash
# Show help
stsw --help

# Inspect a safetensors file
stsw inspect model.safetensors

# Verify checksums
stsw verify model.safetensors

# Convert PyTorch checkpoint
stsw convert model.pt model.safetensors --crc32

# Run self-test
stsw selftest
```

## Features

- 🚀 Stream multi-GB tensor files with <100 MB RAM
- 🔒 CRC32 verification for data integrity
- ⚡ Zero-copy memory-mapped reading
- 🛠️ Simple CLI interface
- 🌍 Cross-platform support (Linux, macOS, Windows)

## Documentation

Full documentation: https://github.com/just-do-halee/stsw

## Python Package

If you prefer to install via pip directly:

```bash
pip install stsw
```

## License

Apache-2.0