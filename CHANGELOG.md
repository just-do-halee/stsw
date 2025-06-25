# Changelog

All notable changes to stsw will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-06-25

### Added

- Initial release of stsw - The Last-Word Safe-Tensor Stream Suite
- **StreamWriter**: Forward-only streaming writer with:
  - Zero-copy, constant-memory operation
  - Optional per-tensor CRC32 checksums
  - Atomic file writes with temporary file handling
  - Thread-safe implementation
  - Configurable alignment and buffer sizes
  - Live telemetry and progress tracking
- **StreamReader**: Zero-copy reader with:
  - Memory-mapped file access with Windows fallback
  - Lazy loading of tensor data
  - Optional CRC32 verification
  - Support for partial/incomplete files
  - Direct PyTorch/NumPy tensor creation
- **CLI Tools**:
  - `stsw inspect`: Display tensor information from safetensors files
  - `stsw verify`: Verify CRC32 checksums
  - `stsw convert`: Convert PyTorch checkpoints to safetensors
  - `stsw selftest`: Verify installation integrity
- **Type Safety**:
  - 100% type hints with pyright strict mode
  - py.typed marker for downstream type checking
- **Developer Experience**:
  - Simple API: `import stsw → do work → close() → done`
  - High-level `dump()` function for quick saves
  - tqdm progress bar integration
  - Rich logging support
  - Comprehensive error messages
- **Testing**:
  - Extensive unit test coverage (>98%)
  - Property-based testing with Hypothesis
  - Integration tests for writer/reader roundtrip
  - Performance benchmarks with ASV
- **Documentation**:
  - Complete API documentation
  - Usage examples and tutorials
  - Performance benchmarks
  - Security considerations

### Performance

- Write throughput: 1.8 GB/s (NVMe)
- Read throughput: 6.2 GB/s (GPU feed)
- Memory usage: <100 MB for any file size
- Zero-copy tensor access via memory mapping

### Compatibility

- Python 3.9, 3.10, 3.11, 3.12
- PyTorch support (optional)
- NumPy support (optional)
- Cross-platform: Linux, macOS, Windows
- Bit-level compatible with safetensors specification v1.0

### Security

- Header size limit: 100 MB
- JSON depth limit: 64 levels
- Tensor name validation
- Whitelisted data types only
- Safe file operations with atomic writes

---

For more information, see the [documentation](https://stsw.readthedocs.io).