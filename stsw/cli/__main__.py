"""Command-line interface for stsw."""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stsw import __version__
from stsw._core.dtype import normalize
from stsw._core.meta import build_aligned_offsets
from stsw.reader.reader import StreamReader
from stsw.writer.writer import StreamWriter

if TYPE_CHECKING:
    from rich.console import Console  # pyright: ignore[reportUnusedImport]
    from rich.logging import RichHandler  # pyright: ignore[reportUnusedImport]
    from rich.table import Table  # pyright: ignore[reportUnusedImport]

logger = logging.getLogger("stsw")


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO

    # Check for rich
    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=verbose,
        )
    except ImportError:
        handler = logging.StreamHandler()

    logging.basicConfig(
        level=level,
        format=(
            "%(message)s"
            if handler.__class__.__name__ == "RichHandler"
            else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        handlers=[handler],
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a safetensors file."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        use_rich = True
    except ImportError:
        console: Any = None
        Table: Any = None  # noqa: N806
        use_rich = False

    try:
        with StreamReader(args.file, mmap=True) as reader:
            if use_rich:
                assert console is not None
                assert Table is not None  # for type checker
                table = Table(title=f"Tensors in {args.file}")
                table.add_column("Name", style="cyan")
                table.add_column("Shape", style="green")
                table.add_column("Dtype", style="yellow")
                table.add_column("Size (MB)", justify="right", style="magenta")
                table.add_column("CRC32", style="blue")

                total_size = 0
                for name in reader:
                    meta = reader.meta(name)
                    size_mb = meta.nbytes / (1024 * 1024)
                    total_size += meta.nbytes

                    table.add_row(
                        name,
                        str(meta.shape),
                        meta.dtype,
                        f"{size_mb:.2f}",
                        str(meta.crc32) if meta.crc32 is not None else "None",
                    )

                console.print(table)
                console.print(f"\nTotal size: {total_size / (1024 * 1024):.2f} MB")
                console.print(f"Format version: {reader.version}")

                if reader.metadata:
                    console.print("\nMetadata:")
                    console.print(reader.metadata)
            else:
                # Plain text output
                print(f"Tensors in {args.file}:")
                print("-" * 80)
                print(
                    f"{'Name':<30} {'Shape':<20} {'Dtype':<10} {'Size (MB)':<12} {'CRC32':<10}"
                )
                print("-" * 80)

                total_size = 0
                for name in reader:
                    meta = reader.meta(name)
                    size_mb = meta.nbytes / (1024 * 1024)
                    total_size += meta.nbytes

                    print(
                        f"{name:<30} {meta.shape!s:<20} {meta.dtype:<10} {size_mb:<12.2f} {meta.crc32 or 'None':<10}"
                    )

                print("-" * 80)
                print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
                print(f"Format version: {reader.version}")

                if reader.metadata:
                    print("\nMetadata:")
                    print(json.dumps(reader.metadata, indent=2))

        return 0
    except Exception as e:
        logger.error(f"Failed to inspect file: {e}")
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify CRC32 checksums in a safetensors file."""
    try:
        with StreamReader(args.file, mmap=True, verify_crc=True) as reader:
            print(f"Verifying {len(reader)} tensors in {args.file}...")

            errors = 0
            for i, name in enumerate(reader.keys()):
                meta = reader.meta(name)
                if meta.crc32 is None:
                    print(f"[{i+1}/{len(reader)}] {name}: No CRC32 stored")
                    continue

                try:
                    # This will trigger CRC verification
                    _ = reader.get_slice(name)
                    print(f"[{i+1}/{len(reader)}] {name}: OK")
                except ValueError as e:
                    print(f"[{i+1}/{len(reader)}] {name}: FAILED - {e}")
                    errors += 1

            if errors > 0:
                print(f"\n{errors} CRC verification failures!")
                return 1
            else:
                print("\nAll CRC checksums verified successfully!")
                return 0

    except Exception as e:
        logger.error(f"Failed to verify file: {e}")
        return 1


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert a PyTorch checkpoint to safetensors format."""
    try:
        import torch
    except ImportError:
        logger.error(
            "PyTorch is required for conversion. Install with: pip install stsw[torch]"
        )
        return 1

    print(f"Loading {args.input}...")
    state_dict = torch.load(args.input, map_location="cpu", weights_only=True)

    if not isinstance(state_dict, dict):
        logger.error("Input file must contain a state dict")
        return 1

    # Build tensor metadata
    print("Analyzing tensors...")
    tensor_info = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            logger.warning(f"Skipping non-tensor '{name}': {type(tensor)}")
            continue

        dtype = normalize(tensor.dtype)
        shape = tuple(tensor.shape)
        nbytes = tensor.numel() * tensor.element_size()

        tensor_info.append((name, dtype, shape, nbytes))

    # Build aligned offsets
    metas = build_aligned_offsets(tensor_info, align=64)

    # Write file
    print(f"Writing {len(metas)} tensors to {args.output}...")
    start_time = time.time()

    with StreamWriter.open(
        args.output,
        metas,
        crc32=args.crc32,
        buffer_size=args.buffer_size * 1024 * 1024,
    ) as writer:
        for i, (name, tensor) in enumerate(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                continue

            # Convert to contiguous if needed
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()

            # Write in chunks
            data = tensor.detach().cpu().numpy().tobytes()
            chunk_size = args.buffer_size * 1024 * 1024

            for offset in range(0, len(data), chunk_size):
                chunk = data[offset : offset + chunk_size]
                writer.write_block(name, chunk)

            writer.finalize_tensor(name)

            # Progress
            print(f"[{i+1}/{len(metas)}] {name}")

    elapsed = time.time() - start_time
    total_mb = sum(m.nbytes for m in metas) / (1024 * 1024)
    print(f"\nConversion complete in {elapsed:.1f}s ({total_mb/elapsed:.1f} MB/s)")

    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run self-test to verify installation."""
    print("Running stsw self-test...")
    print(f"Version: {__version__}")

    try:
        import tempfile

        import numpy as np

        # Create test data
        print("\n1. Creating test data...")
        test_data = {
            "tensor1": np.random.rand(1000, 1000).astype(np.float32),
            "tensor2": np.random.randint(-128, 127, size=(500, 500, 3), dtype=np.int8),
        }

        # Build metadata
        tensor_info = []
        for name, arr in test_data.items():
            dtype = normalize(arr.dtype)
            shape = tuple(arr.shape)
            nbytes = arr.nbytes
            tensor_info.append((name, dtype, shape, nbytes))

        metas = build_aligned_offsets(tensor_info, align=64)

        # Write test file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name

        print(f"\n2. Writing test file to {tmp_path}...")
        with StreamWriter.open(tmp_path, metas, crc32=True) as writer:
            for name, arr in test_data.items():
                writer.write_block(name, arr.tobytes())
                writer.finalize_tensor(name)

        # Read back and verify
        print("\n3. Reading and verifying...")
        with StreamReader(tmp_path, mmap=True, verify_crc=True) as reader:
            for name, original in test_data.items():
                loaded = reader.to_numpy(name)
                if not np.array_equal(original, loaded):
                    raise ValueError(f"Data mismatch for {name}")
                print(f"   {name}: OK")

        # Clean up
        os.unlink(tmp_path)

        print("\nSelf-test PASSED! ✓")
        return 0

    except Exception as e:
        logger.error(f"Self-test FAILED: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    try:
        parser = argparse.ArgumentParser(
            prog="stsw",
            description="Safe-Tensor Stream Suite - The last-word streaming safetensors implementation",
        )
        parser.add_argument(
            "--version", action="version", version=f"stsw {__version__}"
        )
        parser.add_argument(
            "-v", "--verbose", action="store_true", help="Enable verbose logging"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Inspect command
        inspect_parser = subparsers.add_parser(
            "inspect", help="Inspect a safetensors file"
        )
        inspect_parser.add_argument("file", type=Path, help="Path to safetensors file")

        # Verify command
        verify_parser = subparsers.add_parser("verify", help="Verify CRC32 checksums")
        verify_parser.add_argument("file", type=Path, help="Path to safetensors file")

        # Convert command
        convert_parser = subparsers.add_parser(
            "convert", help="Convert PyTorch checkpoint to safetensors"
        )
        convert_parser.add_argument("input", type=Path, help="Input checkpoint file")
        convert_parser.add_argument("output", type=Path, help="Output safetensors file")
        convert_parser.add_argument(
            "--crc32", action="store_true", help="Compute CRC32 checksums"
        )
        convert_parser.add_argument(
            "--buffer-size", type=int, default=8, help="Buffer size in MB (default: 8)"
        )

        # Self-test command
        subparsers.add_parser("selftest", help="Run self-test")

        args = parser.parse_args()

        # Setup logging
        setup_logging(args.verbose)

        # Dispatch command
        if args.command == "inspect":
            return cmd_inspect(args)
        elif args.command == "verify":
            return cmd_verify(args)
        elif args.command == "convert":
            return cmd_convert(args)
        elif args.command == "selftest":
            return cmd_selftest(args)
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        return 130  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)  # Standard exit code for Ctrl+C
