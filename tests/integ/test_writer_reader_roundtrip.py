"""Integration tests for StreamWriter and StreamReader roundtrip."""

import numpy as np
import pytest

from stsw import StreamReader, StreamWriter, TensorMeta, dtype


class TestWriterReaderRoundtrip:
    """Test writing and reading back data."""

    def test_simple_roundtrip(self, tmp_path):
        """Test simple write and read roundtrip."""
        # Create test data
        data1 = np.random.rand(10, 20).astype(np.float32)
        data2 = np.random.randint(0, 100, size=(5, 5), dtype=np.int32)

        # Build metadata
        metas = [
            TensorMeta("tensor1", "F32", data1.shape, 0, data1.nbytes),
            TensorMeta("tensor2", "I32", data2.shape, 64, 64 + data2.nbytes),
        ]

        file_path = tmp_path / "test.safetensors"

        # Write
        with StreamWriter.open(file_path, metas) as writer:
            writer.write_block("tensor1", data1.tobytes())
            writer.finalize_tensor("tensor1")

            writer.write_block("tensor2", data2.tobytes())
            writer.finalize_tensor("tensor2")

        # Read back
        with StreamReader(file_path) as reader:
            loaded1 = reader.to_numpy("tensor1")
            loaded2 = reader.to_numpy("tensor2")

            np.testing.assert_array_equal(data1, loaded1)
            np.testing.assert_array_equal(data2, loaded2)

    def test_roundtrip_with_crc(self, tmp_path):
        """Test roundtrip with CRC32 verification."""
        data = np.arange(1000, dtype=np.float64).reshape(10, 100)

        meta = TensorMeta("test", "F64", data.shape, 0, data.nbytes)
        file_path = tmp_path / "test_crc.safetensors"

        # Write with CRC
        with StreamWriter.open(file_path, [meta], crc32=True) as writer:
            writer.write_block("test", data.tobytes())
            writer.finalize_tensor("test")

        # Read with CRC verification
        with StreamReader(file_path, verify_crc=True) as reader:
            loaded = reader.to_numpy("test")
            np.testing.assert_array_equal(data, loaded)

    def test_multiple_dtypes(self, tmp_path):
        """Test roundtrip with multiple data types."""
        test_data = {
            "float16": np.random.rand(5, 5).astype(np.float16),
            "float32": np.random.rand(10, 10).astype(np.float32),
            "float64": np.random.rand(3, 3).astype(np.float64),
            "int8": np.random.randint(-128, 127, size=(20, 20), dtype=np.int8),
            "int16": np.random.randint(-1000, 1000, size=(15, 15), dtype=np.int16),
            "int32": np.random.randint(-10000, 10000, size=(8, 8), dtype=np.int32),
            "int64": np.random.randint(-100000, 100000, size=(4, 4), dtype=np.int64),
        }

        # Build metadata
        metas = []
        offset = 0
        for name, arr in test_data.items():
            st_dtype = dtype.normalize(arr.dtype)
            meta = TensorMeta(name, st_dtype, arr.shape, offset, offset + arr.nbytes)
            metas.append(meta)
            # Align to 64
            offset += arr.nbytes
            if offset % 64 != 0:
                offset = ((offset + 63) // 64) * 64

        file_path = tmp_path / "multi_dtype.safetensors"

        # Write
        with StreamWriter.open(file_path, metas) as writer:
            for name, arr in test_data.items():
                writer.write_block(name, arr.tobytes())
                writer.finalize_tensor(name)

        # Read back
        with StreamReader(file_path) as reader:
            for name, original in test_data.items():
                loaded = reader.to_numpy(name)
                np.testing.assert_array_equal(original, loaded)
                assert loaded.dtype == original.dtype

    def test_chunked_writing(self, tmp_path):
        """Test writing data in chunks."""
        # Large array (10 MB)
        data = np.random.rand(1000, 1250).astype(np.float64)  # 10 MB
        chunk_size = 1024 * 1024  # 1 MB chunks

        meta = TensorMeta("large", "F64", data.shape, 0, data.nbytes)
        file_path = tmp_path / "chunked.safetensors"

        # Write in chunks
        with StreamWriter.open(file_path, [meta]) as writer:
            data_bytes = data.tobytes()
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i : i + chunk_size]
                writer.write_block("large", chunk)
            writer.finalize_tensor("large")

        # Read back
        with StreamReader(file_path) as reader:
            loaded = reader.to_numpy("large")
            np.testing.assert_array_equal(data, loaded)

    def test_zero_size_tensor(self, tmp_path):
        """Test handling zero-size tensors."""
        # Empty tensor
        data = np.array([], dtype=np.float32)

        meta = TensorMeta("empty", "F32", data.shape, 0, 0)
        file_path = tmp_path / "empty.safetensors"

        # Write
        with StreamWriter.open(file_path, [meta]) as writer:
            writer.finalize_tensor("empty")  # No data to write

        # Read back
        with StreamReader(file_path) as reader:
            loaded = reader.to_numpy("empty")
            np.testing.assert_array_equal(data, loaded)

    @pytest.mark.skipif(
        not pytest.importorskip("torch"), reason="PyTorch not installed"
    )
    def test_torch_roundtrip(self, tmp_path):
        """Test roundtrip with PyTorch tensors."""
        import torch

        # Create PyTorch tensors
        tensor1 = torch.randn(10, 20, dtype=torch.float32)
        tensor2 = torch.randint(0, 255, (5, 5, 3), dtype=torch.uint8)
        tensor3 = torch.randn(100, dtype=torch.bfloat16)

        # Build metadata
        metas = []
        offset = 0
        tensors = {"tensor1": tensor1, "tensor2": tensor2, "tensor3": tensor3}

        for name, t in tensors.items():
            st_dtype = dtype.normalize(t.dtype)
            nbytes = t.numel() * t.element_size()
            meta = TensorMeta(name, st_dtype, tuple(t.shape), offset, offset + nbytes)
            metas.append(meta)
            offset += nbytes
            if offset % 64 != 0:
                offset = ((offset + 63) // 64) * 64

        file_path = tmp_path / "torch.safetensors"

        # Write
        with StreamWriter.open(file_path, metas, crc32=True) as writer:
            for name, t in tensors.items():
                data = t.detach().cpu().numpy().tobytes()
                writer.write_block(name, data)
                writer.finalize_tensor(name)

        # Read back as PyTorch tensors
        with StreamReader(file_path, verify_crc=True) as reader:
            loaded1 = reader.to_torch("tensor1")
            loaded2 = reader.to_torch("tensor2")
            loaded3 = reader.to_torch("tensor3")

            assert torch.allclose(tensor1, loaded1)
            assert torch.equal(tensor2, loaded2)
            # BF16 might have small differences due to conversion
            assert loaded3.dtype == torch.bfloat16

    def test_reader_iterator(self, tmp_path):
        """Test reader iteration interface."""
        # Create multiple tensors
        tensors = {}
        metas = []
        offset = 0

        for i in range(5):
            name = f"tensor_{i}"
            data = np.random.rand(10, 10).astype(np.float32)
            tensors[name] = data

            meta = TensorMeta(name, "F32", data.shape, offset, offset + data.nbytes)
            metas.append(meta)
            offset = ((offset + data.nbytes + 63) // 64) * 64

        file_path = tmp_path / "multi.safetensors"

        # Write
        with StreamWriter.open(file_path, metas) as writer:
            for name, data in tensors.items():
                writer.write_block(name, data.tobytes())
                writer.finalize_tensor(name)

        # Read using iterator
        with StreamReader(file_path) as reader:
            # Test __len__
            assert len(reader) == 5

            # Test __contains__
            assert "tensor_0" in reader
            assert "nonexistent" not in reader

            # Test iteration
            for i, name in enumerate(reader):
                assert name == f"tensor_{i}"
                loaded = reader.to_numpy(name)
                np.testing.assert_array_equal(tensors[name], loaded)

            # Test keys()
            assert reader.keys() == [f"tensor_{i}" for i in range(5)]

    def test_partial_file_reading(self, tmp_path):
        """Test reading a partially written file."""
        # Create metadata for 3 tensors
        metas = [
            TensorMeta("tensor1", "F32", (10, 10), 0, 400),
            TensorMeta("tensor2", "F32", (10, 10), 448, 848),  # Aligned to 64
            TensorMeta("tensor3", "F32", (10, 10), 896, 1296),
        ]

        file_path = tmp_path / "partial.safetensors"

        # Write only first two tensors
        with StreamWriter.open(file_path, metas) as writer:
            data1 = np.random.rand(10, 10).astype(np.float32)
            writer.write_block("tensor1", data1.tobytes())
            writer.finalize_tensor("tensor1")

            data2 = np.random.rand(10, 10).astype(np.float32)
            writer.write_block("tensor2", data2.tobytes())
            writer.finalize_tensor("tensor2")

            # Abort before writing tensor3
            writer.abort()

        # Try to read with allow_partial=True
        with StreamReader(file_path, allow_partial=True) as reader:
            # Should be able to read first two tensors
            loaded1 = reader.to_numpy("tensor1")
            loaded2 = reader.to_numpy("tensor2")

            np.testing.assert_array_equal(data1, loaded1)
            np.testing.assert_array_equal(data2, loaded2)

            # Third tensor should fail
            with pytest.raises(Exception):  # Could be various errors
                reader.to_numpy("tensor3")


class TestErrorConditions:
    """Test error handling in writer/reader."""

    def test_write_out_of_order(self, tmp_path):
        """Test writing tensors out of order fails."""
        metas = [
            TensorMeta("tensor1", "F32", (10,), 0, 40),
            TensorMeta("tensor2", "F32", (10,), 64, 104),
        ]

        file_path = tmp_path / "test.safetensors"

        with StreamWriter.open(file_path, metas) as writer:
            # Try to write tensor2 first
            with pytest.raises(Exception) as exc_info:
                writer.write_block("tensor2", np.zeros(40, dtype=np.float32).tobytes())

            assert "order" in str(exc_info.value).lower()

    def test_write_wrong_size(self, tmp_path):
        """Test writing wrong amount of data fails."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "test.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            # Write too much data
            with pytest.raises(Exception) as exc_info:
                writer.write_block("test", np.zeros(50, dtype=np.float32).tobytes())

            assert (
                "length" in str(exc_info.value).lower()
                or "size" in str(exc_info.value).lower()
            )

    def test_finalize_wrong_tensor(self, tmp_path):
        """Test finalizing wrong tensor fails."""
        metas = [
            TensorMeta("tensor1", "F32", (10,), 0, 40),
            TensorMeta("tensor2", "F32", (10,), 64, 104),
        ]

        file_path = tmp_path / "test.safetensors"

        with StreamWriter.open(file_path, metas) as writer:
            writer.write_block("tensor1", np.zeros(40, dtype=np.float32).tobytes())

            # Try to finalize wrong tensor
            with pytest.raises(Exception) as exc_info:
                writer.finalize_tensor("tensor2")

            assert (
                "tensor1" in str(exc_info.value)
                or "order" in str(exc_info.value).lower()
            )

    def test_read_nonexistent_tensor(self, tmp_path):
        """Test reading non-existent tensor fails."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "test.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            writer.write_block("test", np.zeros(40, dtype=np.float32).tobytes())
            writer.finalize_tensor("test")

        with StreamReader(file_path) as reader:
            with pytest.raises(KeyError):
                reader.to_numpy("nonexistent")
