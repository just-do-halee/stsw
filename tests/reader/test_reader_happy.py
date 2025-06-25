"""Happy path tests for StreamReader."""

import numpy as np
import pytest

from stsw import StreamReader, StreamWriter, TensorMeta


class TestStreamReaderHappyPath:
    """Test successful reader operations."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample safetensors file."""
        data1 = np.arange(100, dtype=np.float32).reshape(10, 10)
        data2 = np.ones((5, 5), dtype=np.int32) * 42

        metas = [
            TensorMeta("array1", "F32", data1.shape, 0, data1.nbytes),
            TensorMeta(
                "array2", "I32", data2.shape, 448, 448 + data2.nbytes
            ),  # Aligned
        ]

        file_path = tmp_path / "sample.safetensors"

        with StreamWriter.open(file_path, metas, crc32=True) as writer:
            writer.write_block("array1", data1.tobytes())
            writer.finalize_tensor("array1")
            writer.write_block("array2", data2.tobytes())
            writer.finalize_tensor("array2")

        return file_path, {"array1": data1, "array2": data2}

    def test_open_reader(self, sample_file):
        """Test opening a reader."""
        file_path, _ = sample_file

        reader = StreamReader(file_path)
        assert reader is not None
        reader.close()

    def test_context_manager(self, sample_file):
        """Test reader as context manager."""
        file_path, _ = sample_file

        with StreamReader(file_path) as reader:
            assert len(reader) == 2
            assert "array1" in reader
            assert "array2" in reader

    def test_list_tensors(self, sample_file):
        """Test listing tensor names."""
        file_path, _ = sample_file

        with StreamReader(file_path) as reader:
            keys = reader.keys()
            assert keys == ["array1", "array2"]

    def test_get_metadata(self, sample_file):
        """Test getting tensor metadata."""
        file_path, _ = sample_file

        with StreamReader(file_path) as reader:
            meta1 = reader.meta("array1")
            assert meta1.name == "array1"
            assert meta1.dtype == "F32"
            assert meta1.shape == (10, 10)
            assert meta1.nbytes == 400
            assert meta1.crc32 is not None  # Was written with CRC

            meta2 = reader.meta("array2")
            assert meta2.name == "array2"
            assert meta2.dtype == "I32"
            assert meta2.shape == (5, 5)

    def test_read_as_numpy(self, sample_file):
        """Test reading tensors as numpy arrays."""
        file_path, expected = sample_file

        with StreamReader(file_path) as reader:
            array1 = reader.to_numpy("array1")
            array2 = reader.to_numpy("array2")

            np.testing.assert_array_equal(array1, expected["array1"])
            np.testing.assert_array_equal(array2, expected["array2"])

            # Check dtypes are preserved
            assert array1.dtype == np.float32
            assert array2.dtype == np.int32

    def test_get_slice(self, sample_file):
        """Test getting raw tensor data as memoryview."""
        file_path, expected = sample_file

        with StreamReader(file_path) as reader:
            # Get raw data
            slice1 = reader.get_slice("array1")
            assert isinstance(slice1, memoryview)
            assert len(slice1) == expected["array1"].nbytes

            # Can create numpy array from it
            array = np.frombuffer(slice1, dtype=np.float32).reshape(10, 10)
            np.testing.assert_array_equal(array, expected["array1"])

    def test_verify_crc(self, sample_file):
        """Test CRC verification."""
        file_path, _ = sample_file

        # Read with CRC verification enabled
        with StreamReader(file_path, verify_crc=True) as reader:
            # This should trigger CRC check
            array1 = reader.to_numpy("array1")
            assert array1 is not None

            # Second access should use cached verification
            array1_again = reader.to_numpy("array1")
            np.testing.assert_array_equal(array1, array1_again)

    def test_reader_without_mmap(self, sample_file):
        """Test reader without memory mapping."""
        file_path, expected = sample_file

        with StreamReader(file_path, mmap=False) as reader:
            # Should still work but use fallback
            array1 = reader.to_numpy("array1")
            np.testing.assert_array_equal(array1, expected["array1"])

    def test_iterate_tensors(self, sample_file):
        """Test iterating over tensors."""
        file_path, expected = sample_file

        with StreamReader(file_path) as reader:
            names = list(reader)
            assert names == ["array1", "array2"]

            # Can also use in for loop
            for i, name in enumerate(reader):
                array = reader.to_numpy(name)
                assert array is not None

    def test_getitem_access(self, sample_file):
        """Test dictionary-style access."""
        file_path, expected = sample_file

        with StreamReader(file_path) as reader:
            # Use [] operator
            slice1 = reader["array1"]
            assert isinstance(slice1, memoryview)

            # Convert to array
            array = np.frombuffer(slice1, dtype=np.float32).reshape(10, 10)
            np.testing.assert_array_equal(array, expected["array1"])

    @pytest.mark.skipif(
        not pytest.importorskip("torch"), reason="PyTorch not installed"
    )
    def test_read_as_torch(self, tmp_path):
        """Test reading as PyTorch tensors."""
        import torch

        # Create file with PyTorch tensor
        tensor = torch.randn(5, 5, dtype=torch.float32)
        meta = TensorMeta("tensor", "F32", tuple(tensor.shape), 0, tensor.numel() * 4)

        file_path = tmp_path / "torch.safetensors"
        with StreamWriter.open(file_path, [meta]) as writer:
            writer.write_block("tensor", tensor.numpy().tobytes())
            writer.finalize_tensor("tensor")

        # Read back
        with StreamReader(file_path) as reader:
            loaded = reader.to_torch("tensor")
            assert loaded.dtype == torch.float32
            assert loaded.shape == tensor.shape
            assert torch.allclose(loaded, tensor)

            # Test GPU loading (if available)
            if torch.cuda.is_available():
                loaded_gpu = reader.to_torch("tensor", device="cuda")
                assert loaded_gpu.is_cuda
                assert torch.allclose(loaded_gpu.cpu(), tensor)

    def test_file_metadata(self, tmp_path):
        """Test reading file-level metadata."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        user_metadata = {"model": "test_model", "version": 1, "author": "test"}

        file_path = tmp_path / "with_metadata.safetensors"

        # Write with metadata (need to modify writer to support this)

        from stsw._core.header import build_header

        header_bytes = build_header([meta], metadata=user_metadata, incomplete=False)

        with open(file_path, "wb") as f:
            f.write(header_bytes)
            f.write(np.zeros(40, dtype=np.float32).tobytes())
            # Add padding to align
            f.write(b"\x00" * 24)

        # Read back
        with StreamReader(file_path) as reader:
            assert reader.metadata == user_metadata
            assert reader.version == "1.0"
