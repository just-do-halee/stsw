"""Happy path tests for StreamWriter."""

import numpy as np

from stsw import StreamWriter, TensorMeta


class TestStreamWriterHappyPath:
    """Test successful writer operations."""

    def test_create_writer(self, tmp_path):
        """Test creating a writer."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "test.safetensors"

        writer = StreamWriter.open(file_path, [meta])
        assert writer is not None
        # Write the tensor data
        writer.write_block("test", np.zeros(10, dtype=np.float32).tobytes())
        writer.finalize_tensor("test")
        writer.close()

        # File should exist
        assert file_path.exists()

    def test_context_manager(self, tmp_path):
        """Test writer as context manager."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "test.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            writer.write_block("test", np.zeros(10, dtype=np.float32).tobytes())
            writer.finalize_tensor("test")

        assert file_path.exists()

    def test_write_single_tensor(self, tmp_path):
        """Test writing a single tensor."""
        data = np.arange(100, dtype=np.float32)
        meta = TensorMeta("data", "F32", data.shape, 0, data.nbytes)
        file_path = tmp_path / "single.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            writer.write_block("data", data.tobytes())
            writer.finalize_tensor("data")

        # Verify file size is reasonable
        file_size = file_path.stat().st_size
        assert file_size > data.nbytes  # Should include header
        assert file_size < data.nbytes * 2  # But not too large

    def test_write_with_custom_align(self, tmp_path):
        """Test writing with custom alignment."""
        data = np.ones(10, dtype=np.float32)
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "aligned.safetensors"

        # Use 128-byte alignment
        with StreamWriter.open(file_path, [meta], align=128) as writer:
            writer.write_block("test", data.tobytes())
            writer.finalize_tensor("test")

        assert file_path.exists()

    def test_write_with_large_buffer(self, tmp_path):
        """Test writing with large buffer size."""
        data = np.random.rand(1000, 1000).astype(np.float32)
        meta = TensorMeta("large", "F32", data.shape, 0, data.nbytes)
        file_path = tmp_path / "buffered.safetensors"

        # Use 16 MB buffer
        with StreamWriter.open(
            file_path, [meta], buffer_size=16 * 1024 * 1024
        ) as writer:
            writer.write_block("large", data.tobytes())
            writer.finalize_tensor("large")

        assert file_path.exists()

    def test_get_stats(self, tmp_path):
        """Test getting writer statistics."""
        data = np.random.rand(100, 100).astype(np.float32)
        meta = TensorMeta("test", "F32", data.shape, 0, data.nbytes)
        file_path = tmp_path / "stats.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            # Check initial stats
            stats1 = writer.stats()
            assert stats1.written == 0
            assert stats1.total == data.nbytes
            assert stats1.mb_per_s >= 0
            assert stats1.eta_s >= 0
            assert stats1.rss_mb >= 0

            # Write half the data
            half_data = data.tobytes()[: data.nbytes // 2]
            writer.write_block("test", half_data)

            # Check updated stats
            stats2 = writer.stats()
            assert stats2.written == len(half_data)
            assert stats2.total == data.nbytes

            # Write rest
            writer.write_block("test", data.tobytes()[data.nbytes // 2 :])
            writer.finalize_tensor("test")

            # Final stats
            stats3 = writer.stats()
            assert stats3.written == data.nbytes

    def test_abort_writer(self, tmp_path):
        """Test aborting writer."""
        meta = TensorMeta("test", "F32", (10,), 0, 40)
        file_path = tmp_path / "aborted.safetensors"

        with StreamWriter.open(file_path, [meta]) as writer:
            writer.write_block("test", np.zeros(5, dtype=np.float32).tobytes())
            # Abort instead of finishing
            writer.abort()

        # Temp file should be cleaned up
        assert not file_path.exists()
        assert not file_path.with_suffix(".safetensors.tmp").exists()
