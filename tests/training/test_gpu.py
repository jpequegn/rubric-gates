"""Tests for GPU detection and model size recommendations."""

from training.gpu import GPUInfo, detect_gpu, recommend_batch_size, recommend_model_size


class TestGPUInfo:
    def test_defaults(self):
        info = GPUInfo()
        assert info.available is False
        assert info.device_name == ""
        assert info.vram_gb == 0.0
        assert info.compute_capability == ""
        assert info.cuda_version == ""
        assert info.device_count == 0

    def test_custom_values(self):
        info = GPUInfo(
            available=True,
            device_name="NVIDIA RTX 4090",
            vram_gb=24.0,
            compute_capability="8.9",
            cuda_version="12.1",
            device_count=1,
        )
        assert info.available is True
        assert info.device_name == "NVIDIA RTX 4090"
        assert info.vram_gb == 24.0


class TestDetectGPU:
    def test_returns_gpu_info(self):
        result = detect_gpu()
        assert isinstance(result, GPUInfo)


class TestRecommendModelSize:
    def test_no_gpu(self):
        info = GPUInfo(available=False)
        assert recommend_model_size(info) == "cpu-only"

    def test_mps(self):
        info = GPUInfo(available=True, compute_capability="mps")
        assert recommend_model_size(info) == "1-3B"

    def test_8gb_vram(self):
        info = GPUInfo(available=True, vram_gb=8.0, compute_capability="8.0")
        assert recommend_model_size(info) == "1-3B"

    def test_16gb_vram(self):
        info = GPUInfo(available=True, vram_gb=16.0, compute_capability="8.0")
        assert recommend_model_size(info) == "3-7B"

    def test_24gb_vram(self):
        info = GPUInfo(available=True, vram_gb=24.0, compute_capability="8.0")
        assert recommend_model_size(info) == "7-13B"

    def test_48gb_vram(self):
        info = GPUInfo(available=True, vram_gb=48.0, compute_capability="9.0")
        assert recommend_model_size(info) == "7-13B"

    def test_small_vram(self):
        info = GPUInfo(available=True, vram_gb=4.0, compute_capability="7.5")
        assert recommend_model_size(info) == "< 1B"


class TestRecommendBatchSize:
    def test_no_gpu(self):
        info = GPUInfo(available=False)
        assert recommend_batch_size(info) == 1

    def test_8gb(self):
        info = GPUInfo(available=True, vram_gb=8.0)
        assert recommend_batch_size(info) == 2

    def test_16gb(self):
        info = GPUInfo(available=True, vram_gb=16.0)
        assert recommend_batch_size(info) == 4

    def test_24gb(self):
        info = GPUInfo(available=True, vram_gb=24.0)
        assert recommend_batch_size(info) == 8

    def test_small_vram(self):
        info = GPUInfo(available=True, vram_gb=4.0)
        assert recommend_batch_size(info) == 1
