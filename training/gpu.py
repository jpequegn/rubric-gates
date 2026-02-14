"""GPU detection and model size recommendations.

Detects local GPU capabilities and recommends appropriate model sizes
for training within available VRAM constraints.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInfo:
    """Information about available GPU hardware."""

    available: bool = False
    device_name: str = ""
    vram_gb: float = 0.0
    compute_capability: str = ""
    cuda_version: str = ""
    device_count: int = 0


def detect_gpu() -> GPUInfo:
    """Detect local GPU capabilities.

    Returns GPUInfo with available=False if no GPU or torch not installed.
    """
    try:
        import torch
    except ImportError:
        return GPUInfo(available=False)

    if not torch.cuda.is_available():
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return GPUInfo(
                available=True,
                device_name="Apple Silicon (MPS)",
                vram_gb=0.0,  # MPS shares system memory
                compute_capability="mps",
                device_count=1,
            )
        return GPUInfo(available=False)

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_mem
    vram_gb = vram_bytes / (1024**3)

    major, minor = torch.cuda.get_device_capability(0)
    compute_cap = f"{major}.{minor}"

    cuda_version = torch.version.cuda or ""

    return GPUInfo(
        available=True,
        device_name=device_name,
        vram_gb=round(vram_gb, 1),
        compute_capability=compute_cap,
        cuda_version=cuda_version,
        device_count=device_count,
    )


def recommend_model_size(gpu_info: GPUInfo) -> str:
    """Suggest appropriate model size based on available VRAM.

    Recommendations assume 4-bit quantization for efficient local training.

    Args:
        gpu_info: GPU information from detect_gpu().

    Returns:
        Recommended model size string.
    """
    if not gpu_info.available:
        return "cpu-only"

    # MPS (Apple Silicon) — use system memory heuristic
    if gpu_info.compute_capability == "mps":
        return "1-3B"

    vram = gpu_info.vram_gb

    if vram >= 24:
        return "7-13B"
    elif vram >= 16:
        return "3-7B"
    elif vram >= 8:
        return "1-3B"
    else:
        return "< 1B"


def recommend_batch_size(gpu_info: GPUInfo) -> int:
    """Suggest training batch size based on available VRAM.

    Args:
        gpu_info: GPU information from detect_gpu().

    Returns:
        Recommended batch size.
    """
    if not gpu_info.available:
        return 1

    vram = gpu_info.vram_gb

    if vram >= 24:
        return 8
    elif vram >= 16:
        return 4
    elif vram >= 8:
        return 2
    else:
        return 1
