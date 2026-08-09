"""Ollama diagnostics and hardware detection."""

from __future__ import annotations

import os
import platform
import subprocess
import urllib.error
import urllib.request


def ollama_is_running(base_url: str, timeout: float | None = None) -> bool:
    # Allow overriding timeout via env var — useful on slow machines or WSL2.
    if timeout is None:
        try:
            timeout = float(os.getenv("OLLAMA_TIMEOUT", "3.0"))
        except ValueError:
            timeout = 3.0
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _system_memory_gb() -> float | None:
    """Return total physical RAM in GB, cross-platform."""
    system = platform.system()
    try:
        if system == "Darwin":
            output = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            return round(int(output) / 1024 ** 3, 1)

        if system == "Windows":
            # os.sysconf is Linux-only; use ctypes on Windows instead.
            import ctypes
            mem_kb = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(mem_kb))
            return round(mem_kb.value / (1024 * 1024), 1)  # KB → GB

        # Linux and other POSIX systems.
        return round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3, 1,
        )
    except (ValueError, OSError, subprocess.SubprocessError, AttributeError):
        return None


def detect_compute_device() -> dict:
    info = {
        "cuda": False,
        "mps": False,
        "selected": "cpu",
        "cpu_threads": os.cpu_count() or 1,
        "cuda_name": None,
        "cuda_vram_gb": None,
        "system_memory_gb": _system_memory_gb(),
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        lines = [ln.strip() for ln in result.stdout.strip().splitlines() if ln.strip()]
        if lines:
            # Pick the GPU with the most VRAM when multiple GPUs are present.
            best_name, best_vram = "", 0.0
            for line in lines:
                try:
                    name, memory_mb = line.split(",", 1)
                    vram = float(memory_mb.strip())
                    if vram > best_vram:
                        best_vram = vram
                        best_name = name.strip()
                except (ValueError, IndexError):
                    continue
            if best_name:
                info.update(
                    cuda=True,
                    cuda_name=best_name,
                    cuda_vram_gb=round(best_vram / 1024, 1),
                )
    except (OSError, subprocess.SubprocessError):
        pass

    info["mps"] = (
        platform.system() == "Darwin"
        and platform.machine().lower() in {"arm64", "aarch64"}
    )
    info["selected"] = "cuda" if info["cuda"] else "mps" if info["mps"] else "cpu"
    return info


def resolve_ollama_profile(requested: str = "auto") -> dict:
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Ollama profile must be auto, cuda, mps, or cpu.")

    hardware = detect_compute_device()
    name = hardware["selected"] if requested == "auto" else requested

    if name == "cuda":
        context = 8192 if (hardware["cuda_vram_gb"] or 0) >= 12 else 4096
    elif name == "mps":
        context = 8192 if (hardware["system_memory_gb"] or 0) >= 24 else 4096
    else:
        context = 4096 if (hardware["system_memory_gb"] or 0) >= 16 else 2048

    return {"name": name, "hardware": hardware, "context": context}


def print_profile(profile: dict) -> None:
    hardware = profile["hardware"]
    print("\nLocal accelerator detection:")
    print(f"  CUDA: {'available' if hardware['cuda'] else 'not found'}")
    if hardware["cuda"] and hardware["cuda_name"]:
        print(f"    GPU: {hardware['cuda_name']} ({hardware['cuda_vram_gb']} GB VRAM)")
    print(f"  Apple Metal: {'available' if hardware['mps'] else 'not found'}")
    print(f"  CPU threads: {hardware['cpu_threads']}")
    print(f"  System memory: {hardware['system_memory_gb'] or 'unknown'} GB")
    print(f"\nRecommended Ollama profile: {profile['name'].upper()}")
    print(f"  Context window: {profile['context']}")
