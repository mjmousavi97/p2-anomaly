"""Runtime environment diagnostics for the anomaly detection project."""

import datetime
import os
import platform

import torch


def _ram_total_gib():
    """Return total system RAM in GiB, or None if unavailable."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as file:
            for line in file:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return round(kib / 1024**2, 2)
    except (OSError, ValueError):
        return None

    return None


def environment_report():
    """Return a JSON-serializable report about the runtime environment."""
    cuda_available = torch.cuda.is_available()

    gpu_name = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "utc_time": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gib": _ram_total_gib(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
    }
