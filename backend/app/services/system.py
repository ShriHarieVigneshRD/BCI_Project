"""Environment and software-version reporting."""

from __future__ import annotations

import platform
from functools import lru_cache
from importlib import metadata


@lru_cache(maxsize=1)
def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform(terse=True)}
    for pkg in ("torch", "numpy", "scipy", "scikit-learn", "mne", "pandas", "fastapi", "pydantic"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def device_info() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    info = {"cuda_available": cuda, "default_device": "cuda" if cuda else "cpu", "cuda_device": None,
            "cuda_version": torch.version.cuda, "torch_threads": torch.get_num_threads()}
    if cuda:
        props = torch.cuda.get_device_properties(0)
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_memory_gb"] = round(props.total_memory / 1024**3, 2)
    return info
