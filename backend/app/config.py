"""Runtime settings (environment variables, see ../.env.example)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_VERSION = "1.0.0"


def _load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env support: KEY=VALUE lines; real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _project_path(value: str | None, default: Path) -> Path:
    p = Path(value) if value else default
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


_load_dotenv()

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = PROJECT_ROOT / "data" / "cache" / "matplotlib"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)


class Settings:
    def __init__(self) -> None:
        self.data_dir = _project_path(os.environ.get("EEGNET_DATA_DIR"), PROJECT_ROOT / "data")
        self.host = os.environ.get("EEGNET_HOST", "127.0.0.1")
        self.port = int(os.environ.get("EEGNET_PORT", "8000"))
        self.max_upload_mb = int(os.environ.get("EEGNET_MAX_UPLOAD_MB", "1024"))
        self.max_import_mb = int(os.environ.get("EEGNET_MAX_IMPORT_MB", "20480"))
        self.import_roots = [r for r in os.environ.get("EEGNET_IMPORT_ROOTS", "").split(os.pathsep) if r.strip()]
        self.cors_origins = [o.strip() for o in os.environ.get(
            "EEGNET_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
        self.frontend_dist = PROJECT_ROOT / "frontend" / "dist"

    def path(self, *parts: str) -> Path:
        p = self.data_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()


def configure(data_dir: Path | str) -> None:
    """Point the application at another data directory (used by tests)."""
    settings.data_dir = Path(data_dir).resolve()
