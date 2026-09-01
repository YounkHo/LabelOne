from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8766
    x_anylabeling_root: Path | None = None
    model_weights_dir: Path | None = None
    data_dir: Path = Path.home() / ".labelone"

    @classmethod
    def from_env(cls) -> "Settings":
        raw_root = os.getenv("LABELONE_X_ANYLABELING_ROOT")
        raw_model_weights = os.getenv("LABELONE_MODEL_WEIGHTS_DIR")
        return cls(
            host=os.getenv("LABELONE_HOST", "127.0.0.1"),
            port=int(os.getenv("LABELONE_PORT", "8766")),
            x_anylabeling_root=Path(raw_root).expanduser() if raw_root else None,
            model_weights_dir=Path(raw_model_weights).expanduser() if raw_model_weights else None,
            data_dir=Path(os.getenv("LABELONE_DATA_DIR", str(Path.home() / ".labelone"))).expanduser(),
        )


settings = Settings.from_env()
