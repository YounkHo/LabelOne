from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class AnnotationEnvelope(BaseModel):
    dataset_id: str
    asset_id: str
    path: Path
    revision: str
    document: dict[str, Any]


class AnnotationSaveRequest(BaseModel):
    document: dict[str, Any]


class AnnotationSaveResponse(BaseModel):
    dataset_id: str
    asset_id: str
    path: Path
    previous_revision: str
    revision: str
    backup_path: Path

