from __future__ import annotations


class LabelOneError(Exception):
    """Base error with a stable API error code."""

    code = "labelone_error"

    def __init__(self, message: str, *, details: dict[str, object] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidPathError(LabelOneError):
    code = "invalid_path"


class ModelCatalogError(LabelOneError):
    code = "model_catalog_error"


class ModelRuntimeError(LabelOneError):
    code = "model_runtime_error"


class AnnotationValidationError(LabelOneError):
    code = "annotation_validation_error"


class RevisionConflictError(LabelOneError):
    code = "revision_conflict"


class AgentBackendUnavailableError(LabelOneError):
    code = "agent_backend_unavailable"
    status_code = 503
