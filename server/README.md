# LabelOne local service

This service owns local filesystem access, dataset matching, the model catalog,
model adapter lifecycles, inference, and tensor artifacts. The hosted UI remains
in demo mode; running the UI on localhost connects to this service at
`http://127.0.0.1:8766/api/v1`.

The X-AnyLabeling integration reads user-provided YAML metadata and uses
independently implemented adapters. It does not import or execute X-AnyLabeling
Python modules.

Implemented clean-room ONNX adapters currently cover generic exported tensors,
YOLO HBB/OBB/segmentation/pose/classification, RT-DETR/D-FINE/DEIM, Depth
Anything, RMBG, RAM/RAM++, PP-OCR v4–v6, and standard SAM/SAM-HQ contracts.

Feature artifacts can apply real channel projection, normalization, spatial
scaling, gain/gamma, and percentile clipping before atomic persistence. The
manifest records both source and output shapes plus the transform parameters.
For single-session ONNX adapters, model load performs bounded graph inspection
and enumerates float NCHW/NTC graph outputs and intermediate tensors. A selected
intermediate tensor is exposed with a one-layer graph rewrite at inference time;
the runtime never requests every enumerated tensor. Captures are limited to one
layer and fixed 256 MiB source/artifact budgets. Scalar projected artifacts also
receive a cached, bounded PNG preview while preserving the original NPY.
External-data and oversized ONNX graphs currently degrade to exported outputs,
and multi-session PP-OCR/SAM adapters retain their explicit stage outputs.

Each runnable adapter publishes a bounded user-parameter JSON Schema even when
its YAML omits those keys. The Web client uses this schema for model-specific
numeric, boolean, and enum controls and sends only the selected model's declared
parameters; detection thresholds are no longer injected into depth, tagging, or
classification adapters.
Depth/mask/cutout outputs are persisted as raster artifacts with a validated
content endpoint and can be overlaid on the Web canvas.

Importing a source tree does not download weights. Remote files are listed in
the model library and require a two-step explicit confirmation. The downloader
accepts only allowlisted HTTPS domains, revalidates redirects, enforces a byte
budget, hashes streaming content, and atomically records local overrides without
modifying the imported YAML.

Whole-dataset pipeline and inference runs are stored in SQLite. Job creation is
idempotent, per-image progress is queryable, and pause/cancel intent survives a
service restart. The current scheduler is intended for one local API process;
distributed workers and multi-process access are not supported yet.

Batch and interactive work share a weighted fair scheduler. Interactive,
user-batch, and background priorities use 8:3:1 deficit round-robin with aging;
CPU pipelines and each model use separate capacity lanes. The default model
lane capacity is one to prevent concurrent session/OOM races.

Pipeline operators publish strict JSON Schemas, semantic versions, Image→Image
contracts, and annotation policies. Imported local operator packages run in a
bounded worker process and must explicitly declare a verified spatial contract.
Non-spatial operators use `spatial_behavior: none`, `size_behavior: preserve`,
and `annotation_policy: preserve`. Pure resize operators use
`spatial_behavior: scale_xy` with deterministic/dynamic sizing and
`annotation_policy: scale`. Other geometry such as crop, flip, rotate,
perspective, or nonlinear warp uses `spatial_behavior: custom`,
`annotation_policy: transform`, and an `annotation_entrypoint`. The annotation
function receives the document, the same parameters, and input/output
dimensions; invalid or out-of-bounds results fail before artifact publication.

```yaml
api_version: labelone.operator/v1
id: acme.resize
name: ACME Resize
description: Resize an image to an explicit width and height.
version: 1.0.0
entrypoint: operator.py:process
size_behavior: dynamic
spatial_behavior: scale_xy
annotation_policy: scale
parameters_schema:
  type: object
  properties:
    width:
      title: Width
      description: Target image width in pixels.
      type: integer
      minimum: 1
    height:
      title: Height
      description: Target image height in pixels.
      type: integer
      minimum: 1
```

Custom spatial packages additionally declare:

```yaml
spatial_behavior: custom
annotation_policy: transform
annotation_entrypoint: operator.py:transform_annotations
```

```python
def transform_annotations(document, parameters, context):
    # Apply the same crop/flip/rotate/perspective mapping as process().
    return document
```

The local Agent is an allowlisted tool layer, not an unrestricted shell. Read
operations execute against the selected dataset; actions that create artifacts
or jobs are persisted as proposals and require explicit confirmation in the Web
UI before idempotent execution. The public Agent API is gated by planner
readiness: `GET /api/v1/agent/status` reports configuration state and the
allowlisted capability matrix, while run and proposal-execution endpoints
return `agent_backend_unavailable` until an enabled configuration and its
credential environment variable are both present.

## Development

```bash
./scripts/dev-local.sh
```

The launcher checks Node.js, pnpm, uv, dependencies, ports, and the API health
endpoint before exposing the Web UI. Node.js 22.13+ is required.

Run backend tests separately with `cd server && uv run --no-sync pytest` after
the initial environment sync.

For very large tiled TIFF/WSI images, install system `libvips` and the optional
Python binding (`uv pip install pyvips`). The tile metadata
reports `backend=pyvips` when region-streaming is active; otherwise the service
uses a bounded Pillow fallback and reports `backend=pillow`.

To import a local X-AnyLabeling checkout at startup:

```bash
LABELONE_X_ANYLABELING_ROOT=/path/to/X-AnyLabeling uv run labelone-server
```

To force a model-weight download directory (overriding the persisted UI setting):

```bash
LABELONE_MODEL_WEIGHTS_DIR=/path/to/model-weights uv run labelone-server
```

The built-in HYPIR-SD2 entry uses a separate, explicitly configured CUDA runtime so its pinned Python 3.10 / Torch 2.6 dependencies do not alter the LabelOne server environment:

```bash
LABELONE_HYPIR_PYTHON=/path/to/hypir-env/bin/python \
LABELONE_HYPIR_ROOT=/path/to/HYPIR \
LABELONE_HYPIR_SD21_BASE=/path/to/stable-diffusion-2-1-base \
LABELONE_HYPIR_SD2_WEIGHT=/path/to/HYPIR_sd2.pth \
uv run labelone-server
```

HYPIR is non-commercial-only without separate permission from SupPixel. LabelOne does not bundle its source, SD2.1 base model, or weights, and does not download them implicitly.

Cloud Agent planning is configured through the local Web settings page. The
persisted settings contain only the HTTPS endpoint, model id, limits, and an
environment-variable name. The credential itself must be provided through that
environment variable and is never returned by the settings API. The cloud model
can only produce an allowlisted tool plan; existing validation and write-action
confirmation still apply. Local tool replies and result summaries are not
replayed into later cloud-planner requests, so display paths and tool results do
not cross the configured planning boundary through conversation history.
