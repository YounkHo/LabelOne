from __future__ import annotations

import os
from pathlib import Path
import sys
from time import sleep

from .worker_protocol import PROTOCOL_VERSION, read_message, write_message


MAXIMUM_BYTES = 8 * 1024 * 1024


def main() -> int:
    init = read_message(sys.stdin.buffer, maximum_bytes=MAXIMUM_BYTES)
    if init is None or init.get("type") != "init" or init.get("protocol") != PROTOCOL_VERSION:
        return 2
    options = init.get("options")
    if not isinstance(options, dict):
        options = {}
    write_message(
        sys.stdout.buffer,
        {"type": "ready", "protocol": PROTOCOL_VERSION, "pid": os.getpid()},
        maximum_bytes=MAXIMUM_BYTES,
    )
    loaded = False
    sequence = 0
    while True:
        request = read_message(sys.stdin.buffer, maximum_bytes=MAXIMUM_BYTES)
        if request is None:
            return 0
        request_id = request.get("id")
        operation = request.get("op")
        payload = request.get("payload")
        if not isinstance(request_id, int) or not isinstance(operation, str) or not isinstance(payload, dict):
            return 3
        delay = options.get("delay_seconds", 0)
        if operation == options.get("delay_op") and isinstance(delay, (int, float)) and delay > 0:
            sleep(float(delay))
        crash_file = options.get("crash_once_file")
        if operation == options.get("crash_op"):
            if options.get("crash_always") is True:
                os._exit(int(options.get("crash_exit_code", 137)))
            if isinstance(crash_file, str):
                marker = Path(crash_file)
                if not marker.exists():
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("crashed", encoding="utf-8")
                    os._exit(int(options.get("crash_exit_code", 137)))
        if operation == "load":
            loaded = True
            result = {"model_id": init.get("model_id"), "state": "loaded", "layers": [], "error": None}
        elif operation == "layers":
            result = {"model_id": init.get("model_id"), "state": "loaded" if loaded else "unloaded", "layers": [], "error": None}
        elif operation == "predict":
            if not loaded:
                write_message(
                    sys.stdout.buffer,
                    {
                        "id": request_id,
                        "ok": False,
                        "error": {"code": "not_loaded", "message": "fixture is not loaded", "details": {}},
                    },
                    maximum_bytes=MAXIMUM_BYTES,
                )
                continue
            sequence += 1
            large = options.get("large_result_bytes", 0)
            result = {
                "model_id": init.get("model_id"),
                "image_path": payload.get("image_path"),
                "annotations": [],
                "classifications": [],
                "artifacts": [],
                "rasters": [],
                "timings_ms": {"total": 0.0},
                "sequence": sequence,
                "large": "x" * int(large) if isinstance(large, int) and large > 0 else "",
            }
        elif operation == "unload":
            loaded = False
            result = {"model_id": init.get("model_id"), "state": "unloaded", "layers": [], "error": None}
        elif operation == "close":
            loaded = False
            result = {"model_id": init.get("model_id"), "state": "unloaded", "layers": [], "error": None}
        else:
            return 4
        write_message(
            sys.stdout.buffer,
            {"id": request_id, "ok": True, "result": result},
            maximum_bytes=MAXIMUM_BYTES,
        )
        if operation == "close":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
