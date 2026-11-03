"""Validate cached result metadata before any resumable pipeline stage."""
import json
from pathlib import Path


def verify_cached_result(path, expected_identity, **expected_fields):
    """Reject a cached result from another protocol or model variant."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("identity") != expected_identity:
        raise ValueError(f"Cached result identity differs: {path}")
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"Cached result field {field!r} differs in {path}: "
                f"expected {expected!r}, found {payload.get(field)!r}"
            )
    return payload


def verify_existing_artifacts(root, run):
    """Check every resumable metric artifact against its locked protocol."""
    out = Path(root) / "results" / "experiments" / run
    protocol_path = out / "protocol.json"
    if not protocol_path.exists():
        return 0
    identity = json.loads(protocol_path.read_text(encoding="utf-8"))["identity"]
    checked = 0
    for path in sorted(out.glob("*.json")):
        if path == protocol_path:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "identity" in payload:
            verify_cached_result(path, identity)
            checked += 1
    controlled = out / "controlled"
    if controlled.exists():
        for path in sorted(controlled.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "identity" in payload:
                verify_cached_result(path, identity)
                checked += 1
    return checked
