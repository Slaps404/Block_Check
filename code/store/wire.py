"""Generic dataclass<->JSON wire codec shared by the server (#114) and the Pi proxy (#115).

This module is a pure stdlib leaf: it imports ONLY ``dataclasses``, ``datetime``,
``pathlib``, ``typing``, ``json``, and ``types``. It MUST NOT import
``session_workflow``, ``slide_qr``, ``pipeline``, ``cv2``, or ``numpy`` — those
pull the heavy CV stack, and ``session_workflow`` importing this module back
would create an import cycle. The codec is therefore fully generic: ``encode``
walks ``dataclasses.fields()`` without naming any concrete class, and
``decode`` receives its target class as an argument.

Self-describing envelope: callers that own a wire dataclass call
``register(cls)`` once at import time to populate the module-global
``WIRE_TYPES`` registry. ``dumps``/``loads`` use that registry to round-trip
without either side needing to import the other's dataclass module.

POLICY (both sides must honor this):

1. No enums. ``verdict`` ('PASS'/'REVIEW'), ``phase``, ``slide_recovery_state``,
   ``upload_state`` are all bare strings on both sides. Do not coerce them into
   an enum type; the codec passes strings through unchanged.
2. Datetime canonical form is timezone-aware UTC ISO-8601. A naive
   ``datetime`` is a hard error on encode (``ValueError``), never silently
   assumed to be UTC.
3. ``Path`` is an opaque handle (often meaningful only on the Pi side, not the
   server). It is encoded as a plain string and never resolved.
4. Tuples always decode back to tuples, never lists — frozen dataclasses rely
   on ``==`` and a list in place of a tuple field would break equality.
5. Version skew fails loud: an unknown wire type name in ``loads`` raises
   ``ValueError`` rather than silently returning something wrong.
"""
from __future__ import annotations

import dataclasses
import json
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

_UnionType = getattr(types, "UnionType", None)

# Populated by each owning module via `register(cls)` at import time.
WIRE_TYPES: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a dataclass under its own name for envelope dispatch."""
    WIRE_TYPES[cls.__name__] = cls
    return cls


def _is_union(origin: object) -> bool:
    return origin is Union or (_UnionType is not None and origin is _UnionType)


def encode(value: Any) -> Any:
    """Turn a dataclass instance (recursively) into JSON-native data."""
    if type(value).__module__ == "numpy":
        raise TypeError("refusing to serialize array over the wire")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encode(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot encode value of type {type(value)!r}")


def _decode_field(hint: Any, raw: Any) -> Any:
    origin = get_origin(hint)

    if _is_union(origin):
        branches = [arg for arg in get_args(hint) if arg is not type(None)]
        if raw is None:
            return None
        if len(branches) == 1:
            return _decode_field(branches[0], raw)
        return raw

    if hint is datetime:
        return datetime.fromisoformat(raw)

    if hint is Path:
        return Path(raw)

    if isinstance(hint, type) and dataclasses.is_dataclass(hint):
        return decode(hint, raw)

    if origin is tuple:
        args = get_args(hint)
        if not args:
            return tuple(raw)
        sub = args[0]
        return tuple(_decode_field(sub, item) for item in raw)

    if origin is list:
        args = get_args(hint)
        if not args:
            return list(raw)
        sub = args[0]
        return [_decode_field(sub, item) for item in raw]

    return raw


def decode(cls: type, data: dict) -> Any:
    """Build an instance of ``cls`` from JSON-native ``data``.

    Driven by ``typing.get_type_hints(cls)`` so PEP563 string annotations
    (every source module uses ``from __future__ import annotations``) resolve
    to real types. Keys absent from ``data`` are omitted from the kwargs so
    dataclass defaults apply — this is what makes decode forward-compatible
    with a server that later adds an optional field.
    """
    hints = get_type_hints(cls)
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _decode_field(hints[field.name], data[field.name])
    return cls(**kwargs)


def dumps(obj: Any) -> str:
    """Serialize a registered dataclass instance with a self-describing envelope."""
    return json.dumps({"type": type(obj).__name__, "data": encode(obj)})


def loads(text: str) -> Any:
    """Deserialize a `dumps` envelope, dispatching on the registered type name."""
    envelope = json.loads(text)
    name = envelope["type"]
    cls = WIRE_TYPES.get(name)
    if cls is None:
        raise ValueError(f"unknown wire type: {name}")
    return decode(cls, envelope["data"])


def loads_as(cls: type, text: str) -> Any:
    """Deserialize into a known ``cls``, accepting either an envelope or bare data."""
    parsed = json.loads(text)
    if isinstance(parsed, dict) and "type" in parsed and "data" in parsed:
        data = parsed["data"]
    else:
        data = parsed
    return decode(cls, data)


def dumps_list(items: Any) -> str:
    """Serialize a homogeneous sequence of dataclass instances (or empty)."""
    items = list(items)
    item_type = type(items[0]).__name__ if items else None
    return json.dumps({
        "type": "list",
        "item": item_type,
        "data": [encode(item) for item in items],
    })


def loads_list(cls: type | None, text: str) -> tuple:
    """Deserialize a `dumps_list` envelope back into a tuple of ``cls`` instances."""
    envelope = json.loads(text)
    data = envelope["data"]
    if cls is None:
        return tuple(data)
    return tuple(decode(cls, item) for item in data)


def passthrough_dict(d: dict) -> dict:
    """Identity round-trip for an untyped, already-JSON-safe dict store return."""
    return json.loads(json.dumps(d))


def passthrough_dict_tuple(items: Any) -> tuple:
    """Identity round-trip for an untyped tuple-of-dicts store return."""
    return tuple(json.loads(json.dumps(list(items))))
