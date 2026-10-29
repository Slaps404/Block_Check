"""Single declarative table for the ``/rpc`` transport seam.

Each remote store operation is declared once here instead of being kept in
sync by hand across ``ProcessingStore``, the ``session/rpc_server.py``
dispatch/arity whitelists, and the ``RemoteProcessingStore`` proxy. Each
``RpcOp`` entry names a method, its wire parameter names (in argument-list
order), which trailing parameters have defaults, whether ``request_id`` is
forwarded into the store call and/or exposed on the proxy's own signature,
any encode/decode transforms needed for non-JSON-native values, and the
result codec used to validate and decode the response.

From this table:

* ``RPC_METHODS`` / ``RPC_ARITY`` (consumed by ``session/rpc_server.py``) are
  DERIVED. ``RPC_ARITY`` is computed as ``(required, len(params))``, so a
  store parameter cannot be added without the arity whitelist following it.
* ``BARE_DICT_TUPLE_METHODS`` is derived from ``bare_dict_tuple``.
* ``PROXY_OPS`` (consumed by ``transport/remote.py``'s ``__getattr__``) holds
  every op whose proxy method is safe to synthesize generically.

Three methods keep a hand-written proxy (``proxy_override=True``) because
they are not shaped like the rest:

* ``resume_session`` -- the proxy raises a custom ``ValueError`` when
  ``session_number is None`` (there is no "resume the latest session" over
  the wire); the dispatch side is still fully generic.
* ``summarize`` -- takes a ``SessionIdentity`` object, not a bare session
  number, and its dispatch does a nested ``store.summarize(store.
  resume_session(n))`` rather than forwarding ``n`` directly.
* ``record_slide_capture`` / ``recapture_hybrid_slide`` -- their proxy
  methods speak the binary framed-upload protocol (8-byte length prefix +
  JSON metadata + PNG bytes) over dedicated ``/slides`` /
  ``/recapture-slide`` routes, never the plain-JSON ``/rpc`` envelope; their
  dispatch lambdas ARE still generic-table-driven, since the server always
  decodes them from an ``args`` list either way.

This module is deliberately lighter than ``session.workflow``: it imports
only stdlib, ``transport.wire``, ``session.workflow_types`` (the lightweight
dataclass-only module), and ``slide.qr``. It must never import
``session.workflow`` or ``session.rpc_server`` -- both of those import
*this* module, so importing either back would be a cycle.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Mapping

import transport.wire as store_wire
from session.workflow_types import (
    BlockReadiness,
    ClaimOutcome,
    FailedBlockWarning,
    HybridPoolFreezeResult,
    SessionSummary,
    ScanOutcome,
    SessionIdentity,
    WorkflowEvent,
)
from slide.qr import SlideQRResult


# ---------------------------------------------------------------------------
# result codecs: how one op's response body is validated (inside the proxy's
# retry loop, before a partially-received response is trusted) and decoded
# (into the value the proxy call returns).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResultCodec:
    validate: Callable[[str, str], None]
    decode: Callable[[str], object]


def _codec_dataclass(cls: type) -> ResultCodec:
    """A single registered dataclass, self-describing envelope."""
    def validate(method: str, text: str) -> None:
        store_wire.loads_as(cls, text)

    def decode(text: str) -> object:
        return store_wire.loads_as(cls, text)

    return ResultCodec(validate, decode)


def _codec_list(cls: type) -> ResultCodec:
    """A homogeneous ``dumps_list`` envelope of a registered dataclass."""
    def validate(method: str, text: str) -> None:
        store_wire.loads_list(cls, text)

    def decode(text: str) -> object:
        return store_wire.loads_list(cls, text)

    return ResultCodec(validate, decode)


def _codec_bare_list() -> ResultCodec:
    """``awaiting_capture_blocks``: a ``dumps_list`` envelope of bare strings.

    Validation is parse-only (no shape check).
    """
    def validate(method: str, text: str) -> None:
        json.loads(text)

    def decode(text: str) -> object:
        return store_wire.loads_list(None, text)

    return ResultCodec(validate, decode)


def _codec_json(
    expected_type: Any = None, cast: Callable[[Any], Any] | None = None
) -> ResultCodec:
    """A bare JSON scalar/collection, optionally type-checked and/or cast.

    ``expected_type=None`` means parse-only validation with no shape
    assertion (used by ``unscan_block``/``has_work_orders``, whose decode
    still casts to ``bool``). This looser validation is intentional; adding
    a shape check there would change existing wire behavior.
    """
    def validate(method: str, text: str) -> None:
        value = json.loads(text)
        if expected_type is not None and not isinstance(value, expected_type):
            expected_names = (
                "/".join(t.__name__ for t in expected_type)
                if isinstance(expected_type, tuple) else expected_type.__name__
            )
            raise TypeError(
                f"{method} response must be {expected_names}, "
                f"got {type(value).__name__}"
            )

    def decode(text: str) -> object:
        value = json.loads(text)
        return cast(value) if cast is not None else value

    return ResultCodec(validate, decode)


def _codec_optional_int() -> ResultCodec:
    """``open_work_order_id``: ``int`` or ``None``, never cast-to-int a None."""
    def validate(method: str, text: str) -> None:
        value = json.loads(text)
        if not isinstance(value, (int, type(None))):
            raise TypeError(
                f"{method} response must be int/NoneType, got {type(value).__name__}"
            )

    def decode(text: str) -> object:
        value = json.loads(text)
        return int(value) if value is not None else None

    return ResultCodec(validate, decode)


def _codec_none() -> ResultCodec:
    """No meaningful return value; validate is parse-only, decode is unused."""
    def validate(method: str, text: str) -> None:
        json.loads(text)

    def decode(text: str) -> object:
        return None

    return ResultCodec(validate, decode)


_BOOL_TYPED = _codec_json(bool, cast=bool)
_BOOL_UNTYPED = _codec_json(None, cast=bool)
_INT_TYPED = _codec_json(int, cast=int)
_NONE = _codec_none()


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RpcOp:
    name: str
    params: tuple[str, ...] = ()
    uses_session: bool = True
    passes_request_id: bool = False
    proxy_request_id: bool = False
    encode_arg: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    decode_arg: Mapping[str, Callable[[Any], Any]] = field(default_factory=dict)
    defaults: Mapping[str, Any] = field(default_factory=dict)
    result_codec: ResultCodec = field(default_factory=_codec_none)
    dispatch_override: Callable[[Any, int, list, "str | None"], object] | None = None
    proxy_override: bool = False
    bare_dict_tuple: bool = False

    @property
    def required(self) -> int:
        """Count of leading params with no default (defaults are always trailing)."""
        return len(self.params) - len(self.defaults)

    @property
    def arity(self) -> tuple[int, int]:
        return (self.required, len(self.params))


def _summarize_dispatch(store: Any, n: int, args: list, request_id: "str | None") -> object:
    return store.summarize(store.resume_session(n))


RPC_TABLE: tuple[RpcOp, ...] = (
    RpcOp(
        "start_session",
        params=("started_at", "session_mode"),
        uses_session=False,
        passes_request_id=True,
        encode_arg={"started_at": store_wire.encode},
        decode_arg={"started_at": datetime.fromisoformat},
        defaults={"session_mode": "normal"},
        result_codec=_codec_dataclass(SessionIdentity),
    ),
    RpcOp(
        "resume_session",
        result_codec=_codec_dataclass(SessionIdentity),
        proxy_override=True,
    ),
    RpcOp(
        "scan_block",
        params=("block_id",),
        passes_request_id=True,
        proxy_request_id=True,
        result_codec=_codec_dataclass(ScanOutcome),
    ),
    RpcOp(
        "awaiting_capture_blocks",
        result_codec=_codec_bare_list(),
    ),
    RpcOp(
        "unscan_block",
        params=("block_id",),
        passes_request_id=True,
        result_codec=_BOOL_UNTYPED,
    ),
    RpcOp(
        "add_capture_note",
        params=("capture_id", "note", "block_id"),
        passes_request_id=True,
        defaults={"block_id": None},
        result_codec=_NONE,
    ),
    RpcOp("begin_block_drain", result_codec=_NONE),
    RpcOp("try_enter_slides", result_codec=_BOOL_TYPED),
    RpcOp(
        "freeze_hybrid_pool",
        params=("descriptor_names", "candidate_configuration"),
        passes_request_id=True,
        proxy_request_id=True,
        encode_arg={"descriptor_names": list},
        defaults={"candidate_configuration": None},
        result_codec=_codec_dataclass(HybridPoolFreezeResult),
    ),
    RpcOp("reconcile_session_metadata", result_codec=_NONE),
    RpcOp("begin_finalization", result_codec=_NONE),
    RpcOp("prepare_finalization", result_codec=_BOOL_TYPED),
    RpcOp("complete_finalization", result_codec=_BOOL_TYPED),
    RpcOp("wait_for_jobs", uses_session=False, result_codec=_NONE),
    RpcOp(
        "get_set",
        params=("block_id",),
        result_codec=_codec_json(dict),
    ),
    RpcOp(
        "precheck_slide_scan",
        params=("block_id",),
        result_codec=_BOOL_TYPED,
    ),
    RpcOp(
        "record_slide_capture",
        params=("source", "captured_at", "result", "duration_ms", "priority"),
        passes_request_id=True,
        decode_arg={
            "captured_at": datetime.fromisoformat,
            "result": lambda raw: store_wire.decode(SlideQRResult, raw),
        },
        defaults={"priority": None},
        proxy_override=True,
    ),
    RpcOp(
        "retry_hybrid_slide",
        params=("capture_id",),
        passes_request_id=True,
        proxy_request_id=True,
        result_codec=_BOOL_TYPED,
    ),
    RpcOp(
        "recapture_hybrid_slide",
        params=("superseded_capture_id", "source", "captured_at", "result", "duration_ms"),
        passes_request_id=True,
        decode_arg={
            "captured_at": datetime.fromisoformat,
            "result": lambda raw: store_wire.decode(SlideQRResult, raw),
        },
        proxy_override=True,
    ),
    RpcOp(
        "resolve_claim",
        params=("block_id", "slide_capture_id", "slide_path"),
        passes_request_id=True,
        proxy_request_id=True,
        encode_arg={"slide_path": str},
        result_codec=_codec_dataclass(ClaimOutcome),
    ),
    RpcOp(
        "slide_captures",
        result_codec=_codec_json(list, cast=tuple),
        bare_dict_tuple=True,
    ),
    RpcOp("slide_recovery_state", result_codec=_codec_json(str)),
    RpcOp("skip_unreadable_slide", passes_request_id=True, result_codec=_NONE),
    RpcOp("mark_waiting_for_slide", passes_request_id=True, result_codec=_NONE),
    RpcOp(
        "active_warnings",
        result_codec=_codec_list(FailedBlockWarning),
    ),
    RpcOp(
        "summarize",
        result_codec=_codec_dataclass(SessionSummary),
        dispatch_override=_summarize_dispatch,
        proxy_override=True,
    ),
    RpcOp(
        "dismiss_block",
        params=("block_id", "reason"),
        passes_request_id=True,
        result_codec=_NONE,
    ),
    RpcOp(
        "block_readiness",
        params=("block_id",),
        result_codec=_codec_dataclass(BlockReadiness),
    ),
    RpcOp("events", result_codec=_codec_list(WorkflowEvent)),
    RpcOp(
        "record_event",
        params=("kind", "message"),
        passes_request_id=True,
        proxy_request_id=True,
        result_codec=_NONE,
    ),
    RpcOp(
        "record_finalization_error",
        params=("message", "reconcile"),
        defaults={"reconcile": True},
        result_codec=_NONE,
    ),
    RpcOp(
        "record_profile_capture",
        params=("capture_id", "fields"),
        passes_request_id=True,
        encode_arg={"fields": dict},
        result_codec=_NONE,
    ),
    RpcOp(
        "record_slide_benchmark",
        params=("capture_id", "fields"),
        encode_arg={"fields": dict},
        result_codec=_NONE,
    ),
    RpcOp("start_work_order", result_codec=_INT_TYPED),
    RpcOp(
        "finish_work_order",
        params=("start_job",),
        defaults={"start_job": True},
        passes_request_id=True,
        result_codec=_INT_TYPED,
    ),
    RpcOp(
        "list_results_ready_work_orders",
        result_codec=_codec_json(list, cast=tuple),
        bare_dict_tuple=True,
    ),
    RpcOp("open_work_order_id", result_codec=_codec_optional_int()),
    RpcOp("has_work_orders", result_codec=_BOOL_UNTYPED),
    RpcOp(
        "list_hybrid_results",
        result_codec=_codec_json(list, cast=tuple),
        bare_dict_tuple=True,
    ),
    RpcOp(
        "list_retrieval_results",
        result_codec=_codec_json(list, cast=tuple),
        bare_dict_tuple=True,
    ),
    RpcOp(
        "list_hybrid_profile_rows",
        result_codec=_codec_json(list, cast=tuple),
        bare_dict_tuple=True,
    ),
)

RPC_TABLE_BY_NAME: dict[str, RpcOp] = {op.name: op for op in RPC_TABLE}

assert len(RPC_TABLE_BY_NAME) == len(RPC_TABLE), "duplicate RpcOp.name in RPC_TABLE"


# ---------------------------------------------------------------------------
# server side: derive _RPC_METHODS / _RPC_ARITY / _BARE_DICT_TUPLE_METHODS
# ---------------------------------------------------------------------------


def _build_dispatch(op: RpcOp) -> Callable[[Any, int, list, "str | None"], object]:
    if op.dispatch_override is not None:
        return op.dispatch_override

    def _dispatch(store: Any, n: int, args: list, request_id: "str | None") -> object:
        kwargs: dict[str, Any] = {}
        for index, name in enumerate(op.params):
            if index < len(args):
                raw = args[index]
                decoder = op.decode_arg.get(name)
                kwargs[name] = decoder(raw) if decoder is not None else raw
            else:
                kwargs[name] = op.defaults[name]
        if op.passes_request_id:
            kwargs["request_id"] = request_id
        call_args = (n,) if op.uses_session else ()
        return getattr(store, op.name)(*call_args, **kwargs)

    return _dispatch


RPC_METHODS: dict[str, Callable[[Any, int, list, "str | None"], object]] = {
    op.name: _build_dispatch(op) for op in RPC_TABLE
}

RPC_ARITY: dict[str, tuple[int, int]] = {op.name: op.arity for op in RPC_TABLE}

BARE_DICT_TUPLE_METHODS: frozenset[str] = frozenset(
    op.name for op in RPC_TABLE if op.bare_dict_tuple
)


# ---------------------------------------------------------------------------
# client side: generic proxy-call invocation for RemoteProcessingStore's
# __getattr__. Only ops with proxy_override=False are exposed here. The
# other four (resume_session, summarize, record_slide_capture,
# recapture_hybrid_slide) keep a real, hand-written method on the class, so
# normal attribute lookup finds those before __getattr__ is ever consulted.
# ---------------------------------------------------------------------------

PROXY_OPS: dict[str, RpcOp] = {
    op.name: op for op in RPC_TABLE if not op.proxy_override
}


def validate_rpc_payload(method: str, body: bytes) -> None:
    """Validate a successful /rpc response shape, still inside the retry loop.

    An unrecognized method name (never expected, since every proxied method
    comes from this table) falls back to a parse-only check rather than
    raising.
    """
    text = body.decode("utf-8")
    op = RPC_TABLE_BY_NAME.get(method)
    if op is None:
        json.loads(text)
        return
    op.result_codec.validate(method, text)


def invoke_proxy(
    store: Any,
    op: RpcOp,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    request_id: "str | None",
) -> object:
    """Build and issue one generic proxy call for ``op``.

    ``store`` is the ``RemoteProcessingStore`` instance (needs ``_rpc`` and
    ``_PLACEHOLDER_SESSION``). Accepts both calling conventions every current
    proxy method mixes: positional-by-order and keyword-by-name for each
    wire parameter.
    """
    if op.uses_session:
        if not args:
            raise TypeError(f"{op.name}() missing required argument: 'session_number'")
        session_number, *positional = args
    else:
        session_number, positional = store._PLACEHOLDER_SESSION, list(args)

    if len(positional) > len(op.params):
        raise TypeError(
            f"{op.name}() takes at most {len(op.params)} positional "
            f"argument(s) after session_number; received {len(positional)}"
        )

    values: dict[str, object] = dict(zip(op.params, positional))
    for key, value in kwargs.items():
        if key not in op.params:
            raise TypeError(f"{op.name}() got an unexpected keyword argument {key!r}")
        if key in values:
            raise TypeError(f"{op.name}() got multiple values for argument {key!r}")
        values[key] = value

    wire_args = []
    for name in op.params:
        if name in values:
            raw = values[name]
        elif name in op.defaults:
            raw = op.defaults[name]
        else:
            raise TypeError(f"{op.name}() missing required argument: {name!r}")
        encoder = op.encode_arg.get(name)
        wire_args.append(encoder(raw) if encoder is not None else raw)

    rpc_kwargs = {"request_id": request_id} if op.proxy_request_id else {}
    body = store._rpc(session_number, op.name, wire_args, **rpc_kwargs)
    return op.result_codec.decode(body.decode("utf-8"))
