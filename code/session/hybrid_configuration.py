"""Versioned Hybrid Configuration loaded from #242's integration handoff (#249).

CONTEXT.md "Hybrid Configuration": missing or incompatible configuration must
prevent Hybrid startup, loudly, before capture. This module is the production
side of that contract. It reads and validates the JSON artifact
`tools/scoring_diagnostics/integration_handoff.py` serializes -- it never
imports that (or any other) diagnostics module, so the tools/ -> code/
dependency arrow enforced by `tests/test_architecture_boundaries.py` stays
one-way.

Split mirrors `code/verify/work_order_evaluator.py`: :func:`parse_hybrid_configuration`
is pure data-in/dataclass-out (a JSON-shaped `Mapping` in, `HybridConfiguration`
out, no I/O); :func:`load_hybrid_configuration` is the thin file-reading shell
around it, the only place this module touches disk.

Scope note (#249): this module only answers "may Hybrid/Hybrid Shadow start?".
The isolated-failure-after-startup fallback to complete N^2 accurate scoring
(CONTEXT.md "Hybrid Configuration", #245 user story 53) is queue/scoring-slice
work for a later issue and is deliberately not implemented here.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Optional


# The one schema version this build of production code understands. An older
# or newer handoff is rejected by name (see `parse_hybrid_configuration`)
# rather than guessed-compatible, so a stale or from-the-future handoff can
# never silently pass as a working Hybrid configuration.
HYBRID_CONFIGURATION_SCHEMA_VERSION = 1

# Architecture label vocabulary. This intentionally duplicates
# `tools/scoring_diagnostics/candidate_retrieval_analysis.ArchitectureKind`'s
# string values rather than importing that enum: the retrieval *search* that
# produces one of these labels is diagnostic-only, but production still needs
# to recognize which labels are legal without crossing the one-way import
# boundary (#245/#246).
_ARCHITECTURE_KINDS = frozenset({"individual", "fusion", "union", "router"})

# Safety-critical fallback behaviors #245 requires of any Hybrid deployment
# (CONTEXT.md "Hybrid Configuration"; user stories 53/54/55). The handoff must
# explicitly declare all three -- production does not assume them silently,
# because an omission here would otherwise be invisible until an incident.
REQUIRED_FALLBACK_IDS = frozenset({
    "complete_accurate_scoring_on_missing_fingerprint",
    "complete_accurate_scoring_on_invalid_candidate_output",
    "claim_inserted_separately_from_candidate_band",
})

# Production-owned files whose content the calibrated handoff is only valid
# against. Deliberately excludes the diagnostics-only "normalization" entry
# that `tools/scoring_diagnostics/retrieval_evidence.py` also records in its
# own provenance -- that file lives under `tools/` and is not on the
# production Hybrid scoring path, so production neither reads nor hashes it.
_PRODUCTION_IMPLEMENTATION_FILES = {
    "preparation": ("session", "preparation.py"),
    "descriptors": ("verify", "invariant_descriptors.py"),
    "gates": ("verify", "gates.py"),
    "scorer": ("verify", "scorer.py"),
    "evaluator": ("verify", "work_order_evaluator.py"),
}

# The one status literal `tools/scoring_diagnostics/integration_handoff
# .PROOF_OF_CONCEPT_STATUS` emits today. #245 treats "may Hybrid start" and
# "is Hybrid production-approved" as separate questions; this allowlist is
# the loader's half of that distinction. Promoting Hybrid to production
# later means deliberately adding a new literal here -- a conscious code
# review event -- not silently accepting whatever string a hand-edited or
# future handoff happens to carry.
_ALLOWED_HANDOFF_STATUSES = frozenset({
    "proof_of_concept_not_production_approved",
})

_REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "architecture",
    "descriptor_recipe",
    "candidate_band_thresholds",
    "veto",
    "candidate_evidence",
    "known_misses",
    "weak_stratum",
    "required_fallbacks",
    "provenance",
    "status",
)
_REQUIRED_ARCHITECTURE_FIELDS = ("kind", "name", "methods")
_REQUIRED_DESCRIPTOR_FIELDS = ("name", "version", "dimension", "comparison")
_REQUIRED_VETO_FIELDS = ("enabled", "threshold", "reason")
# Runtime/reduction fields are deliberately excluded (F8): #242's
# `retrieval_evidence._efficiency_summary` can legitimately have no runtime
# measurement, so those three stay optional (`.get(..., default=None)`
# below) rather than required. The count fields below are always computed
# by `BandEvaluation`'s properties, so a typo'd key here is a real defect,
# not a legitimate omission.
_REQUIRED_CANDIDATE_EVIDENCE_FIELDS = (
    "mean_candidate_count", "median_candidate_count",
    "p95_candidate_count", "max_candidate_count",
)
_REQUIRED_PROVENANCE_FIELDS = (
    "manifest_path", "manifest_hash", "code_revision",
    "implementation_hashes", "calibration_run_id",
)


class HybridConfigurationError(Exception):
    """Raised when a Hybrid integration handoff cannot safely start Hybrid.

    Every raise site names the concrete problem (missing file, malformed
    JSON, schema-version mismatch naming expected vs. found, stale
    provenance, an unknown descriptor, a malformed field that failed a
    coercion, ...) so a caller can print the message directly and exit
    non-zero before any capture side effect.

    Deliberately does NOT subclass `ValueError` (or any other builtin):
    every numeric/iterable coercion in `parse_hybrid_configuration` is
    guarded so a malformed payload always raises exactly this type, never a
    bare `ValueError`/`TypeError` escaping from an unguarded `int()`/
    `float()`/iteration. Inheriting from `ValueError` would let this type be
    caught unintentionally by an unrelated `except ValueError` elsewhere in
    the codebase; nothing in this repo relies on that inheritance (grepped
    for `except ValueError` at #249's review), so there is no reason to keep
    it.
    """


@dataclass(frozen=True)
class DescriptorRecipeEntry:
    name: str
    version: str
    dimension: int
    comparison: str
    prior_evidence: str = ""


@dataclass(frozen=True)
class VetoConfiguration:
    enabled: bool
    threshold: Optional[float]
    reason: str


@dataclass(frozen=True)
class CandidateEvidence:
    mean_candidate_count: float
    median_candidate_count: float
    p95_candidate_count: int
    max_candidate_count: int
    observed_runtime_seconds: Optional[float]
    estimated_runtime_seconds: Optional[float]
    full_comparison_reduction: Optional[float]


@dataclass(frozen=True)
class WeakStratum:
    field: str
    group: str
    coverage: Optional[float]


@dataclass(frozen=True)
class HandoffProvenance:
    manifest_path: str
    manifest_hash: str
    code_revision: str
    implementation_hashes: Mapping[str, str]
    calibration_run_id: str


@dataclass(frozen=True)
class HybridConfiguration:
    """The one versioned, provenance-linked Hybrid Configuration for a session."""

    schema_version: int
    architecture_kind: str
    architecture_name: str
    architecture_methods: tuple[str, ...]
    descriptor_recipe: tuple[DescriptorRecipeEntry, ...]
    candidate_band_thresholds: Mapping[str, float]
    veto: VetoConfiguration
    candidate_evidence: CandidateEvidence
    known_misses: tuple[str, ...]
    weak_stratum: Optional[WeakStratum]
    required_fallbacks: tuple[str, ...]
    provenance: HandoffProvenance
    status: str
    source_path: str = "<memory>"
    extra_fields: Mapping[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "unavailable"


def current_implementation_hashes() -> dict[str, str]:
    """Hash the production files the handoff's provenance must still match.

    Public (not underscore-prefixed): the loader uses it to detect a stale
    handoff, and both the CLI that authors synthetic fixtures and tests that
    build a compatible synthetic handoff call it directly, so nobody
    hand-copies file paths or hashing logic a second time.
    """
    code_root = Path(__file__).resolve().parents[1]
    return {
        name: _sha256(code_root.joinpath(*parts))
        for name, parts in _PRODUCTION_IMPLEMENTATION_FILES.items()
    }


def known_descriptor_names() -> frozenset[str]:
    """The production Heuristic Descriptor Catalog's names (#246 owns the catalog)."""
    from verify.invariant_descriptors import descriptor_catalog

    return frozenset(spec.name for spec in descriptor_catalog())


def _require_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HybridConfigurationError(f"Hybrid configuration {what} must be a JSON object")
    return value


def _require_fields(payload: Mapping[str, Any], fields_: tuple[str, ...], what: str) -> None:
    missing = [name for name in fields_ if name not in payload]
    if missing:
        raise HybridConfigurationError(
            f"Hybrid configuration {what} missing required field(s): {', '.join(missing)}"
        )


def _require_sequence(value: Any, what: str) -> Collection[Any]:
    """Reject a JSON string/object/scalar masquerading as a JSON array (F1).

    `str`/`bytes` and `Mapping` both satisfy `isinstance(value, Collection)`
    too, so a bare `Collection` check would silently iterate a string's
    characters or a mapping's keys instead of rejecting it -- and a bare
    scalar (e.g. an `int`) would raise an unguarded `TypeError` from the
    caller's `for`/`enumerate` instead of a clear `HybridConfigurationError`.
    """
    if isinstance(value, (str, bytes)) or isinstance(value, Mapping) or not isinstance(
        value, Collection
    ):
        raise HybridConfigurationError(
            f"Hybrid configuration {what} must be a JSON array, found {value!r}"
        )
    return value


def _require_number(value: Any, what: str) -> float:
    """Reject bool/str/other non-numeric JSON values masquerading as a number.

    Python's `float()`/`int()` constructors silently coerce numeric strings
    and `bool` (a subtype of `int`), which is exactly the "value means its
    opposite" class of bug #249's review flagged (F5): a JSON string
    threshold, or `true` flowing in as `1`, must be rejected outright, not
    coerced. Used for the safety-critical numeric fields (candidate-band
    thresholds, veto.threshold) where a silently-coerced value can change
    Hybrid's behavior; the more permissive `_coerce_float`/`_coerce_int`
    below is used where the only concern is not crashing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HybridConfigurationError(
            f"Hybrid configuration {what} must be a JSON number, found {value!r}"
        )
    return float(value)


def _coerce_float(value: Any, what: str) -> float:
    """`float(value)`, turning a malformed value into `HybridConfigurationError`
    instead of a bare `ValueError`/`TypeError` escaping to the caller (F1)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HybridConfigurationError(
            f"Hybrid configuration {what} must be a number, found {value!r}"
        ) from None


def _coerce_int(value: Any, what: str) -> int:
    """`int(value)`, turning a malformed value into `HybridConfigurationError`
    instead of a bare `ValueError`/`TypeError` escaping to the caller (F1)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        raise HybridConfigurationError(
            f"Hybrid configuration {what} must be an integer, found {value!r}"
        ) from None


def _optional_float(value: Any, what: str) -> Optional[float]:
    return None if value is None else _coerce_float(value, what)


def parse_hybrid_configuration(
    payload: Mapping[str, Any],
    *,
    known_descriptor_names: Collection[str],
    current_implementation_hashes: Mapping[str, str],
    source_path: str = "<memory>",
) -> HybridConfiguration:
    """Validate one decoded JSON handoff and return the pure `HybridConfiguration`.

    Extra/unrecognized top-level or nested keys are tolerated (forward
    compatible): a diagnostic-side addition of a new informational field must
    not itself block a deployed Hybrid station. Every field listed in
    `_REQUIRED_TOP_LEVEL_FIELDS` (and the nested contracts below) is REQUIRED
    -- its absence, a schema-version mismatch, a stale provenance hash, or an
    unknown descriptor name all raise :class:`HybridConfigurationError`
    naming exactly what is wrong.
    """
    payload = _require_mapping(payload, "payload")
    _require_fields(payload, _REQUIRED_TOP_LEVEL_FIELDS, "payload")

    found_version = payload["schema_version"]
    if isinstance(found_version, bool) or not isinstance(found_version, int):
        raise HybridConfigurationError(
            "Hybrid configuration schema_version must be a JSON integer, found "
            f"{found_version!r} (source: {source_path})"
        )
    if found_version != HYBRID_CONFIGURATION_SCHEMA_VERSION:
        raise HybridConfigurationError(
            "Hybrid configuration schema version mismatch: expected "
            f"{HYBRID_CONFIGURATION_SCHEMA_VERSION!r}, found {found_version!r} "
            f"(source: {source_path})"
        )

    architecture = _require_mapping(payload["architecture"], "architecture")
    _require_fields(architecture, _REQUIRED_ARCHITECTURE_FIELDS, "architecture")
    kind = str(architecture["kind"])
    if kind not in _ARCHITECTURE_KINDS:
        raise HybridConfigurationError(
            f"Hybrid configuration architecture.kind {kind!r} is not one of "
            f"{sorted(_ARCHITECTURE_KINDS)}"
        )
    raw_methods = _require_sequence(architecture["methods"], "architecture.methods")
    methods = tuple(str(method) for method in raw_methods)
    if not methods:
        raise HybridConfigurationError(
            "Hybrid configuration architecture.methods must name at least one descriptor"
        )

    raw_recipe = _require_sequence(payload["descriptor_recipe"], "descriptor_recipe")
    if not raw_recipe:
        raise HybridConfigurationError(
            "Hybrid configuration descriptor_recipe must name at least one "
            "descriptor -- an empty recipe would freeze a Hybrid Candidate "
            "Pool with zero comparable fingerprints per block"
        )
    recipe_entries = []
    for index, raw_entry in enumerate(raw_recipe):
        entry = _require_mapping(raw_entry, f"descriptor_recipe[{index}]")
        _require_fields(entry, _REQUIRED_DESCRIPTOR_FIELDS, f"descriptor_recipe[{index}]")
        recipe_entries.append(DescriptorRecipeEntry(
            name=str(entry["name"]),
            version=str(entry["version"]),
            dimension=_coerce_int(entry["dimension"], f"descriptor_recipe[{index}].dimension"),
            comparison=str(entry["comparison"]),
            prior_evidence=str(entry.get("prior_evidence", "")),
        ))
    descriptor_recipe = tuple(recipe_entries)

    referenced_names = {*methods, *(entry.name for entry in descriptor_recipe)}
    unknown = sorted(name for name in referenced_names if name not in known_descriptor_names)
    if unknown:
        raise HybridConfigurationError(
            "Hybrid configuration names descriptor(s) absent from the production "
            f"Heuristic Descriptor Catalog: {', '.join(unknown)}"
        )

    # F2: candidate_band_thresholds contents. This is the pruning-safety
    # lever -- a NaN/zero/negative threshold makes every `score >= threshold`
    # comparison false, silently selecting ZERO candidates. Every present
    # value must be a real (non-bool, non-string) finite positive number,
    # and the per-ArchitectureKind key set required is NOT "one threshold per
    # architecture.methods name" for every kind: `candidate_retrieval_analysis
    # ._threshold_keys` keys a FUSION architecture's thresholds by the
    # literal "fusion" alone, never by its member method names, while
    # INDIVIDUAL/UNION/ROUTER all key by every name in `architecture.methods`
    # (verified by reading `_threshold_keys` directly, not inferred).
    thresholds_raw = _require_mapping(
        payload["candidate_band_thresholds"], "candidate_band_thresholds"
    )
    if not thresholds_raw:
        raise HybridConfigurationError(
            "Hybrid configuration candidate_band_thresholds must not be empty"
        )
    candidate_band_thresholds: dict[str, float] = {}
    for raw_key, raw_value in thresholds_raw.items():
        key = str(raw_key)
        value = _require_number(raw_value, f"candidate_band_thresholds[{key!r}]")
        if not math.isfinite(value):
            raise HybridConfigurationError(
                f"Hybrid configuration candidate_band_thresholds[{key!r}] must be "
                f"finite, found {value!r}"
            )
        if value <= 0:
            raise HybridConfigurationError(
                f"Hybrid configuration candidate_band_thresholds[{key!r}] must be "
                f"greater than 0, found {value!r}"
            )
        candidate_band_thresholds[key] = value
    required_threshold_names = (
        frozenset({"fusion"}) if kind == "fusion" else frozenset(methods)
    )
    missing_thresholds = sorted(required_threshold_names - set(candidate_band_thresholds))
    if missing_thresholds:
        raise HybridConfigurationError(
            "Hybrid configuration candidate_band_thresholds is missing threshold(s) "
            f"required by architecture.kind {kind!r}: {', '.join(missing_thresholds)}"
        )

    veto_raw = _require_mapping(payload["veto"], "veto")
    _require_fields(veto_raw, _REQUIRED_VETO_FIELDS, "veto")
    veto_enabled = veto_raw["enabled"]
    if not isinstance(veto_enabled, bool):
        raise HybridConfigurationError(
            f"Hybrid configuration veto.enabled must be a JSON boolean, found "
            f"{veto_enabled!r}"
        )
    raw_veto_threshold = veto_raw["threshold"]
    veto_threshold = (
        None if raw_veto_threshold is None
        else _require_number(raw_veto_threshold, "veto.threshold")
    )
    # F3: nothing consumes `config.veto` yet, but an enabled veto with a
    # null/non-finite/non-positive threshold is a latent trap -- every
    # `score >= veto.threshold` comparison it would drive must be meaningful
    # the moment Heuristic Veto starts reading this field.
    if veto_enabled and (
        veto_threshold is None or not math.isfinite(veto_threshold) or veto_threshold <= 0
    ):
        raise HybridConfigurationError(
            "Hybrid configuration veto.enabled is true but veto.threshold "
            f"({raw_veto_threshold!r}) is not a finite number greater than 0"
        )
    veto = VetoConfiguration(
        enabled=veto_enabled,
        threshold=veto_threshold,
        reason=str(veto_raw["reason"]),
    )

    candidate_evidence_raw = _require_mapping(
        payload["candidate_evidence"], "candidate_evidence"
    )
    _require_fields(
        candidate_evidence_raw, _REQUIRED_CANDIDATE_EVIDENCE_FIELDS, "candidate_evidence"
    )
    candidate_evidence = CandidateEvidence(
        mean_candidate_count=_coerce_float(
            candidate_evidence_raw["mean_candidate_count"],
            "candidate_evidence.mean_candidate_count",
        ),
        median_candidate_count=_coerce_float(
            candidate_evidence_raw["median_candidate_count"],
            "candidate_evidence.median_candidate_count",
        ),
        p95_candidate_count=_coerce_int(
            candidate_evidence_raw["p95_candidate_count"],
            "candidate_evidence.p95_candidate_count",
        ),
        max_candidate_count=_coerce_int(
            candidate_evidence_raw["max_candidate_count"],
            "candidate_evidence.max_candidate_count",
        ),
        observed_runtime_seconds=_optional_float(
            candidate_evidence_raw.get("observed_runtime_seconds"),
            "candidate_evidence.observed_runtime_seconds",
        ),
        estimated_runtime_seconds=_optional_float(
            candidate_evidence_raw.get("estimated_runtime_seconds"),
            "candidate_evidence.estimated_runtime_seconds",
        ),
        full_comparison_reduction=_optional_float(
            candidate_evidence_raw.get("full_comparison_reduction"),
            "candidate_evidence.full_comparison_reduction",
        ),
    )

    known_misses = tuple(
        str(item) for item in _require_sequence(payload["known_misses"], "known_misses")
    )

    weak_stratum_raw = payload["weak_stratum"]
    weak_stratum = None
    if weak_stratum_raw is not None:
        weak_stratum_raw = _require_mapping(weak_stratum_raw, "weak_stratum")
        _require_fields(weak_stratum_raw, ("field", "group", "coverage"), "weak_stratum")
        weak_stratum = WeakStratum(
            field=str(weak_stratum_raw["field"]),
            group=str(weak_stratum_raw["group"]),
            coverage=_optional_float(weak_stratum_raw["coverage"], "weak_stratum.coverage"),
        )

    required_fallbacks = tuple(
        str(item)
        for item in _require_sequence(payload["required_fallbacks"], "required_fallbacks")
    )
    missing_fallbacks = sorted(REQUIRED_FALLBACK_IDS - set(required_fallbacks))
    if missing_fallbacks:
        raise HybridConfigurationError(
            "Hybrid configuration is missing required safety fallback "
            f"declaration(s): {', '.join(missing_fallbacks)}"
        )

    provenance_raw = _require_mapping(payload["provenance"], "provenance")
    _require_fields(provenance_raw, _REQUIRED_PROVENANCE_FIELDS, "provenance")
    handoff_hashes = _require_mapping(
        provenance_raw["implementation_hashes"], "provenance.implementation_hashes"
    )
    stale = sorted(
        name for name, expected_hash in current_implementation_hashes.items()
        if handoff_hashes.get(name) != expected_hash
    )
    if stale:
        raise HybridConfigurationError(
            "Hybrid configuration provenance is stale: implementation hash(es) "
            f"for {', '.join(stale)} no longer match the installed production "
            "code; recalibrate #242 and reissue the integration handoff "
            f"(source: {source_path})"
        )
    provenance = HandoffProvenance(
        manifest_path=str(provenance_raw["manifest_path"]),
        manifest_hash=str(provenance_raw["manifest_hash"]),
        code_revision=str(provenance_raw["code_revision"]),
        implementation_hashes={str(k): str(v) for k, v in handoff_hashes.items()},
        calibration_run_id=str(provenance_raw["calibration_run_id"]),
    )

    # F4: a hand-edited handoff must not be able to claim production
    # approval merely by typing a different string into `status`.
    status = str(payload["status"])
    if status not in _ALLOWED_HANDOFF_STATUSES:
        raise HybridConfigurationError(
            f"Hybrid configuration status {status!r} is not one of "
            f"{sorted(_ALLOWED_HANDOFF_STATUSES)}"
        )

    known_keys = set(_REQUIRED_TOP_LEVEL_FIELDS)
    extra_fields = {key: payload[key] for key in payload if key not in known_keys}

    return HybridConfiguration(
        schema_version=found_version,
        architecture_kind=kind,
        architecture_name=str(architecture["name"]),
        architecture_methods=methods,
        descriptor_recipe=descriptor_recipe,
        candidate_band_thresholds=candidate_band_thresholds,
        veto=veto,
        candidate_evidence=candidate_evidence,
        known_misses=known_misses,
        weak_stratum=weak_stratum,
        required_fallbacks=required_fallbacks,
        provenance=provenance,
        status=status,
        source_path=source_path,
        extra_fields=extra_fields,
    )


def _reject_non_finite_json_constant(token: str) -> float:
    """`json.loads`'s ``parse_constant`` hook (F2 second line of defense).

    Python's `json` module accepts the non-standard bare literals `NaN`,
    `Infinity`, and `-Infinity` by default, decoding them straight into
    `float('nan')`/`float('inf')`/`float('-inf')` -- so a hand-edited or
    corrupted handoff file can carry a non-finite candidate-band threshold
    that round-trips through JSON without ever looking malformed at the text
    level. `parse_hybrid_configuration`'s own `math.isfinite` checks are the
    primary defense (they run on the already-decoded payload regardless of
    call path); this hook rejects the literal at the JSON layer itself, for
    every numeric field in the file, before a payload dict even exists.
    """
    raise HybridConfigurationError(
        f"Hybrid configuration contains the non-finite JSON literal {token!r}; "
        "all numeric fields must be finite"
    )


def load_hybrid_configuration(path: str | Path) -> HybridConfiguration:
    """Read, decode, and validate the Hybrid Configuration handoff at ``path``.

    The only I/O in this module. Missing file and malformed JSON each raise
    :class:`HybridConfigurationError` with the offending path, exactly like
    every validation failure inside :func:`parse_hybrid_configuration`, so a
    caller (see `tools/run_pi_session.py::main`) can catch one exception type
    and exit non-zero before touching the store or camera.
    """
    source = Path(path)
    if not source.is_file():
        raise HybridConfigurationError(
            f"Hybrid configuration file not found: {source} -- Hybrid and Hybrid "
            "Shadow cannot start without a versioned #242 integration handoff"
        )
    try:
        raw = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise HybridConfigurationError(
            f"Hybrid configuration could not be read at {source}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw, parse_constant=_reject_non_finite_json_constant)
    except json.JSONDecodeError as exc:
        raise HybridConfigurationError(
            f"Hybrid configuration at {source} is not valid JSON: {exc}"
        ) from exc
    return parse_hybrid_configuration(
        payload,
        known_descriptor_names=known_descriptor_names(),
        current_implementation_hashes=current_implementation_hashes(),
        source_path=str(source),
    )
