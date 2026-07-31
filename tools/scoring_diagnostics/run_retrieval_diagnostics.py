"""CLI entry point for the offline approximate-retrieval experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (str(_ROOT / "code"), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from retrieval_evidence import (  # noqa: E402
    build_evidence,
    calibrate_cached_evidence,
    select_hybrid_handoff_inputs,
)
from retrieval_manifest import load_retrieval_manifest  # noqa: E402
from integration_handoff import build_integration_handoff, write_integration_handoff  # noqa: E402
from verify.invariant_descriptors import descriptor_catalog  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or analyze retrieval diagnostic evidence"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    build = commands.add_parser("build", help="prepare specimens and update raw evidence")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--evidence", required=True, type=Path)
    calibrate = commands.add_parser(
        "calibrate", help="analyze existing evidence without image work"
    )
    calibrate.add_argument("--evidence", required=True, type=Path)
    calibrate.add_argument("--report", type=Path)
    calibrate.add_argument(
        "--handoff", type=Path,
        help="also emit the #249 versioned Hybrid integration handoff to this "
        "path (synthetic/proof-of-concept only; requires >=2 work orders in "
        "the evidence)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.operation == "build":
        result = build_evidence(load_retrieval_manifest(args.manifest), args.evidence)
    else:
        kwargs = {"report_path": args.report} if args.report else {}
        result = calibrate_cached_evidence(args.evidence, **kwargs)
        if args.handoff:
            evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
            inputs = select_hybrid_handoff_inputs(evidence)
            handoff = build_integration_handoff(
                architecture=inputs["architecture"],
                descriptor_catalog=descriptor_catalog(),
                candidate_band_thresholds=inputs["thresholds"],
                veto=inputs["veto"],
                candidate_evidence=inputs["candidate_evidence"],
                efficiency=inputs["efficiency"],
                known_misses=inputs["known_misses"],
                weak_stratum=inputs["weak_stratum"],
                provenance=inputs["provenance"],
                calibration_run_id=hashlib.sha256(
                    Path(args.evidence).read_bytes()
                ).hexdigest(),
            )
            write_integration_handoff(args.handoff, handoff)
            result["handoff_path"] = str(args.handoff)
    print(json.dumps(result, default=str, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
