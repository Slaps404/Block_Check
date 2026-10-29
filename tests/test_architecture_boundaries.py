"""Static dependency boundaries for deployed production code."""

import ast
import inspect
from pathlib import Path

from session.processing_store import ProcessingStore
from transport import rpc_table
from transport.remote import RemoteProcessingStore


def test_code_does_not_import_tools_or_archive():
    code_dir = Path(__file__).resolve().parents[1] / "code"
    violations = []
    for path in code_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if (
                    name == "tools" or name.startswith("tools.")
                    or name == "archive" or name.startswith("archive.")
                ):
                    violations.append(f"{path.name}:{node.lineno} imports {name}")
    assert not violations, "\n".join(violations)


def _diagnostics_experiment_dirs(tools_dir: Path) -> list[Path]:
    """Directories under `tools/` that hold diagnostics-experiment modules.

    `tests/conftest.py` puts several `tools/<subdir>` folders on `sys.path`.
    Of those, the diagnostics/experiment ones (as opposed to the operational
    `capture/`, `manifest/`, `identity/` tooling) are what production must
    never depend on. Matching on the "diagnostic" substring rather than a
    hardcoded list picks up new diagnostics directories automatically.
    """
    if not tools_dir.is_dir():
        return []
    return sorted(
        path
        for path in tools_dir.iterdir()
        if path.is_dir() and "diagnostic" in path.name.lower()
    )


def _diagnostics_experiment_module_owners(tools_dir: Path) -> dict[str, str]:
    """Map each diagnostics-experiment module basename to its owning dir name.

    Same source of truth as `_diagnostics_experiment_module_names`, but keeps
    the owning directory (e.g. "scoring_diagnostics") so violation messages
    can name the actual `tools/<dir>` the forbidden import came from.
    """
    owners: dict[str, str] = {}
    for diagnostics_dir in _diagnostics_experiment_dirs(tools_dir):
        for path in diagnostics_dir.rglob("*.py"):
            if path.stem != "__init__":
                owners.setdefault(path.stem, diagnostics_dir.name)
    return owners


def _diagnostics_experiment_module_names(tools_dir: Path) -> set[str]:
    """Bare module basenames that live under a diagnostics-experiment dir.

    These are exactly the names a production file could shadow-import by
    bare name (`from candidate_retrieval_analysis import ...`) thanks to the
    sys.path append in conftest.py, bypassing the `tools.`-prefix check in
    `test_code_does_not_import_tools_or_archive`. Rebuilt from disk on every
    run.
    """
    return set(_diagnostics_experiment_module_owners(tools_dir))


def _imported_top_level_names(tree: ast.AST):
    """Yield (imported_name, lineno) for each absolute import in `tree`.

    Covers `import x`, `import x.y`, and `from x import ...` / `from x.y
    import ...`. Relative imports (`from . import x`, level > 0) are
    excluded: they resolve within the importing package itself and cannot
    shadow-resolve to an unrelated `tools/` module on sys.path.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, node.lineno


def _find_diagnostics_import_violations(
    code_dir: Path, forbidden_owners: dict[str, str]
) -> list[str]:
    """Scan every `*.py` under `code_dir` for an import of a forbidden name.

    `forbidden_owners` maps a forbidden module basename to the `tools/<dir>`
    it lives under, so the violation message can name its source package.
    Parameterized (rather than reading real repo paths) so the detection
    logic can be unit-tested against a synthetic temp directory.
    """
    violations = []
    for path in sorted(code_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, lineno in _imported_top_level_names(tree):
            top_level = name.split(".", 1)[0]
            owner_dir = forbidden_owners.get(top_level)
            if owner_dir is not None:
                violations.append(
                    f"{path.name}:{lineno} imports {name!r}, which is a "
                    f"diagnostics-experiment module from tools/{owner_dir} -- "
                    f"the dependency arrow between tools/ and code/ is one-way "
                    f"(diagnostics may import production, production may never "
                    f"import diagnostics)"
                )
    return violations


def test_code_does_not_import_diagnostics_experiment_modules_by_bare_name():
    """No production module may import a diagnostics-experiment module.

    `tests/conftest.py` puts the diagnostics dirs on sys.path, so their
    modules are importable by bare name (e.g.
    `from candidate_retrieval_analysis import worst_subgroup`).
    `test_code_does_not_import_tools_or_archive` only catches the
    `tools.`-prefixed form, so without this test a bare-name import would
    silently invert the one-way tools -> code dependency arrow.
    """
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_owners = _diagnostics_experiment_module_owners(repo_root / "tools")
    violations = _find_diagnostics_import_violations(repo_root / "code", forbidden_owners)
    assert not violations, "\n".join(violations)


def test_invariant_descriptor_catalog_is_production_owned_not_duplicated_in_diagnostics():
    """The Heuristic Descriptor Catalog is production-owned (`code/retrieval`).

    Diagnostic tooling under `tools/scoring_diagnostics` must import the catalog
    from production rather than keeping its own copy of the math, so the
    dependency arrow points diagnostics -> production and never the reverse.
    """
    repo_root = Path(__file__).resolve().parents[1]
    production_module = repo_root / "code" / "retrieval" / "invariant_descriptors.py"
    diagnostics_module = (
        repo_root / "tools" / "scoring_diagnostics" / "invariant_descriptors.py"
    )
    assert production_module.is_file(), (
        "invariant descriptor catalog must live under code/retrieval"
    )
    assert not diagnostics_module.is_file(), (
        "tools/scoring_diagnostics must not keep its own copy of the descriptor catalog"
    )

    retrieval_evidence = repo_root / "tools" / "scoring_diagnostics" / "retrieval_evidence.py"
    tree = ast.parse(
        retrieval_evidence.read_text(encoding="utf-8"), filename=str(retrieval_evidence)
    )
    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert "retrieval.invariant_descriptors" in imported_modules, (
        "diagnostics must import the descriptor catalog from the production module"
    )
    assert "invariant_descriptors" not in imported_modules, (
        "diagnostics must not import a bare/local invariant_descriptors module"
    )


# `summarize` takes a `SessionIdentity`, not a bare session number: the server
# dispatch resolves it with a nested `resume_session(n)` call, so its table
# entry deliberately does not mirror the store signature.
_STORE_SIGNATURE_EXEMPT = frozenset({"summarize"})

# Store parameters this table entry deliberately does not carry. Both methods
# are `proxy_override=True`: the Pi reaches them over the dedicated binary
# `/slides` and `/recapture-slide` routes, not the plain-JSON `/rpc` envelope,
# so `source_token`, `priority` and `profile` travel in the framed metadata
# dict (`rpc_server.py` reads them back with `metadata.get(...)`) rather than
# in the positional `args` list this table describes. They are reachable over
# the wire; they are just not this table's business.
#
# `start_job` is the one genuinely unreachable parameter: nothing sends it for
# `record_slide_capture` on either route, so a remote caller always gets the
# default `True`. Listing these here makes each gap an explicit decision.
_NOT_THREADED_OVER_RPC = {
    "record_slide_capture": ("source_token", "start_job", "profile"),
    "recapture_hybrid_slide": ("source_token",),
}


def test_rpc_table_is_the_single_source_for_dispatch_arity_and_proxy():
    """The `/rpc` contract lives in exactly one table.

    `RPC_METHODS` and `RPC_ARITY` are derived from `RPC_TABLE`, and the
    `RemoteProcessingStore` proxy is synthesized from it via `__getattr__`.
    The one seam still hand-maintained is whether each table entry matches
    the real `ProcessingStore` method it forwards to.
    """
    assert set(rpc_table.RPC_METHODS) == set(rpc_table.RPC_ARITY)

    for op in rpc_table.RPC_TABLE:
        required = sum(1 for name in op.params if name not in op.defaults)
        assert op.arity == (required, len(op.params)), (
            f"{op.name}: arity {op.arity} disagrees with params/defaults"
        )

    # Every op is reachable on the proxy: either synthesized from the table
    # or deliberately hand-written for a non-`/rpc` route.
    for op in rpc_table.RPC_TABLE:
        assert op.name in rpc_table.PROXY_OPS or hasattr(
            RemoteProcessingStore, op.name
        ), f"{op.name} has neither a table-synthesized nor a hand-written proxy"


def test_rpc_table_params_match_the_real_store_method_signatures():
    """A new `ProcessingStore` parameter must be threaded through the RPC table.

    Comparing the table to the real store signature (not to other copies of
    the contract) is what catches a parameter that was never forwarded.
    """
    mismatches = []
    for op in rpc_table.RPC_TABLE:
        if op.name in _STORE_SIGNATURE_EXEMPT:
            continue
        method = getattr(ProcessingStore, op.name, None)
        if method is None:
            continue
        parameters = list(inspect.signature(method).parameters)
        assert parameters and parameters[0] == "self"
        expected = parameters[1:]
        if op.uses_session:
            if not expected or expected[0] != "session_number":
                mismatches.append(
                    f"{op.name}: table says uses_session but store takes {expected}"
                )
                continue
            expected = expected[1:]
        expected = [name for name in expected if name != "request_id"]
        expected = [
            name
            for name in expected
            if name not in _NOT_THREADED_OVER_RPC.get(op.name, ())
        ]
        if list(op.params) != expected:
            mismatches.append(
                f"{op.name}: table params {list(op.params)} != store params {expected}"
            )
    assert not mismatches, "RPC table drifted from ProcessingStore:\n" + "\n".join(
        mismatches
    )
