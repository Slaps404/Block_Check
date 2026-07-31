"""Static dependency boundaries for deployed production code."""

import ast
import inspect
from pathlib import Path

from session import workflow as session_workflow
from store.remote import RemoteProcessingStore


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

    `tests/conftest.py` appends several `tools/<subdir>` folders straight
    onto `sys.path` (capture, manifest, identity, scoring_diagnostics,
    visual_audit, diagnostics, calibration) so their modules are importable
    by bare name inside the test session. Of those, the ones whose name
    marks them as diagnostics/experiment tooling -- as opposed to the
    operational tooling in `capture/`, `manifest/`, and `identity/` that
    `tools/README.md` documents as a separate category -- are the ones #245
    and #246 forbid production from depending on. Matching on the
    "diagnostic" substring (rather than a hardcoded folder list) means a
    future `tools/diagnostics_v2` or `tools/foo_diagnostics` directory is
    picked up automatically, with no edit to this test required.
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
    run, so adding a new diagnostic module never requires touching this
    test to keep the guard current.
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
    it lives under (see `_diagnostics_experiment_module_owners`), so the
    violation message can name the real diagnostics package the import came
    from. Parameterized on `code_dir`/`forbidden_owners` (rather than reading
    the real repo paths directly) so it can be driven against a synthetic
    temp directory in a unit test, proving the detection logic catches a
    violation without ever touching the real `code/` tree.
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
    """#246: no production module may import a diagnostics-experiment module.

    `tests/conftest.py` appends `tools/scoring_diagnostics` (and sibling
    diagnostics dirs) directly onto sys.path, so every module living there
    is importable by bare name -- e.g.
    `from candidate_retrieval_analysis import worst_subgroup` -- with no
    `tools.` prefix. `test_code_does_not_import_tools_or_archive` above only
    catches the prefixed form, so a bare-name diagnostics import added to
    `code/` would otherwise pass every existing test and silently invert
    the tools -> code dependency arrow that #245/#246 require to stay
    one-way.

    The forbidden module-name set is derived from the filesystem (see
    `_diagnostics_experiment_module_owners`) rather than hardcoded, so this
    test does not rot as diagnostic modules are added, renamed, or removed.
    """
    repo_root = Path(__file__).resolve().parents[1]
    forbidden_owners = _diagnostics_experiment_module_owners(repo_root / "tools")
    violations = _find_diagnostics_import_violations(repo_root / "code", forbidden_owners)
    assert not violations, "\n".join(violations)


def test_invariant_descriptor_catalog_is_production_owned_not_duplicated_in_diagnostics():
    """#246: the Heuristic Descriptor Catalog is production-owned (`code/verify`).

    Diagnostic tooling under `tools/scoring_diagnostics` must import the catalog
    from production rather than keeping its own copy of the math, so the
    dependency arrow points diagnostics -> production and never the reverse.
    """
    repo_root = Path(__file__).resolve().parents[1]
    production_module = repo_root / "code" / "verify" / "invariant_descriptors.py"
    diagnostics_module = (
        repo_root / "tools" / "scoring_diagnostics" / "invariant_descriptors.py"
    )
    assert production_module.is_file(), (
        "invariant descriptor catalog must live under code/verify"
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
    assert "verify.invariant_descriptors" in imported_modules, (
        "diagnostics must import the descriptor catalog from the production module"
    )
    assert "invariant_descriptors" not in imported_modules, (
        "diagnostics must not import a bare/local invariant_descriptors module"
    )


def test_rpc_whitelist_arity_and_proxy_surface_stay_in_sync():
    """`_RPC_METHODS`, `_RPC_ARITY`, and the `RemoteProcessingStore` proxy are
    hand-maintained in three places (#195); this is the sole test protecting
    that they never drift apart. `snapshot` is excluded from the proxy
    comparison because it keeps its dedicated `/status` route and is
    deliberately absent from `_RPC_METHODS`.
    """
    rpc_methods = set(session_workflow._RPC_METHODS)
    rpc_arity = set(session_workflow._RPC_ARITY)
    assert rpc_methods == rpc_arity, (
        f"_RPC_METHODS/_RPC_ARITY key mismatch: "
        f"only in _RPC_METHODS={rpc_methods - rpc_arity}, "
        f"only in _RPC_ARITY={rpc_arity - rpc_methods}"
    )

    proxy_members = inspect.getmembers(RemoteProcessingStore, predicate=inspect.isfunction)
    proxy_methods = {
        name for name, _ in proxy_members
        if not name.startswith("_") and name not in {"snapshot", "results_evidence_bytes"}
    }
    assert proxy_methods == rpc_methods, (
        f"RemoteProcessingStore proxy surface out of sync with _RPC_METHODS: "
        f"only on proxy={proxy_methods - rpc_methods}, "
        f"only in _RPC_METHODS={rpc_methods - proxy_methods}"
    )
