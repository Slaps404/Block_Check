# Block Check

Computer-vision check for whether a histology slide is visually consistent with
the paraffin block it is labeled as coming from.

Built at **La Jolla Institute for Immunology (LJI)**.

## What it does

1. A barcode / filename claim says which block a slide came from.
2. Each capture is prepared into a comparable tissue mask.
3. A normalized-mask scorer compares the claimed pair.
4. Strong evidence may PASS; uncertainty routes to REVIEW (fail-closed).

Open block-vs-all-slide retrieval is a diagnostic stress test, not the product
workflow.

## Repo map

| Path | What it is |
| --- | --- |
| `code/` | Production library: prepare -> gates -> score -> decide |
| `tools/` | Standalone CLIs. They call `code/`; production code does not import them |
| `tests/` | Pytest suite. Default run skips image-heavy tests that need private datasets |
| `docs/` | Public context, lessons, and ADRs |
| `requirements.txt` | Python dependencies |

### `code/` packages

Data flow in short: capture/session bring images in; block/slide turn them into
masks; verify scores the claimed pair; kiosk/store are the UI and two-machine
boundary.

| Package | Role |
| --- | --- |
| `code/session/` | Live capture session and work-order workflow |
| `code/capture/` | Pi camera adapter, calibration, publish to the processing PC |
| `code/kiosk/` | Touchscreen UI screens and event router |
| `code/block/` | Block (paraffin cassette) prep and segmentation |
| `code/slide/` | Slide prep and segmentation |
| `code/verify/` | Scoring, hard gates, work-order evaluation, verdicts |
| `code/store/` | Two-machine store / RPC boundary (Pi <-> processing PC) |

### `tools/` (what you actually run)

Start here:

| Entry | Purpose |
| --- | --- |
| `tools/setup_check.py` | Pre-flight env check after install |
| `tools/start_live_session.bat` | Launch processing PC + Pi session (Windows) |
| `tools/start_pi_session.bat` | Pi-only session launcher |
| `tools/README.md` | Full per-script table for every tools folder |

Folders shipped in this public repo:

| Folder | When you use it |
| --- | --- |
| `tools/capture/` | Launch Pi auto-capture; tag / rename capture pairs |
| `tools/manifest/` | Build PNG manifests used by diagnostics |
| `tools/identity/` | Decode slide QR / barcodes into label CSVs |
| `tools/scoring_diagnostics/` | All-pairs / selected-pair numeric diagnostics (CSVs, metrics) |
| `tools/visual_audit/` | PNG overlays and contact sheets for human review |
| `tools/matching_corpus/` | Sync / score / freeze matching-pair rows from session DB |

Rule of thumb: change behavior in `code/`; use `tools/` to exercise, audit, or
launch that behavior.

## Docs

- `docs/PROJECT_CONTEXT.md` - product shape, architecture, commands, key terms
- `docs/lessons_learned.md` - condensed failed-approach lessons
- `docs/adr/` - architecture decision records
- `tools/README.md` - detailed tools entry-point list

## Setup

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe tools\setup_check.py
.\venv\Scripts\python.exe -m pytest tests/ -q
```

Default pytest excludes integration tests that need private image datasets.

## License

MIT
