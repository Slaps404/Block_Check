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
| `tools/` | Setup check and session launchers. Production code does not import them |
| `tests/` | Pytest suite. Default run skips image-heavy tests that need private datasets |
| `docs/` | Public context, lessons, and ADRs |
| `requirements.txt` | Python dependencies |

### `code/` packages

Data flow in short: capture/session bring images in; block/slide turn them into
masks; verify scores the claimed pair; retrieval ranks candidates when the claim
is missing; kiosk is the UI; transport is the two-machine boundary; app wires it
all together.

| Package | Role |
| --- | --- |
| `code/session/` | Live capture session and work-order workflow |
| `code/capture/` | Pi camera adapter, calibration, publish to the processing PC |
| `code/kiosk/` | Touchscreen UI screens and event router |
| `code/block/` | Block (paraffin cassette) prep and segmentation |
| `code/slide/` | Slide prep and segmentation |
| `code/verify/` | Scoring, hard gates, work-order evaluation, verdicts |
| `code/retrieval/` | Optional candidate retrieval and Hybrid configuration |
| `code/transport/` | Pi <-> processing PC remote store and wire encoding |
| `code/app/` | Pi session and processing-PC receiver entry points |

### `tools/` (what you actually run)

| Entry | Purpose |
| --- | --- |
| `tools/setup_check.py` | Pre-flight env check after install |
| `tools/run_receiver.py` | Start the processing-PC receiver |
| `tools/run_pi_session.py` | Start the Pi capture session and kiosk |
| `tools/start_live_session.bat` / `.ps1` | Launch processing PC + Pi session (Windows) |
| `tools/start_pi_session.bat` | Pi-only session launcher |

Research and diagnostic tools stay in the private lab repo.

## Docs

- `docs/PROJECT_CONTEXT.md` - product shape, architecture, commands, key terms
- `docs/lessons_learned.md` - condensed failed-approach lessons
- `docs/adr/` - architecture decision records

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
