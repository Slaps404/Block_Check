# Block Check - Project Context

> Public, trimmed context for the open `Block_Check` repo.
> Full lab notes, tuning logs, and private datasets stay in the private home repo.

## At a glance

- **Product:** claimed-pair verifier. A barcode/filename claims block-slide; CV checks conservatively.
- **Verdicts:** `PASS` | `REVIEW` (fail-closed). Open ranking is diagnostic only.
- **Runtime shape:** Raspberry Pi 5 captures; a processing computer prepares, scores, and decides.
- **Production code:** `code/` - prepare -> gates -> score -> decision.
- **Open Retrieval:** opt-in work-order N^2 identification (`--open-retrieval`); off by default.

## Product shape

One session / work-order bracket captures blocks then slides. Each claimed pair gets one verdict.
The system is **not** open retrieval by default: it checks the claimed pair and routes uncertainty to REVIEW.

## Key terms

| Term | Meaning |
| --- | --- |
| Specimen | One backlit capture of a block or a slide |
| Block | Paraffin cassette used as the reference |
| Slide | Glass slide cut from a block (verification target) |
| Claimed pair | The block/slide pair asserted by barcode or filename |
| Prepare | Turn a capture into a comparable binary tissue mask |
| Gate | Hard reject before scoring (geometry / quality) |
| REVIEW | Fail-closed outcome when evidence is weak or conflicting |
| Open Retrieval | Diagnostic within-work-order ranking; not the default product mode |

## Architecture (public map)

Data flow: capture/session bring images in; block/slide prepare masks; verify
scores the claimed pair; kiosk/store handle UI and the Pi <-> PC boundary.
`tools/` is CLI-only (not imported by production).

| Area | Role |
| --- | --- |
| `code/session/` | Live capture session / work-order workflow |
| `code/capture/` | Pi camera adapter, calibration, publish |
| `code/kiosk/` | Touchscreen UI + router |
| `code/block/`, `code/slide/` | Role-specific prep / segmentation |
| `code/verify/` | Scoring, gates, work-order evaluation |
| `code/store/` | Two-machine store / RPC boundary |
| `tools/setup_check.py` | Pre-flight env check |
| `tools/capture/` | Pi auto-capture launch + capture tagging |
| `tools/manifest/` | PNG manifest builders |
| `tools/identity/` | QR / barcode decode helpers |
| `tools/scoring_diagnostics/` | Numeric pair diagnostics (CSVs, metrics) |
| `tools/visual_audit/` | Overlay / contact-sheet PNGs for review |
| `tools/matching_corpus/` | Matching-pair sync / score / freeze from session DB |
| `tests/` | Unit/integration tests (image-heavy tests skip without datasets) |

See root `README.md` for the short map and `tools/README.md` for every script.
Durable decisions live in `docs/adr/`. Condensed "do not repeat" lessons live in
`docs/lessons_learned.md`.

## Commands

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -q
.\venv\Scripts\python.exe -m flake8 code/
.\venv\Scripts\python.exe tools\setup_check.py
```

Live two-machine launchers (when hardware is available):

- `tools/start_live_session.bat` - processing PC + Pi
- `tools/start_pi_session.bat` - Pi only

## What this public repo does not include

Private image corpora, regenerable `outputs/`, archived experiments, MVP tuning
notebooks, and internal agent/process docs. Those stay in the private lab home.
