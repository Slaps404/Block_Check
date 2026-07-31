# tools/

Standalone CLI scripts that consume core pipeline modules. Not imported by
production code in `code/`.

Scripts one level under `tools/<folder>/` resolve the repo root as
`Path(__file__).resolve().parent.parent.parent` and add `code/` to `sys.path`.

Stale Phase 3 and iPhone-era tools were moved to `archive/tools/` on 2026-07-04.

## Layout

| Folder | Purpose |
|--------|---------|
| `capture/` | Pi capture launcher and data-ingest helpers |
| `manifest/` | Build and maintain manifest CSVs |
| `identity/` | Decode barcodes/QRs into label-truth CSVs |
| `scoring_diagnostics/` | Numeric scorer diagnostics — CSVs, metrics, gap reports |
| `visual_audit/` | PNG contact sheets and mask overlays for human review |
| `matching_corpus/` | Sync, score, and freeze `matching_pairs` rows from `sessions.sqlite3` |

`setup_check.py` stays at the `tools/` root — pre-flight env check, not a diagnostic.

Live two-machine session launchers (Windows): `start_live_session.bat` (receiver + Pi)
and `start_pi_session.bat` (Pi-only). Shared orchestrator: `start_live_session.ps1`.

Empty aborted sessions (folder + SQLite row): `prune_empty_sessions.py`
(dry-run by default; `--apply` to delete). Stop the receiver first. Do **not**
rename `session_*` folders in place — resume looks them up by
`session_{NNNNNN}_*` under the receiver root.

## Entry points

### capture/

| Script | Purpose |
|--------|---------|
| `run_auto_capture.py` | Launch Pi auto-capture console (block or slide mode) |
| `tag_pi_images.py` | Rename Pi capture pairs into `set_NN_…` convention |
| `add_extra_slides_for_blocks.py` | Add extra slide captures for existing blocks |

### manifest/

| Script | Purpose |
|--------|---------|
| `build_png_manifest.py` | Build the active v3 PNG manifest with label-source provenance |

### identity/

| Script | Purpose |
|--------|---------|
| `decode_slide_qrs.py` | Batch-decode slide QR codes → CSV report |

### scoring_diagnostics/

| Script | Purpose |
|--------|---------|
| `run_diagnostics.py` | Canonical production-parity all-pairs diagnostic command |
| `pair_diagnostics.py` | Core all-pairs / selected-pairs diagnostic library |
| `analyze_locked_alignment.py` | Per-metric true-pair vs near-miss separation analysis |
| `run_selected_pairs.py` | Re-score true-pair + near-miss rows from existing CSV |
| `run_selected_pairs_timed.py` | Same as above with per-step wall-clock timing |
| `scoring_contact_sheet.py` | Multi-panel PNG per scored pair (masks, overlay, score bar) |
| `diagnostic_metrics.py` | Diagnostic-only similarity metrics library |
| `router_summary.py` | Per-set router summary from diagnostic CSVs |
| `render_production_locked_overlays.py` | Locked-alignment overlay PNGs for proof pairs |
| `robust_normalization.py` | Shared radial normalization helpers for diagnostic runners |
| `run_retrieval_diagnostics.py` | Build cached invariant-retrieval evidence or calibrate it without rerunning images/scoring |

### Invariant candidate-retrieval proof of concept

This is an offline Open Retrieval experiment only. It is not production-promoted
and does not change production scoring, gates, margins, or verdicts.

The input CSV is a strictly curated manifest. It requires explicit row, claim,
set, label-source, inclusion, and capture provenance; exclusions must be declared
before scoring. Build evidence first, then calibrate only from that cache:

```powershell
.\venv\Scripts\python.exe tools\scoring_diagnostics\run_retrieval_diagnostics.py build --manifest <curated-manifest.csv> --evidence <evidence.json>
.\venv\Scripts\python.exe tools\scoring_diagnostics\run_retrieval_diagnostics.py calibrate --evidence <evidence.json> --report <report.md>
```

`build` prepares each stable specimen once, records full accurate and heuristic
matrices, and incrementally reuses compatible specimens and unchanged work
orders. Cache reuse is rejected when manifest/image identity, Git revision,
preparation, normalization, descriptor, gate, scorer, or evaluator provenance
changes. `calibrate` performs architecture, threshold, hybrid-safety, and veto
analysis without image preparation or pair scoring.

### Locked-alignment metric ablation

Generate raw scores for the four baseline and six experimental metrics:

```powershell
.\venv\Scripts\python.exe tools\scoring_diagnostics\run_diagnostics.py outputs\diagnostics\metric_ablation.csv --no-manifest --dataset images\pi_images_v3
.\venv\Scripts\python.exe tools\scoring_diagnostics\analyze_locked_alignment.py outputs\diagnostics\metric_ablation.csv --metric symmetric_chamfer_mean
```

The CSV is diagnostic evidence, not a production verdict. Compare each metric's
Near-Miss Margin; do not interpret a high score as proof that two specimens are
the same identity.

### visual_audit/

| Script | Purpose |
|--------|---------|
| `render_v3_segmentation_overlays.py` | Per-image segmentation overlay PNGs for v3 dataset |
| `threshold_tune_overlay.py` | Single-image segmentation threshold tuning panels |
| `v3_mask_cleanup_diagnostic.py` | Baseline vs strict cleanup variant comparison |
| `v3_relative_floor_audit.py` | Slide relative component floor audit across v3 sets |
| `audit_slide_label_mask.py` | Slide label masking visual audit |

### matching_corpus/

Offline matching corpus tools for the processing-root `sessions.sqlite3` index.
Mask PNGs stay in existing session artifact dirs; these scripts read/write pair
rows and optional freeze exports only. See `matching_corpus/README.md` for
example commands.

| Script | Purpose |
|--------|---------|
| `sync_matching_pairs.py` | Backfill/sync `true_pair` and same-WO `candidate` rows for a work order |
| `score_matching_pairs.py` | Deferred classical score writeback for unscored pairs; promote near-misses |
| `freeze_matching_corpus.py` | Read-only training snapshot (`pairs.csv`, `specimens.csv`, `README.md`) |

Until Hybrid accept hooks land (#245), run sync/score after capture or overnight.
When Hybrid ships, the same store APIs should be called from the queue:

- Durable slide accept (after job row commits): upsert `true_pair` and
  `sync_matching_pairs_for_work_order(...)`.
- Accurate score completion: `write_matching_pair_score(...)` for scored
  claim/candidates; optional `promote_matching_near_misses`.
- Capture critical path still ends at durable accept (#245 user story 1).
