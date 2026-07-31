```text
.\venv\Scripts\python.exe tools\matching_corpus\inventory_matching_corpus.py --output exports\matching_corpus_inventory.csv
.\venv\Scripts\python.exe tools\matching_corpus\sync_matching_pairs.py --work-order 7842
.\venv\Scripts\python.exe tools\matching_corpus\score_matching_pairs.py --work-order 7842
.\venv\Scripts\python.exe tools\matching_corpus\freeze_matching_corpus.py --output exports\matching_corpus_7842 --work-order 7842
```

`--live-root` defaults to `outputs\live_session`. Pass it explicitly for other roots (for example `outputs\hardware_qa_117`).

`inventory_matching_corpus.py` opens `sessions.sqlite3` read-only and does not
score or reprocess images. Its CSV links each block to claimed slides and flags
missing image files, missing claims, duplicate slide claims, superseded captures,
and block IDs repeated across capture brackets. Omit `--output` for counts only;
add `--session N` to inspect one session.

`score_matching_pairs.py` scores unscored `candidate` and `near_miss` rows by default. Use `--include-true-pairs` to score true pairs too. Near-miss promotion uses `--margin` (default `0.05`, project `MATCH_MARGIN`).

`freeze_matching_corpus.py` writes `pairs.csv`, `specimens.csv`, and `README.md` under `--output`. Default is path-only (no mask copy). Add `--copy-masks` to copy referenced masks into `output/masks/` and rewrite CSV paths relative to the freeze.

Note: interim until Hybrid accept hooks land; operators/devs run this after capture (or overnight).
