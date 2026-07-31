---
status: accepted
---

# Share one durable per-slide queue across retrieval modes

Open Retrieval, Hybrid, and Hybrid Shadow use one durable Retrieval Slide Job
lifecycle and live result projection, so every accepted slide appears as PENDING
and updates as soon as its background score completes. Scoring strategy remains
mode-specific: Open Retrieval evaluates every block in the work order, Hybrid
uses its heuristic candidate subset with the existing full-pool fallback, and
Hybrid Shadow evaluates the complete frozen Hybrid pool. This supersedes ADR
0009's deferred all-at-once Open Retrieval reveal because a second batch queue
duplicated recovery and hid useful completed work from the operator.
