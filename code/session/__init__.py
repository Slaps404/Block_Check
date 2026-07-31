"""Session package cluster.

Stable public import path for the durable session seam is ``session.workflow``
(#201). Submodules under ``session`` are imported directly; this package does
not eagerly re-export workflow symbols (that would create import cycles).
"""
