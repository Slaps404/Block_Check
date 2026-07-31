"""Tiny production-safe stage observation protocol.

The runtime harness may pass an observer during a separate profiling run.
Ordinary production and authoritative timing leave the observer as ``None``.
"""

from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter_ns
from typing import Iterator, Protocol


REQUIRED_STAGES = frozenset({
    "decode_load",
    "block_setup",
    "slide_preparation",
    "locked_cache",
    "quality_gates",
    "alignment_scoring",
    "verdict_qc_serialization",
    "end_to_end",
})


class RuntimeObserver(Protocol):
    def record(self, stage: str, elapsed_ns: int, item_id: str) -> None:
        """Record one non-authoritative stage duration."""


class NullRuntimeObserver:
    def record(self, stage: str, elapsed_ns: int, item_id: str) -> None:
        return None


@contextmanager
def observed(
    observer: RuntimeObserver | None,
    stage: str,
    item_id: str,
) -> Iterator[None]:
    started = perf_counter_ns()
    try:
        yield
    finally:
        if observer is not None:
            observer.record(stage, perf_counter_ns() - started, item_id)
