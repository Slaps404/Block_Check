"""Slide image preparation and claimed-pair evaluation, without database writes.

The store owns identity lookup, replay checks, and saving the returned result.
Retrieval workers also use prepare_slide_image with their own error policy.
"""
from pathlib import Path
from time import perf_counter_ns
from typing import Callable

import cv2
import numpy as np

from runtime_observer import RuntimeObserver, observed
from session.pipeline import ClaimDecision, VERDICT_REVIEW, decide_claim
from session.preparation import PreparationFailure, PreparedResult
from session.workflow_types import BlockReadiness


def prepare_slide_image(
    path: Path, item_id: str,
    preprocessor: Callable[[np.ndarray], PreparedResult], *,
    observer: RuntimeObserver | None = None, fail_closed: bool = True,
) -> tuple[np.ndarray | None, PreparedResult]:
    """Decode once; optionally turn unexpected preparation errors into REVIEW input."""
    image = None

    def prepare():
        nonlocal image
        image = cv2.imread(str(path))
        if image is None:
            return None, PreparationFailure(role="slide", reason=f"could not read image: {path}")
        return image, preprocessor(image)

    with observed(observer, "slide_preparation", item_id):
        if not fail_closed:
            return prepare()
        try:
            return prepare()
        except Exception as exc:
            return image, PreparationFailure(role="slide", reason=str(exc))


def evaluate_claimed_slide(
    *, block_id: str, slide_path: Path, block_path: str,
    readiness: BlockReadiness, load_block: Callable[[], PreparedResult],
    prepare_slide: Callable[[], tuple[np.ndarray | None, PreparedResult]],
    record_preparation: Callable[[int], None], observer: RuntimeObserver | None = None,
) -> tuple[PreparedResult | None, PreparedResult, np.ndarray | None, ClaimDecision]:
    """Reuse the prepared block, prepare the slide, and evaluate one claim."""
    if not readiness.evaluable:
        failure = PreparationFailure(
            role="slide", reason="slide preparation skipped: block is unusable"
        )
        decision = ClaimDecision(
            claim_id=block_id, block_path=block_path, slide_path=str(slide_path),
            verdict=VERDICT_REVIEW, stage="block_unusable",
            reason=readiness.review_reason or "block is not evaluable",
        )
        return None, failure, None, decision
    block = load_block()
    started = perf_counter_ns()
    image, slide = prepare_slide()
    record_preparation(started)
    decision = decide_claim(
        block_id, block, slide, block_path=block_path,
        slide_path=str(slide_path), observer=observer,
    )
    return block, slide, image, decision
