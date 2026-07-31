"""Topology-preserving growth core for production block tissue masks.

The production mask supplies immutable seed components. Candidate pixels may
expand those components, but competing labels retain a background boundary.
Only short paths made entirely of selective dark-brown pixels can authorize a
seed-label union. This module holds the growth CORE used by production
block preparation (`grow_block_mask`). Historical diagnostic harnesses were
removed in issue #206; use `docs/mvp_tuning_log/` for the promotion evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.segmentation import watershed

from constants import (
    BLOCK_GROW_LOOSE_VALUE_MAX,
    BLOCK_GROW_SELECTIVE_HUE_MAX,
    BLOCK_GROW_SELECTIVE_SAT_MIN,
    BLOCK_GROW_SELECTIVE_VALUE_MAX,
    BLOCK_HALO_GROW_HUE_MAX,
    BLOCK_HALO_GROW_LAB_B_MIN,
    BLOCK_HALO_GROW_SAT_MIN,
    MIN_AREA_FRACTION,
    MIN_BLOCK_COMPONENT_AREA,
)
from verify.scale import scale_area_min
from verify.segmentation import _postprocess


@dataclass(frozen=True)
class GrowthResult:
    """A candidate binary mask plus the label decisions that produced it."""

    mask: np.ndarray
    labels: np.ndarray
    unioned_seed_pairs: tuple[tuple[int, int], ...]


class _UnionFind:
    def __init__(self, labels: range) -> None:
        self.parent = {label: label for label in labels}

    def find(self, label: int) -> int:
        parent = self.parent[label]
        if parent != label:
            self.parent[label] = self.find(parent)
        return self.parent[label]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _inside_window(
    shape: tuple[int, int],
    window: tuple[int, int, int, int] | None,
) -> np.ndarray:
    inside = np.zeros(shape, dtype=bool)
    if window is None:
        return inside
    x, y, width, height = window
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(shape[1], x + width), min(shape[0], y + height)
    if x1 > x0 and y1 > y0:
        inside[y0:y1, x0:x1] = True
    return inside


def build_candidate_regions(
    bgr_image: np.ndarray,
    window: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed selective and loose predicates scoped to the cassette window."""
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError("bgr_image must have shape (height, width, 3)")

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    lab_b = lab[:, :, 2]
    inside = _inside_window(value.shape, window)

    selective = (
        inside
        & (hue <= BLOCK_GROW_SELECTIVE_HUE_MAX)
        & (saturation >= BLOCK_GROW_SELECTIVE_SAT_MIN)
        & (value <= BLOCK_GROW_SELECTIVE_VALUE_MAX)
    )
    halo = (
        inside
        & (hue <= BLOCK_HALO_GROW_HUE_MAX)
        & (lab_b >= BLOCK_HALO_GROW_LAB_B_MIN)
        & (saturation >= BLOCK_HALO_GROW_SAT_MIN)
    )
    loose = selective | halo | (inside & (value <= BLOCK_GROW_LOOSE_VALUE_MAX))
    return selective, loose


def find_selective_unions(
    seed_labels: np.ndarray,
    selective: np.ndarray,
    bridge_margin: int,
) -> set[tuple[int, int]]:
    """Find seed-label pairs whose margin-limited selective fronts overlap."""
    if seed_labels.shape != selective.shape:
        raise ValueError("seed_labels and selective must have the same shape")
    if bridge_margin < 0:
        raise ValueError("bridge_margin must be non-negative")
    if bridge_margin == 0:
        return set()

    label_ids = [int(label) for label in np.unique(seed_labels) if label > 0]
    kernel = np.ones((3, 3), dtype=np.uint8)
    reaches: dict[int, tuple[tuple[int, int, int, int], np.ndarray]] = {}
    for label in label_ids:
        ys, xs = np.where(seed_labels == label)
        x0 = max(0, int(xs.min()) - bridge_margin - 1)
        y0 = max(0, int(ys.min()) - bridge_margin - 1)
        x1 = min(seed_labels.shape[1], int(xs.max()) + bridge_margin + 2)
        y1 = min(seed_labels.shape[0], int(ys.max()) + bridge_margin + 2)
        local_labels = seed_labels[y0:y1, x0:x1]
        local_selective = selective[y0:y1, x0:x1]
        reached = local_labels == label
        frontier = reached.copy()
        for _ in range(bridge_margin):
            dilated = cv2.dilate(frontier.astype(np.uint8), kernel) > 0
            frontier = dilated & local_selective & ~reached
            reached |= frontier
            if not np.any(frontier):
                break
        reaches[label] = ((x0, y0, x1, y1), reached)

    pairs: set[tuple[int, int]] = set()
    for index, left in enumerate(label_ids):
        for right in label_ids[index + 1:]:
            left_box, left_reach = reaches[left]
            right_box, right_reach = reaches[right]
            ix0, iy0 = max(left_box[0], right_box[0]), max(left_box[1], right_box[1])
            ix1, iy1 = min(left_box[2], right_box[2]), min(left_box[3], right_box[3])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            left_view = left_reach[
                iy0 - left_box[1]:iy1 - left_box[1],
                ix0 - left_box[0]:ix1 - left_box[0],
            ]
            right_view = right_reach[
                iy0 - right_box[1]:iy1 - right_box[1],
                ix0 - right_box[0]:ix1 - right_box[0],
            ]
            left_touch = cv2.dilate(left_view.astype(np.uint8), kernel) > 0
            if np.any(left_touch & right_view):
                pairs.add((left, right))
    return pairs


def _group_seed_labels(
    seed_labels: np.ndarray,
    pairs: set[tuple[int, int]],
) -> np.ndarray:
    count = int(seed_labels.max())
    groups = _UnionFind(range(1, count + 1))
    for left, right in pairs:
        groups.union(left, right)

    roots = sorted({groups.find(label) for label in range(1, count + 1)})
    root_to_marker = {root: index + 1 for index, root in enumerate(roots)}
    grouped = np.zeros(seed_labels.shape, dtype=np.int32)
    for label in range(1, count + 1):
        grouped[seed_labels == label] = root_to_marker[groups.find(label)]
    return grouped


def _watershed_labels(markers: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    labels = watershed(
        np.zeros(markers.shape, dtype=np.uint8),
        markers=markers,
        connectivity=np.ones((3, 3), dtype=bool),
        mask=allowed,
        watershed_line=True,
    ).astype(np.int32)
    labels[markers > 0] = markers[markers > 0]
    return labels


def _clean_each_label(
    labels: np.ndarray,
    immutable_markers: np.ndarray,
    *,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Run production-shaped block cleanup independently for every label."""
    min_area = max(
        scale_area_min(MIN_BLOCK_COMPONENT_AREA, pixel_scale),
        int(labels.size * MIN_AREA_FRACTION),
    )
    cleaned_masks: dict[int, np.ndarray] = {}
    for label in (int(value) for value in np.unique(labels) if value > 0):
        mask = (labels == label).astype(np.uint8) * 255
        cleaned = _postprocess(
            mask, min_area=min_area, role="block", pixel_scale=pixel_scale
        )
        cleaned[immutable_markers == label] = 255
        cleaned_masks[label] = cleaned > 0

    occupancy = np.zeros(labels.shape, dtype=np.uint8)
    combined = np.zeros(labels.shape, dtype=np.int32)
    for label, mask in cleaned_masks.items():
        occupancy[mask] += 1
        combined[(combined == 0) & mask] = label
    combined[occupancy > 1] = 0
    combined[immutable_markers > 0] = immutable_markers[immutable_markers > 0]
    return combined


def grow_topology_preserving(
    seed_mask: np.ndarray,
    selective: np.ndarray,
    loose: np.ndarray,
    bridge_margin: int,
    *,
    clean_labels: bool = False,
    pixel_scale: float = 1.0,
) -> GrowthResult:
    """Grow immutable seed labels through loose pixels without accidental merges."""
    if seed_mask.shape != selective.shape or seed_mask.shape != loose.shape:
        raise ValueError("seed_mask, selective, and loose must have the same shape")
    _, seed_labels = cv2.connectedComponents(
        (seed_mask > 0).astype(np.uint8),
        connectivity=8,
    )
    pairs = find_selective_unions(seed_labels, selective, bridge_margin)
    return _grow_with_pairs(
        seed_mask, loose, seed_labels, pairs, clean_labels, pixel_scale
    )


def _grow_with_pairs(
    seed_mask: np.ndarray,
    loose: np.ndarray,
    seed_labels: np.ndarray,
    pairs: set[tuple[int, int]],
    clean_labels: bool,
    pixel_scale: float,
) -> GrowthResult:
    grouped_markers = _group_seed_labels(seed_labels, pairs)
    allowed = loose | (seed_mask > 0)
    labels = _watershed_labels(grouped_markers, allowed)
    if clean_labels:
        labels = _clean_each_label(
            labels, grouped_markers, pixel_scale=pixel_scale
        )
    labels[grouped_markers > 0] = grouped_markers[grouped_markers > 0]
    mask = (labels > 0).astype(np.uint8) * 255
    return GrowthResult(mask, labels, tuple(sorted(pairs)))


def grow_block_mask(
    bgr_image: np.ndarray,
    seed_mask: np.ndarray,
    window: tuple[int, int, int, int] | None,
    *,
    bridge_margin: int = 0,
    pixel_scale: float = 1.0,
) -> np.ndarray:
    """Grow a production block seed mask via topology-preserving growth.

    Returns a uint8 mask (0/255). No-op-safe: window=None grows nothing.
    """
    selective, loose = build_candidate_regions(bgr_image, window)
    return grow_topology_preserving(
        seed_mask,
        selective,
        loose,
        bridge_margin,
        clean_labels=True,
        pixel_scale=pixel_scale,
    ).mask
