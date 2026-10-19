"""Release host tracking arrays once Recast preparation has materialized."""

from __future__ import annotations


_TRACKING_ARRAY_KEYS = ("source_frames", "mask_frames", "mapping_masks")


def release_recast_tracking_arrays(mapped_tracking, mapping_anchors=None):
    """Drop large timeline arrays and array-view anchors after their last use.

    Recast keeps only the already-written mask/reference/shot files during
    generation. This helper deliberately preserves scalar timeline metadata
    (shot ranges, summaries, frame indices, FPS, and the video-mask path).
    The caller must also clear any local NumPy views before generation starts.
    """
    released = []
    if isinstance(mapped_tracking, dict):
        for key in _TRACKING_ARRAY_KEYS:
            if key in mapped_tracking:
                mapped_tracking.pop(key, None)
                released.append(key)

    if isinstance(mapping_anchors, list):
        mapping_anchors.clear()

    return tuple(released)
