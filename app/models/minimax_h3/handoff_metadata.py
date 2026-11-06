"""Record Ref2VA conditioning evidence without changing the model inputs."""
from __future__ import annotations

import copy
import hashlib


def capture_reference_handoff(trace, references, *, window_start_frame,
                              rendered_frames, fps, continuation_audio_rows=0):
    """Attach observed encoder input identities and reference-only row ranges.

    These are conditioning diagnostics, not a claim that rendered voices match.
    Waveforms have already been decoded by the normal pipeline; hashing them
    neither reloads source files nor invokes a model.
    """
    if not isinstance(trace, dict):
        return None
    result = copy.deepcopy(trace)
    result.update(stage="runtime", window_start_frame=int(window_start_frame),
                  rendered_frames=int(rendered_frames), fps=float(fps))
    result["continuation_audio_rows"] = int(continuation_audio_rows)
    audio_offset = 0
    video_offset = 0
    encoded = []
    for index, reference in enumerate(references):
        entry = {"reference_index": index, "kind": reference.kind,
                 "role": reference.role}
        if reference.kind != "audio":
            count = int(reference.num_video_rows)
            entry["reference_video_rows"] = [video_offset, video_offset + count]
            video_offset += count
        if reference.has_audio:
            count = int(reference.num_audio_rows)
            entry["reference_audio_rows"] = [audio_offset, audio_offset + count]
            entry["packed_condition_audio_rows"] = [
                int(continuation_audio_rows) + audio_offset,
                int(continuation_audio_rows) + audio_offset + count,
            ]
            waveform = reference.waveform
            if waveform is not None:
                try:
                    array = waveform.detach().cpu().contiguous().numpy()
                    entry["waveform_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
                    entry["waveform_shape"] = list(array.shape)
                    entry["waveform_dtype"] = str(array.dtype)
                except (AttributeError, TypeError, RuntimeError) as error:
                    entry["waveform_hash_error"] = type(error).__name__
            audio_offset += count
        encoded.append(entry)
    result["encoded_references"] = encoded
    result["reference_audio_row_count"] = audio_offset
    result["reference_video_row_count"] = video_offset
    return result


def accumulate_reference_handoffs(previous, artifact_metadata):
    """Carry completed windows into the final joined artifact's metadata.

    Replace a repeated window by its final handoff, keeping prior snapshots
    independent so earlier output files cannot acquire future-window entries.
    """
    if not isinstance(artifact_metadata, dict):
        return previous, artifact_metadata
    current = artifact_metadata.get("h3_reference_handoffs")
    if not isinstance(current, list):
        return previous, artifact_metadata
    indexed = {
        (item.get("window_index"), item.get("window_start_frame")): copy.deepcopy(item)
        for item in previous if isinstance(item, dict)
    }
    for item in current:
        if isinstance(item, dict) and item.get("schema_version") == 1:
            indexed[(item.get("window_index"), item.get("window_start_frame"))] = copy.deepcopy(item)
    combined = list(indexed.values())
    metadata = dict(artifact_metadata)
    metadata["h3_reference_handoffs"] = copy.deepcopy(combined)
    return combined, metadata
