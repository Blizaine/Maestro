# Adapted from Hy-Nomas (https://github.com/Tencent-Hunyuan/Hy-Nomas, hymm/models/camera/geometry: generate_trajectories.py,
# keyboard_mouse_to_extrinsics.py, normalization.py, rotation_scale.py; hymm/pipeline/bidirectional.py and hymm/utils/camera_utils.py),
# Copyright (C) 2026 Tencent, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Camera control of Hy-Nomas worlds: each segment may carry a "camera" (a named trajectory or held keyboard keys) that is turned into
camera-to-world poses (OpenCV axes: +X right, +Y down, +Z forward) on the canonical camera timeline of the clip ((L - 1) * 4 + 1 poses for
L latent frames), normalized per segment as in training (anchored on the segment's first pose, relative to the previous pose,
translations divided by their 99th percentile) and with rotations amplified 2.5 times, the inference preset of the release."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

FPS, TEMPORAL_RATIO = 24, 4
PRESET_FRAMES = 33  # the named trajectories are sampled on 33 poses and resampled to the segment
ROTATION_SCALE = 2.5
MOVE_SPEED, TURN_SPEED, MOUSE_SENSITIVITY = 2.8 / FPS, 25.0 / FPS, 0.1  # units and degrees per frame, degrees per mouse unit
KEYS = ("w", "s", "a", "d", "up", "down", "left", "right")

# Immutable Tencent inference.yaml at 18974d41: per-chunk camera attention
# with one latent frame per chunk. The local runtime implements only this
# singleton-key grouping; other modes need a distinct tested adapter.
PINNED_CAMERA_ATTENTION = {
    "per_segment_camera_attn": False,
    "per_chunk_camera_attn": True,
    "camera_chunk_size": 1,
}


def validate_camera_attention_config(config=None):
    """Reject camera attention groupings that the local runtime does not implement."""
    if config is None:
        return dict(PINNED_CAMERA_ATTENTION)
    if not isinstance(config, dict):
        raise ValueError("Hy-Nomas camera attention config must be an object.")
    unknown = set(config) - set(PINNED_CAMERA_ATTENTION)
    if unknown:
        raise ValueError(f"Unknown Hy-Nomas camera attention settings: {', '.join(sorted(unknown))}.")
    merged = {**PINNED_CAMERA_ATTENTION, **config}
    if (type(merged["per_segment_camera_attn"]) is not bool
            or type(merged["per_chunk_camera_attn"]) is not bool
            or type(merged["camera_chunk_size"]) is not int
            or merged != PINNED_CAMERA_ATTENTION):
        raise ValueError(
            "Hy-Nomas currently supports only per_chunk_camera_attn=true with "
            "camera_chunk_size=1 and per_segment_camera_attn=false; multi-key "
            "camera attention is not implemented."
        )
    return merged


def _look_at(position, target):
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.cross(forward, np.array([0.0, -1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down / np.linalg.norm(down), forward], axis=1)


def _poses(rotations, translations):
    poses = np.tile(np.eye(4), (len(translations), 1, 1))
    poses[:, :3, :3], poses[:, :3, 3] = rotations, translations
    return poses


def _random_walk(frames, step=0.05, rotation_std=2.0, seed=42):
    random = np.random.RandomState(seed)
    position, rotation, poses = np.zeros(3), np.eye(3), []
    for _ in range(frames):
        poses.append(_poses(rotation[None], position[None])[0])
        position = position + rotation[:, 2] * step + random.randn(3) * step * 0.3
        rotation = rotation @ Rotation.from_euler("xyz", random.randn(3) * np.radians(rotation_std)).as_matrix()
    return np.stack(poses)


def _orbit(positions, targets):
    return _poses(np.stack([_look_at(p, t) for p, t in zip(positions, targets)]), positions)


def preset_poses(name, frames=PRESET_FRAMES):
    """Poses of a named trajectory (the source's trajectory generators)."""
    i = np.arange(frames, dtype=np.float64)
    t, zero, still = i / (frames - 1), np.zeros(frames), np.tile(np.eye(3), (frames, 1, 1))
    def line(x, y, z):
        return _poses(still, np.stack([x, y, z], axis=1))
    if name == "circular":
        angle = 2 * np.pi * i / frames
        return _orbit(np.stack([np.cos(angle), zero, np.sin(angle)], axis=1), np.zeros((frames, 3)))
    if name == "spiral":
        angle = 2 * np.pi * t
        return _orbit(np.stack([np.cos(angle), t, np.sin(angle)], axis=1), np.stack([zero, t, zero], axis=1))
    if name == "arc":
        angle = np.radians(-45.0 + 90.0 * t)
        return _orbit(np.stack([2 * np.sin(angle), zero, 2 * np.cos(angle)], axis=1), np.zeros((frames, 3)))
    if name == "random_walk":
        return _random_walk(frames)
    turns = {"pan_left": ("y", -1, 45.0), "pan_right": ("y", 1, 45.0), "tilt_up": ("x", 1, 30.0), "tilt_down": ("x", -1, 30.0)}
    if name in turns:
        axis, sign, degrees = turns[name]
        return _poses(Rotation.from_euler(axis, sign * np.radians(degrees * i / (frames - 1))[:, None]).as_matrix(), np.zeros((frames, 3)))
    lines = {"forward": (zero, zero, 0.08 * i), "dolly": (zero, zero, 2.0 * t), "crane_up": (zero, -t, zero), "crane_down": (zero, t, zero),
             "truck_left": (-t, zero, zero), "truck_right": (t, zero, zero), "zigzag": (0.5 * np.sin(4 * np.pi * t), zero, 0.1 * i)}
    if name not in lines:
        raise ValueError(f"Unknown camera trajectory '{name}': use one of {', '.join(PRESETS)}, or held keys such as {{\"keys\": \"w+left\"}}")
    return line(*lines[name])


PRESETS = ("forward", "dolly", "pan_left", "pan_right", "tilt_up", "tilt_down", "truck_left", "truck_right", "crane_up", "crane_down",
           "arc", "circular", "spiral", "zigzag", "random_walk")


def _held_keys(entry):
    keys = entry.get("keys", [])
    keys = set(keys.lower().replace(" ", "").split("+") if isinstance(keys, str) else (key.lower() for key in keys)) - {""}
    if keys - set(KEYS):
        raise ValueError(f"Unknown camera keys {sorted(keys - set(KEYS))}: use {', '.join(KEYS)}")
    return keys, float(entry.get("mouse_dx", 0.0)), float(entry.get("mouse_dy", 0.0))


def keyboard_poses(spec, frames):
    """Poses of held keys over `frames` video frames (the source's keyboard / mouse generator): W / S move along the view, A / D strafe,
    the arrows turn (25 degrees per second), mouse deltas turn by 0.1 degree per unit and frame. spec: {"keys": "w+left"} for the whole
    segment, or a list of such entries with a "duration" in seconds (an entry without one lasts until the end of the segment, the
    camera stays still after the last one)."""
    entries = [spec] if isinstance(spec, dict) else list(spec)
    inputs = []
    for entry in entries:
        count = frames - len(inputs) if entry.get("duration") is None else round(float(entry["duration"]) * FPS)
        inputs += [_held_keys(entry)] * max(0, min(count, frames - len(inputs)))
    inputs += [(set(), 0.0, 0.0)] * (frames - len(inputs))
    position, pitch, yaw, poses = np.zeros(3), 0.0, 0.0, [np.eye(4)]
    for keys, mouse_dx, mouse_dy in inputs[1:]:  # pose i follows the inputs of frames 0 .. i - 1
        pitch += TURN_SPEED * (("up" in keys) - ("down" in keys)) - mouse_dy * MOUSE_SENSITIVITY
        yaw += TURN_SPEED * (("right" in keys) - ("left" in keys)) + mouse_dx * MOUSE_SENSITIVITY
        pitch = max(-89.0, min(89.0, pitch))
        p, y = np.radians(pitch), np.radians(yaw)
        forward = np.array([np.sin(y) * np.cos(p), -np.sin(p), np.cos(y) * np.cos(p)])
        right = np.array([np.cos(y), 0.0, -np.sin(y)])
        position = position + MOVE_SPEED * (forward * (("w" in keys) - ("s" in keys)) + right * (("d" in keys) - ("a" in keys)))
        poses.append(_poses(Rotation.from_euler("YX", [y, p]).as_matrix()[None], position[None])[0])
    return np.stack(poses)


def resample(poses, count):
    """Poses on `count` evenly spaced instants: rotations by quaternion SLERP, translations linearly."""
    if len(poses) == count:
        return poses
    if len(poses) == 1:
        return np.repeat(poses, count, axis=0)
    source, target = np.linspace(0.0, 1.0, len(poses)), np.linspace(0.0, 1.0, count)
    rotations = Slerp(source, Rotation.from_matrix(poses[:, :3, :3]))(target).as_matrix()
    return _poses(rotations, np.stack([np.interp(target, source, poses[:, axis, 3]) for axis in range(3)], axis=1))


def _se3_inverse(poses):
    inverse = np.zeros_like(poses)
    inverse[..., :3, :3] = np.swapaxes(poses[..., :3, :3], -1, -2)
    inverse[..., :3, 3:4] = -inverse[..., :3, :3] @ poses[..., :3, 3:4]
    inverse[..., 3, 3] = 1.0
    return inverse


def normalize_segment(poses):
    """Training normalization of one segment's poses: anchored on the first one, then each relative to the previous one, the
    translations divided by the 99th percentile of their norms; rotations then amplified ROTATION_SCALE times."""
    out = np.linalg.inv(poses[0]) @ poses
    out[0] = np.eye(4)
    if len(out) > 1:
        out[1:] = _se3_inverse(out[:-1]) @ out[1:]
        out[:, :3, 3] /= max(float(np.quantile(np.linalg.norm(out[1:, :3, 3], axis=1), 0.99)), 1e-6)
    rotations = Rotation.from_matrix(out[:, :3, :3]).as_rotvec() * ROTATION_SCALE
    if np.linalg.norm(rotations, axis=1).max() >= np.pi - 1e-7:
        raise ValueError("A camera trajectory turns too fast for its segment (more than 72 degrees between two camera poses): use a longer "
                         "segment or a slower trajectory")
    out[:, :3, :3] = Rotation.from_rotvec(rotations).as_matrix()
    return out


def camera_boundaries(latent_boundaries):
    """Segment boundaries on the canonical camera timeline: latent frame boundary l -> camera pose 4l - 3 (0 for the clip start)."""
    return [max(0, boundary * TEMPORAL_RATIO - (TEMPORAL_RATIO - 1)) for boundary in latent_boundaries]


def world_camera(segments, latent_boundaries, segment_frames):
    """The normalized poses of a world's camera, 4 per latent frame of the segments (a clip start is padded in front by repeating its
    first pose, as the canonical timeline has 4L - 3 poses), and per segment whether its camera is on; (None, None) without any camera.
    segment_frames: the video frames of each segment (the time base of held keys)."""
    specs = [segment.get("camera") for segment in segments]
    if not any(spec is not None for spec in specs):
        return None, None
    bounds, poses = camera_boundaries(latent_boundaries), []
    for index, spec in enumerate(specs):
        count = bounds[index + 1] - bounds[index]
        if spec is None:
            poses.append(np.tile(np.eye(4), (count, 1, 1)))
            continue
        source = preset_poses(spec) if isinstance(spec, str) else keyboard_poses(spec, segment_frames[index])
        poses.append(normalize_segment(resample(source, count)))
    poses = np.concatenate(poses)
    pad = (-len(poses)) % 4
    poses = np.concatenate([np.repeat(poses[:1], pad, axis=0), poses])
    return torch.from_numpy(poses.astype(np.float32)), [spec is not None for spec in specs]
