"""Per-generation placement and private, hardware-scoped performance evidence.

No GPU allocations, remote telemetry, prompts, or automatic benchmark jobs.
The existing model workspace guard remains authoritative over learned budgets.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
import threading
import time
from pathlib import Path
from contextlib import contextmanager

REVISION = 1


def _number(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def auto_owns(config, key):
    services = config.get("services", {})
    defaults = services.get("auto_performance_defaults", {})
    return bool(services.get("auto_performance", True) and key in defaults
                and config.get(key) == defaults[key])


def hardware_key(hardware):
    return _digest({key: hardware.get(key) for key in (
        "gpu_name", "gpu_vram_gb", "ram_gb", "gpu_capability", "driver_version",
        "torch_version", "runtime_version", "platform", "machine", "allocator_active", "runtime_signature",
    )} | {"revision": REVISION})


def workload_key(config, model_def, params):
    # Hash file identities/stat revisions; never persist local filenames.
    def file_stamp(value):
        if isinstance(value, (list, tuple)):
            return [file_stamp(item) for item in value]
        if isinstance(value, dict):
            return {key: file_stamp(item) for key, item in value.items()}
        if isinstance(value, str):
            try:
                info = Path(value).stat()
                return (value, info.st_size, info.st_mtime_ns)
            except OSError:
                pass
        return value

    keys = ("model_type", "resolution", "image_mode", "num_inference_steps",
            "activated_loras", "loras_multipliers", "loras_multipliers2",
            "guidance_scale", "flow_shift", "sampler", "cache_type", "tea_cache",
            "single_stage_pipeline", "progressive_pipeline", "duration_seconds")
    workload = {key: file_stamp(params.get(key)) for key in keys}
    frames = _number(params.get("video_length"), 1)
    window = _number(params.get("sliding_window_size"), frames)
    workload["frames"] = min(frames, window) if window > 0 else frames
    workload["total_frames"] = frames
    workload["repeats"] = params.get("repeat_generation", params.get("num_repeats", 1))
    workload["window_prompts"] = [len(str(item)) for item in params.get("window_prompts", [])]
    workload["prompt_lengths"] = [len(item) for item in str(params.get("prompt", "")).split("\n")]
    # Reference identity is hashed, including duration/trimming/type metadata.
    # Exact matches are deliberately conservative: no extrapolation to heavier jobs.
    workload["references"] = {key: file_stamp(params.get(key)) for key in (
        "image_refs", "scene_references", "video_guide", "audio_guide", "video_source",
        "minimax_h3_references", "audio_guide2", "audio_guide3", "image_start", "image_end",
    )}
    # Include remaining generation options (e.g. per-request attention, cache
    # and sliding-window overlap). Placement overrides are comparison axes.
    excluded = {"override_profile", "workspace", "seed", "client_submission_id",
                "show_in_gallery", "performance_plan", "enhancement"}
    workload["generation_options"] = {
        key: file_stamp(value) for key, value in params.items()
        if key not in excluded and not key.startswith("_") and "prompt" not in key
    }
    workload["model"] = file_stamp(model_def)
    workload["runtime"] = {key: config.get(key) for key in (
        "transformer_quantization", "attention_mode", "int8_kernels", "compile",
        "attention_head_split", "vae_config", "smart_memory_pinning", "read_ahead",
        "vram_safety_coefficient", "generation_preview", "generation_preview_mode",
    )}
    return _digest(workload)


def build_performance_plan(hardware, config, model_def, params, adjustment=None,
                           *, output_type="video", profile=1, cli=None,
                           available_ram_gb=None, own_pinned_gb=0, external_vram_gb=0, history=None):
    """Plan temporary settings without altering saved config or job quality."""
    adjustment = adjustment or {}
    cli = cli or {}
    vram = _number(hardware.get("gpu_vram_gb"))
    ram = _number(hardware.get("ram_gb"))
    default_ram_fraction = .4 if hardware.get("platform") == "win32" else .5
    mode = config.get(f"{output_type}_preload_mode", "default")
    explicit_profile = _number(params.get("override_profile"), -1) >= 0
    applied = bool(hardware.get("cuda_available") and vram > 0
                   and auto_owns(config, f"{output_type}_profile")
                   and mode == "default" and not explicit_profile
                   and _number(cli.get("profile"), -1) < 0
                   and not any(_number(cli.get(key)) > 0 for key in ("preload", "transformer_budget")))
    plan = {"applied": applied, "output_type": output_type, "profile": profile,
            "transformer_budget_mb": int(cli.get("transformer_budget") or 0),
            "hardware_key": hardware_key(hardware),
            "workload_key": workload_key(config, model_def, params),
            "reasons": [], "warnings": [], "matching_renders": 0,
            "source": "hardware" if applied else "manual",
            "override_read_ahead": False, "override_reserved_ram": False,
            "read_ahead": bool(config.get("read_ahead", False)),
            "reserved_ram_fraction": _number(cli.get("perc_reserved_mem_max")) or _number(config.get("perc_reserved_mem_max")) / 100 or default_ram_fraction}
    if (model_def.get("minimax_h3_full_checkpoint") and ram >= 96
            and _number(cli.get("perc_reserved_mem_max")) <= 0 and not config.get("perc_reserved_mem_max")):
        plan["reserved_ram_fraction"] = round(min(.65, max(.50, 60 / ram)), 3)
    if not applied:
        plan["reasons"].append("Saved settings or explicit overrides remain in control.")
        return plan
    is_h3 = str(model_def.get("architecture", "")).startswith("minimax_h3") or "h3_weight_budget_gb" in adjustment
    clamped = adjustment.get("h3_activation_reserve_clamped", False)
    # The previous guard exports this policy when the requested workspace cannot fit.
    clamped = clamped or adjustment.get("h3_residency_policy") == "profile_default"
    cap_gb = min(_number(adjustment.get("h3_weight_budget_gb"), vram),
                 _number(adjustment.get("effective_coef"), .7) * vram) * .97
    # A cold worker can see memory occupied by other applications. A warm
    # worker passes zero here because its own model residency is reusable.
    cap_gb = max(0, cap_gb - max(0, _number(external_vram_gb) - .75))
    if external_vram_gb > 2:
        plan["warnings"].append("Other GPU activity is using VRAM before model loading; close other GPU applications if memory is tight.")
    allowed_profiles = [profile]
    if is_h3:
        if ram >= 96 and vram >= 20:
            plan["profile"] = 2
            allowed_profiles = [2, 5]
            plan["reasons"].append("H3 streams transformer weights while keeping host models pinned (Profile 2).")
        elif ram < 64:
            # Extra host pinning is only learned from repeated safe evidence;
            # it is never the untested starting point on a smaller RAM host.
            allowed_profiles = list(dict.fromkeys([profile, 4.5, 5])) if ram >= 24 else [profile, 5]
            plan["reasons"].append("Conservative host placement protects the operating system on a smaller RAM machine.")
        if clamped:
            plan["transformer_budget_mb"] = 0
            plan["warnings"].append("The requested H3 workspace is tight for this GPU. Profile defaults are retained; a shorter window or lower resolution may be needed.")
        else:
            target = vram / 3 if ram >= 96 else min(3, vram / 5)
            plan["transformer_budget_mb"] = max(0, int(min(target, cap_gb) * 1024) // 256 * 256)
            plan["reasons"].append(f"Transformer allowance {plan['transformer_budget_mb'] / 1024:g} GB; activation and reference reserves remain protected.")
    # Available RAM includes reclaimable MMGP-owned pinned buffers. Do not
    # mistake our warm model cache for an unrelated application's memory.
    if available_ram_gb is not None and ram > 0:
        spare = max(0, _number(available_ram_gb) + _number(own_pinned_gb) - max(4, ram * .10))
        if auto_owns(config, "read_ahead"):
            plan["override_read_ahead"] = True
            plan["read_ahead"] = bool(config.get("read_ahead") and spare >= 32)
        if auto_owns(config, "perc_reserved_mem_max") and _number(cli.get("perc_reserved_mem_max")) <= 0:
            plan["override_reserved_ram"] = True
            desired = max(.50, 60 / ram) if is_h3 and model_def.get("minimax_h3_full_checkpoint") and ram >= 96 else default_ram_fraction
            plan["reserved_ram_fraction"] = round(max(.01, min(.65, desired, spare / ram)), 4)
            if spare < 4 and is_h3:
                plan["profile"] = 5
                allowed_profiles = [5]
                plan["warnings"].append("Very little host RAM is available; full model pinning is avoided (Profile 5).")
        if spare < 8:
            plan["warnings"].append("Low available host RAM can cause paging; close other memory-heavy applications before generating.")
    if history is not None:
        choice, count = history.choose(plan, vram, ram, allowed_profiles,
                                       max_budget_mb=int(cap_gb * 1024), clamped=clamped)
        plan["matching_renders"] = count
        if choice:
            plan.update(profile=choice["profile"], transformer_budget_mb=choice["budget"], source="local_history")
            plan["reasons"].append("Placement selected from repeated matching local renders with memory headroom.")
    return plan


class PerformanceHistory:
    """Bounded local SQLite evidence. Connections are short-lived/thread-safe."""
    def __init__(self, path, limit=500):
        self.path = Path(path)
        self.limit = limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS renders (id TEXT PRIMARY KEY, hardware TEXT, workload TEXT, created REAL, data TEXT)")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=3)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, job_id, plan, measurements, *, completed=False):
        # Cancelled/failed generations and missing device telemetry are not evidence.
        if not completed or not plan or measurements.get("samples", 0) < 2:
            return False
        if _number(measurements.get("denoising_seconds")) <= 0 or _number(measurements.get("peak_vram_gb")) <= 0:
            return False
        data = {"profile": plan["profile"], "budget": plan["transformer_budget_mb"],
                "reserved": plan.get("reserved_ram_fraction"), "read_ahead": plan.get("read_ahead"),
                **measurements}
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO renders VALUES (?, ?, ?, ?, ?)",
                       (str(job_id), plan["hardware_key"], plan["workload_key"], time.time(), json.dumps(data)))
            db.execute("DELETE FROM renders WHERE id NOT IN (SELECT id FROM renders ORDER BY created DESC LIMIT ?)", (self.limit,))
        return True

    def choose(self, plan, vram, ram, allowed_profiles, *, max_budget_mb, clamped=False):
        with self._connect() as db:
            rows = db.execute("SELECT data FROM renders WHERE hardware=? AND workload=? ORDER BY created DESC LIMIT 100",
                              (plan["hardware_key"], plan["workload_key"])).fetchall()
        groups = {}
        for row in rows:
            data = json.loads(row[0])
            # A new RAM-pressure plan must not reuse evidence from looser budgets.
            if data.get("reserved") != plan.get("reserved_ram_fraction") or data.get("read_ahead") != plan.get("read_ahead"):
                continue
            profile, budget = data["profile"], data["budget"]
            if profile not in allowed_profiles or budget > max_budget_mb or (clamped and budget):
                continue
            if data["peak_vram_gb"] > vram - max(.5, vram * .03) or _number(data.get("min_available_ram_gb")) < max(3, ram * .05):
                continue
            groups.setdefault((profile, budget), []).append(data)
        candidates = []
        for (profile, budget), samples in groups.items():
            if len(samples) >= 2:
                candidates.append({"profile": profile, "budget": budget,
                                   "seconds": statistics.median(item["denoising_seconds"] for item in samples),
                                   "vram": statistics.median(item["peak_vram_gb"] for item in samples)})
        # A single candidate has no measured comparison; don't call it a winner.
        if len(candidates) < 2:
            return None, len(rows)
        fastest = min(item["seconds"] for item in candidates)
        return min((item for item in candidates if item["seconds"] <= fastest * 1.03), key=lambda item: item["vram"]), len(rows)

    def summary(self, hw_key):
        with self._connect() as db:
            count, workloads = db.execute("SELECT count(*), count(DISTINCT workload) FROM renders WHERE hardware=?", (hw_key,)).fetchone()
        return {"completed_renders": count, "workloads": workloads,
                "minimum_samples": 2, "scope": "This hardware, runtime, checkpoint and matching workload"}

    def reset(self, hw_key):
        with self._connect() as db:
            db.execute("DELETE FROM renders WHERE hardware=?", (hw_key,))


class GenerationPerformanceMonitor:
    """Sample physical device memory, not PyTorch's custom-allocator counters.

    NVML is device-wide and polling can miss short spikes. Only denoising
    time drives comparisons, avoiding cold-load/encoding timing confounds.
    """
    def __init__(self, stats, phase, interval=1.0, clock=time.monotonic):
        self.stats, self.phase, self.interval, self.clock = stats, phase, interval, clock
        self.stop_event = threading.Event()
        self.thread = None
        self.started = clock()
        self.last = self.started
        self.last_denoising = False
        self.result = {"samples": 0, "peak_vram_gb": 0, "min_available_ram_gb": None,
                       "denoising_seconds": 0, "elapsed_seconds": 0,
                       "memory_scope": "Device-wide NVML samples; brief peaks may be missed"}

    def sample(self):
        now = self.clock()
        if self.last_denoising:
            self.result["denoising_seconds"] += now - self.last
        self.last = now
        phase = str(self.phase()).lower()
        self.last_denoising = "denois" in phase and "decod" not in phase
        try:
            stats = self.stats()
            gpu, ram = stats.get("gpu", {}), stats.get("ram", {})
            if gpu.get("available") and _number(gpu.get("vram_used_gb")) > 0:
                self.result["samples"] += 1
                self.result["peak_vram_gb"] = max(self.result["peak_vram_gb"], gpu["vram_used_gb"])
            available = ram.get("available_gb")
            if available is not None and _number(ram.get("total_gb")) > 0:
                previous = self.result["min_available_ram_gb"]
                self.result["min_available_ram_gb"] = min(previous, available) if previous is not None else available
        except Exception:
            pass

    def start(self):
        self.sample()
        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=run, name="maestro-performance", daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        self.sample()
        self.result["elapsed_seconds"] = self.clock() - self.started
        return {key: round(value, 3) if isinstance(value, float) else value
                for key, value in self.result.items()}
