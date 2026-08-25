"""Local LLM service using Qwen3.5-0.8B via llama-server subprocess.

Launches llama-server (from pre-built llama.cpp binaries) as a subprocess
and communicates via its OpenAI-compatible HTTP API.
Runs on CPU by default to avoid VRAM conflicts with WanGP.
"""

import os
import gc
import time
import subprocess
import threading
import logging
import requests
from typing import Optional

from services.text_integrity import repair_text

logger = logging.getLogger(__name__)

# Singleton state
_process: Optional[subprocess.Popen] = None
_lock = threading.Lock()
_model_id: str = ""
_device: str = ""
_server_port: int = 0
_vision_available: bool = False

# Rolling tail of llama-server's stdout/stderr, drained by a background
# thread once the server is up. Two jobs: (1) keep the OS pipe from
# filling — an undrained PIPE deadlocks the server after ~64 KB of logs,
# which looked like a permanent "frozen at Planning…" hang on long runs;
# (2) give us the last lines to quote when the subprocess dies mid-request
# so the pipeline error is actionable instead of a bare ConnectionError.
import collections as _collections
_server_log: "_collections.deque[str]" = _collections.deque(maxlen=200)
_log_reader: Optional[threading.Thread] = None

_GEMMA_TEMPLATE_COMPAT_MARKER = "detected an outdated gemma4 chat template"


def _is_benign_gemma_template_warning(line: str) -> bool:
    """Identify llama.cpp's non-fatal Gemma 4 compatibility notice."""

    return _GEMMA_TEMPLATE_COMPAT_MARKER in str(line or "").lower()

# Provider state: "local" | "remote" | "openai" | "anthropic"
_provider: str = "local"
_remote_url: str = ""       # Base URL for remote/OpenAI-compatible servers
_api_key: str = ""           # API key for OpenAI/Anthropic

# Auto-unload idle timer
_idle_timer: Optional[threading.Timer] = None
_idle_timeout: float = 60.0  # seconds before auto-unload

# Streaming state — accumulates tokens during generation
_stream_buffer: str = ""
_stream_done: bool = True
_stream_lock = threading.Lock()

# Last call state — for pipeline dashboard capture. The user prompt is
# captured alongside the system prompt so the Director Dashboard can
# render the full LLM input (system + user) for each pass, not just
# the system prompt. Without _last_user_prompt the dashboard's "Pass 1
# System Prompt" view was misleading — users saw only the rules, not
# the actual concept the LLM was being asked to expand into a script.
_last_system_prompt: str = ""
_last_user_prompt: str = ""
_last_thinking_text: str = ""

# Defaults — Gemma 4 4B as of 2026-05-03. Smaller (~5 GB weights vs the
# Qwen3.5 9B Opus build's ~6.85 GB), runs comfortably on lower-VRAM
# machines, and is fast enough that Director planning feels snappy
# rather than ponderous. Keep these in sync with the matching MODELS
# entry below (Abhiray/gemma-4-E4B-it-heretic-GGUF).
DEFAULT_HF_REPO = "Abhiray/gemma-4-E4B-it-heretic-GGUF"
DEFAULT_GGUF_FILE = "gemma-4-E4B-it-heretic-Q4_K_M.gguf"
DEFAULT_MMPROJ_FILE = "mmproj-F16.gguf"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(_BASE_DIR, "..", "ckpts", "llm")
DEFAULT_BIN_DIR = os.path.join(DEFAULT_CACHE_DIR, "bin")

import re as _re
# Matches all common thinking/reasoning tag patterns used by various models
_THINKING_TAG_RE = _re.compile(
    r"<(?:think|thinking|seed:think|reasoning|reflection)>[\s\S]*?"
    r"</(?:think|thinking|seed:think|reasoning|reflection)>\s*",
    _re.IGNORECASE,
)
_THINKING_TAG_UNCLOSED_RE = _re.compile(
    r"<(?:think|thinking|seed:think|reasoning|reflection)>[\s\S]*$",
    _re.IGNORECASE,
)
# Gemma 4 thinking tags: <|channel>thought\n...<channel|>
_GEMMA_THINKING_RE = _re.compile(
    r"<\|channel>thought\n[\s\S]*?<channel\|>\s*",
)
_GEMMA_THINKING_UNCLOSED_RE = _re.compile(
    r"<\|channel>thought\n[\s\S]*$",
)
# Capture-group variant — pulls the thought content WITHOUT the channel
# markers, so the dashboard can render thinking text cleanly. Used by
# the streaming finalizer; the strip-tags variants above keep their
# original non-capturing form to stay compatible with re.sub.
_GEMMA_THINKING_INNER_RE = _re.compile(
    r"<\|channel>thought\n([\s\S]*?)<channel\|>",
)

def _strip_thinking_tags(text: str) -> str:
    """Strip all known thinking/reasoning tag patterns from LLM output."""
    text = _THINKING_TAG_RE.sub("", text)
    text = _THINKING_TAG_UNCLOSED_RE.sub("", text)
    text = _GEMMA_THINKING_RE.sub("", text)
    text = _GEMMA_THINKING_UNCLOSED_RE.sub("", text)
    return text

# ─── VRAM estimator ────────────────────────────────────────────────────
# Each registry entry below carries:
#   weights_gb: GB on disk for the .gguf weights (≈ VRAM when loaded)
#   mmproj_gb:  GB for the vision projector (0 if no mmproj)
#   arch:       key into LLM_ARCHITECTURES used to size the KV cache
# Plus optional extra_flags whose -c / --cache-type-{k,v} values are
# parsed for the KV cache. The size_hint string in the dropdown is built
# at module load by _build_size_hint(info) so it always reflects the
# *configured* context — change the -c flag and the hint updates with it.

# Approximate per-architecture parameters used for KV-cache sizing.
# Formula: kv_bytes = 2 * layers * kv_heads * head_dim * ctx * dtype_bytes
# `sliding_window` (if set) caps the per-token cache for "local" layers
# at that window size, while `global_layer_ratio` is the fraction of
# layers that use full context — Gemma 3/4 alternate ~5 local : 1 global.
LLM_ARCHITECTURES = {
    # (layers, kv_heads, head_dim, sliding_window or None, global_layer_ratio)
    "qwen3-2b":   {"layers": 28, "kv_heads": 4,  "head_dim": 128, "sliding_window": None, "global_layer_ratio": 1.0},
    "qwen3-4b":   {"layers": 36, "kv_heads": 8,  "head_dim": 128, "sliding_window": None, "global_layer_ratio": 1.0},
    "qwen3-9b":   {"layers": 36, "kv_heads": 8,  "head_dim": 128, "sliding_window": None, "global_layer_ratio": 1.0},
    "qwen3-27b":  {"layers": 48, "kv_heads": 8,  "head_dim": 128, "sliding_window": None, "global_layer_ratio": 1.0},
    # Gemma 3/4: 5:1 local:global attention pattern, local window 4096.
    # Most layers' KV is bounded by the window regardless of total ctx.
    "gemma4-2b":  {"layers": 26, "kv_heads": 4,  "head_dim": 256, "sliding_window": 4096, "global_layer_ratio": 1/6},
    "gemma4-4b":  {"layers": 34, "kv_heads": 4,  "head_dim": 256, "sliding_window": 4096, "global_layer_ratio": 1/6},
    # Gemma 4 12B "unified" — 48 layers per the model card. kv_heads/head_dim
    # mirror Gemma 3 12B as an approximation; this only feeds the VRAM size hint,
    # not functional loading, so refine if the real config differs.
    "gemma4-12b": {"layers": 48, "kv_heads": 8,  "head_dim": 256, "sliding_window": 4096, "global_layer_ratio": 1/6},
    "gemma4-27b": {"layers": 62, "kv_heads": 16, "head_dim": 128, "sliding_window": 4096, "global_layer_ratio": 1/6},
}


def _kv_dtype_bytes(t: str) -> float:
    """Bytes per KV cache element for llama.cpp cache-type strings."""
    t = (t or "").lower().strip()
    return {
        "f32":  4.0, "fp32": 4.0,
        "f16":  2.0, "fp16": 2.0, "bf16": 2.0,
        "q8_0": 1.0625,
        "q5_0": 0.6875, "q5_1": 0.6875,
        "q4_0": 0.5625, "q4_1": 0.5625,
    }.get(t, 2.0)


def _estimate_kv_gb(arch_key: str, extra_flags: list) -> float:
    """Estimate KV cache size in GB for the given arch + llama.cpp flags.

    Uses a realistic-usage context cap rather than the configured maximum.
    llama.cpp lazy-grows the KV cache as tokens are processed, so a model
    configured with `-c 262144` only allocates that full 13+ GB if a user
    actually fills 256k tokens of conversation. Empirical measurement
    (e.g. Qwen3.6 27B @ 256k → 21.8 GB total) shows real KV usage tracks
    typical session lengths much more closely than the configured max.
    Capping at 64k gives a number that buckets correctly to the GPU tier
    a user actually needs to run the model comfortably.
    """
    arch = LLM_ARCHITECTURES.get(arch_key)
    if not arch:
        return 0.0
    # Defaults match llama.cpp's: ctx=4096, fp16 cache.
    ctx = 4096
    cache_dtype = "f16"
    flags = list(extra_flags or [])
    for i, f in enumerate(flags):
        if f == "-c" and i + 1 < len(flags):
            try:
                ctx = int(flags[i + 1])
            except (TypeError, ValueError):
                pass
        elif f in ("--cache-type-k", "--cache-type-v") and i + 1 < len(flags):
            cache_dtype = flags[i + 1]  # assume k & v match (they do in our configs)
    # Cap at 64k for the estimate. Models configured with bigger contexts
    # (e.g. 256k Qwen) won't actually use that much KV in typical use,
    # and bucketing the estimate to a GPU tier becomes more accurate
    # this way. Bumping a model's -c above 64k won't change the
    # displayed VRAM tier — that's deliberate.
    TYPICAL_USAGE_CTX_CAP = 65536
    effective_ctx = min(ctx, TYPICAL_USAGE_CTX_CAP)
    bpe = _kv_dtype_bytes(cache_dtype)
    layers = arch["layers"]
    kv_heads = arch["kv_heads"]
    head_dim = arch["head_dim"]
    window = arch.get("sliding_window")
    global_ratio = arch.get("global_layer_ratio", 1.0)
    if window and effective_ctx > window and global_ratio < 1.0:
        # Mixed local/global attention (Gemma): apportion KV between
        # local layers (capped at the window) and global layers (full ctx).
        global_layers = max(1, round(layers * global_ratio))
        local_layers = layers - global_layers
        bytes_total = 2 * bpe * kv_heads * head_dim * (
            global_layers * effective_ctx + local_layers * window
        )
    else:
        bytes_total = 2 * layers * kv_heads * head_dim * effective_ctx * bpe
    return bytes_total / (1024 ** 3)


# Standard consumer/workstation GPU VRAM tiers. A model's displayed
# requirement is rounded UP to the smallest tier that comfortably runs
# it. This communicates "what GPU do I need?" much better than a precise
# decimal — a user with a 12 GB card sees "12 GB VRAM" and immediately
# knows it fits, vs. seeing "9.96 GB" and having to do mental math about
# headroom.
#
# Headroom note: the bucket is the recommended *minimum*. Maestro's
# generation pipelines also need VRAM concurrently if you're running an
# LLM during video gen — in that case, pick a card with the LLM's bucket
# size PLUS your video model's footprint, or run the LLM on a remote
# host.
GPU_VRAM_TIERS = (6, 8, 12, 16, 24, 32, 48, 80)


def _bucket_to_tier(gb: float) -> int:
    """Round a measured-or-estimated VRAM total up to the smallest GPU
    tier that fits it. Returns 6 for tiny models, 80 for very large.
    Beyond 80 GB it just rounds up to the next 16 GB step."""
    for tier in GPU_VRAM_TIERS:
        if gb <= tier:
            return tier
    last = GPU_VRAM_TIERS[-1]
    extra = gb - last
    return last + int(((extra + 15.999) // 16) * 16)


def _build_size_hint(info: dict) -> str:
    """Compose the dropdown's '~N GB VRAM' string from registry metadata.

    Bucketed to standard GPU tiers (6 / 8 / 12 / 16 / 24 / 32 / 48 / 80)
    rather than reporting a precise estimate, so users can match their
    hardware at a glance. Falls back to a manual `size_hint` field if
    the entry doesn't carry enough metadata to compute (e.g. legacy
    entries or remote models).
    """
    if "weights_gb" not in info:
        return info.get("size_hint", "")
    weights = float(info.get("weights_gb", 0))
    mmproj = float(info.get("mmproj_gb", 0))
    kv = _estimate_kv_gb(info.get("arch", ""), info.get("extra_flags", []))
    total = weights + mmproj + kv
    return f"{_bucket_to_tier(total)} GB VRAM"


# Model registry — maps HF repo IDs to their GGUF filenames.
# size_hint is built automatically from weights_gb + mmproj_gb + KV-cache
# estimate at module load (see post-loop below).
MODEL_REGISTRY = {
    "unsloth/Qwen3.5-2B-GGUF": {
        "label": "Qwen3.5 2B (Fast)",
        "gguf_file": "Qwen3.5-2B-Q4_K_S.gguf",
        "weights_gb": 1.13, "mmproj_gb": 0.0, "arch": "qwen3-2b",
    },
    "unsloth/Qwen3.5-4B-GGUF": {
        "label": "Qwen3.5 4B (Balanced)",
        "gguf_file": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "weights_gb": 2.9, "mmproj_gb": 0.0, "arch": "qwen3-4b",
    },
    "mradermacher/Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated-i1-GGUF": {
        "label": "Qwen3.5 9B Claude Opus Abliterated Q6_K",
        "gguf_file": "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated.i1-Q6_K.gguf",
        "mmproj_file": "mmproj-Q8_0.gguf",
        "weights_gb": 6.85, "mmproj_gb": 0.58, "arch": "qwen3-9b",
        "cache_dir_override": "Huihui-Qwen3.5-9B-Claude-4.6-Opus-abliterated",
        "extra_flags": [
            "-c", "65536",
            "-np", "1",
            "-fa", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
    "Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF": {
        "label": "Qwen3.6 27B Abliterated Heretic (Uncensored, Vision)",
        "gguf_file": "Qwen3.6-27B-Abliterated-Heretic-Uncensored-Q4_K_M.gguf",
        # The Heretic GGUF repo doesn't ship an mmproj file, but the base
        # Qwen3.6-27B vision architecture is preserved in the abliterated
        # weights — so pull the mmproj from the upstream unsloth GGUF repo.
        "mmproj_file": "mmproj-BF16.gguf",
        "mmproj_repo": "unsloth/Qwen3.6-27B-GGUF",
        "weights_gb": 15.4, "mmproj_gb": 0.87, "arch": "qwen3-27b",
        # Qwen3.6 inherits Qwen3.5's 256k native context. Note: full 256k
        # context is the dominant VRAM cost here — that single -c flag
        # alone allocates ~15 GB of KV cache even with q4_0 quantization.
        "extra_flags": [
            "-c", "262144",
            "-np", "1",
            "-fa", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
    "Nesuwka/gemma-4-E2B-it-heretic-ara-Q4_K_M-GGUF": {
        "label": "Gemma 4 2B Heretic Uncensored (Vision, Tiny)",
        "gguf_file": "model-q4_k_m.gguf",
        "mmproj_file": "mmproj-gemma-4-e2b-it-f16.gguf",
        "mmproj_repo": "ggml-org/gemma-4-E2B-it-GGUF",
        "weights_gb": 3.4, "mmproj_gb": 1.0, "arch": "gemma4-2b",
        "thinking_style": "gemma",
        # Gemma 4 was tuned at temp=1.0; running below that leaves it
        # more deterministic than its sweet spot. frequency/presence
        # penalty=0 because Gemma 4 doesn't have the Qwen 3.x repetition-
        # cascade pathology, and the OpenAI-style penalties dampen
        # reasoning-vocabulary diversity in long thinking — observed
        # empirically as shallow Pass 1 thinking compared to LM Studio
        # output (which ships with no penalty by default).
        "sampling_defaults": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "frequency_penalty": 0, "presence_penalty": 0,
        },
    },
    "Abhiray/gemma-4-E4B-it-heretic-GGUF": {
        "label": "Gemma 4 4B Heretic Uncensored (Vision, Fast) (Recommended)",
        "gguf_file": "gemma-4-E4B-it-heretic-Q4_K_M.gguf",
        "mmproj_file": "mmproj-F16.gguf",
        "weights_gb": 4.97, "mmproj_gb": 0.92, "arch": "gemma4-4b",
        "thinking_style": "gemma",
        "sampling_defaults": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "frequency_penalty": 0, "presence_penalty": 0,
        },
    },
    "SulphurAI/Sulphur-2-base": {
        # Sulphur-2's own uncensored prompt enhancer — a ~9.6B multimodal
        # (text+image) llama.cpp model the checkpoint author trained to prompt
        # the Sulphur-2 LTX-2.3 finetune. Used in raw-passthrough mode (no
        # system prompt) by any gen model that declares
        # prompt_enhancer_model: "SulphurAI/Sulphur-2-base". The GGUFs live in
        # the repo's prompt_enhancer_uncensored/ subfolder (hf_hub_download
        # handles the subfolder path). No `arch` key → the KV size hint is
        # skipped (the base arch isn't published); weights+mmproj still count.
        "label": "Sulphur-2 Uncensored Prompt Enhancer (Vision)",
        "gguf_file": "prompt_enhancer_uncensored/prompt_enhancer_uncensored-q8_0.gguf",
        "mmproj_file": "prompt_enhancer_uncensored/mmproj-prompt_enhancer_uncensored.gguf",
        "weights_gb": 9.79, "mmproj_gb": 0.92,
        "sampling_defaults": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "frequency_penalty": 0, "presence_penalty": 0,
        },
    },
    "Jiunsong/supergemma4-26b-uncensored-gguf-v2": {
        "label": "Gemma 4 26B MoE Uncensored (Fast)",
        "gguf_file": "supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
        # No mmproj — the upstream repo doesn't publish one, and the
        # supergemma fine-tune fused the vision adapter back into the
        # base weights. Text-only model.
        "weights_gb": 16.8, "mmproj_gb": 0.0, "arch": "gemma4-26b",
        # Use the embedded jinja template (default behavior — no
        # disable_jinja flag set). Earlier attempts tried forcing
        # `--chat-template gemma` to override what appeared to be a
        # broken "Neutral" embedded template, but that produced a
        # near-empty 3-token prompt at runtime — llama-server's gemma
        # template either isn't recognized in this build or doesn't
        # round-trip OpenAI-format messages correctly. The embedded
        # template is fine; the original failure was missing
        # repeat_penalty (below) AND no thinking budget allocation.
        #
        # `thinking_style: "gemma_prefix"` — the SuperGemma fine-tune's
        # embedded "Neutral" chat template doesn't honor
        # chat_template_kwargs.enable_thinking. Verified empirically:
        # with thinking_style="gemma" (kwarg-based activation), the
        # streaming finalizer reported `reasoning_content: 0 chars,
        # gemma_inline_thinking: 0 chars` despite the kwarg being sent.
        # The model just generated 29,521 chars of raw content that
        # eventually degraded into Unicode garbage.
        #
        # This is still a Gemma 4 variant — same trained-in thinking
        # activation token (`<|think|>`) as the Heretic / Abliterated
        # fine-tunes whose templates DO honor the kwarg. The
        # `gemma_prefix` style reproduces what the kwarg-aware path
        # would have done: injects the literal `<|think|>` at the top
        # of the system prompt. llama.cpp's tokenizer recognizes it as
        # the model's special activation token regardless of whether
        # the string came from the chat template's emission or from
        # raw system-prompt text.
        #
        # If a future fine-tune of this model fixes the template to
        # honor enable_thinking, flip back to thinking_style="gemma".
        "thinking_style": "gemma_prefix",
        #
        # Sampling: escalated past LM Studio's defaults because
        # Director Pass 1 in Music Video mode sends a 5-10k token
        # prompt (rules + audio analysis + structured JSON ask) that
        # most chat-app interactions never approach. LM Studio's
        # baseline `repeat_penalty: 1.1` over a 64-token window was
        # enough for classification (200-token prompts) but Pass 1
        # still collapsed into single-suffix repetition like
        # "ness ness ness ness..." — a common BPE token the model
        # locks onto when the long prompt overwhelms attention.
        #
        # Stack:
        #   - temperature: 0.7 (down from 0.8) — less random sampling
        #     starves the cascade of variation that lets it rebuild
        #     after the penalty knocks it down
        #   - top_p: 0.92 (down from 0.95) — tighter nucleus
        #   - top_k: 40, min_p: 0.05 — LM Studio standard
        #   - repeat_penalty: 1.15 (up from 1.1) — stronger logit
        #     divisor against repeated tokens
        #   - repeat_last_n: 256 (default is 64) — wider penalty
        #     window protects against word-spaced repetition that
        #     can build up across multi-sentence stretches
        #   - frequency_penalty: 0.1 — OpenAI-style accumulating
        #     penalty as belt-and-suspenders against the same token
        #     surviving the repeat_penalty hit
        #   - presence_penalty: 0.1 — diversity nudge
        #
        # These are more aggressive than LM Studio's defaults, which
        # is fine: LM Studio's default user is doing chat, not
        # 8k-token Director planning. We're paying for the larger
        # prompt with tighter sampling.
        "sampling_defaults": {
            "temperature": 0.7, "top_p": 0.92, "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
            "frequency_penalty": 0.1, "presence_penalty": 0.1,
        },
        # 26B-A4B MoE: 26B total params on disk (16.8 GB at Q4_K_M),
        # but only ~4B active per token thanks to the mixture-of-
        # experts routing → generation speed closer to a 4B model
        # despite the bigger weight footprint and 26B-scale quality.
        #
        # KV cache: q8_0 instead of q4_0. MoE attention is more
        # sensitive to cache lossy-ness because routing decisions are
        # amplified through the small active expert subset.
        #
        # Context: 32k. Director Mode screenplay generation rarely
        # needs more, and the lower ceiling offsets the q8_0 memory
        # bump so VRAM stays comparable to the 31B Abliterated entry.
        "extra_flags": [
            "-c", "32768",
            "-np", "1",
            "-fa", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
        ],
    },
    "paperscarecrow/Gemma-4-31B-it-abliterated-gguf": {
        "label": "Gemma 4 31B Abliterated Q4_K_M (Vision)",
        "gguf_file": "gemma-4-31b-abliterated-Q4_K_M.gguf",
        "mmproj_file": "mmproj-gemma-4-31B-it-f16.gguf",
        "mmproj_repo": "ggml-org/gemma-4-31B-it-GGUF",  # mmproj from official repo
        "weights_gb": 17.4, "mmproj_gb": 1.12, "arch": "gemma4-27b",
        "thinking_style": "gemma",
        "sampling_defaults": {
            "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "frequency_penalty": 0, "presence_penalty": 0,
        },
        "extra_flags": [
            "-c", "65536",
            "-np", "1",
            "-fa", "on",
            "--cache-type-k", "q4_0",
            "--cache-type-v", "q4_0",
        ],
    },
    "mradermacher/gemma-4-12B-it-abliterated-uncensored-i1-GGUF": {
        # EXPERIMENTAL — not recommended for Director yet. It loads and runs, but
        # on the structured Director pipeline it under-writes Pass 1 (~4.5k chars
        # vs the 4B's ~10.6k on an identical 5-min prompt) and loops on Pass 2
        # JSON (~96k chars -> failed fallback) even with low temp + strong repeat
        # penalties. Root cause is model/architecture, NOT sampling (three sampling
        # passes did not fix it): gemma4_unified support in llama.cpp is brand-new
        # (PR #24118, 2026-06-04) and this is an abliterated build, which erodes
        # long-form structured coherence. Kept selectable; revisit when llama.cpp's
        # unified support matures. For Director use the 4B / 26B / 31B entries.
        "label": "Gemma 4 12B Abliterated (Text, Experimental)",
        "gguf_file": "gemma-4-12B-it-abliterated-uncensored.i1-Q4_K_M.gguf",
        # Encoder-free "gemma4_unified" architecture: its multimodal projector
        # ships in the new `gemma4uv` format (NOT a standard mmproj-*.gguf), and
        # llama.cpp's unified vision/audio support is brand-new (PR #24118,
        # 2026-06-04). Registered TEXT-ONLY for now (no mmproj).
        # IMPORTANT: requires a llama-server build from AFTER 2026-06-04 — older
        # builds cannot load gemma4_unified at all and will fail at load time.
        "weights_gb": 7.5, "mmproj_gb": 0.0, "arch": "gemma4-12b",
        # Template (verified from the GGUF) honors enable_thinking and activates
        # with <|think|>; Maestro already sends the kwarg + launches with --jinja,
        # so "gemma" (kwarg) activation is correct here — do NOT switch to
        # gemma_prefix (this template emits an empty <|channel>thought<channel|>
        # when thinking is off, which would fight a manually-injected token).
        "thinking_style": "gemma",
        # Repeat-loop fix — COMPLEMENT the caller, don't clobber it.
        # registry sampling_defaults OVERRIDE per-call values (see
        # _apply_sampling_defaults), and the Director passes are tuned per pass:
        # Pass 2 (_call_llm_json) NEEDS low temp 0.7 + frequency_penalty 0.3 to
        # keep structured JSON from looping. An earlier version set temperature
        # 1.0 + frequency_penalty 0.1 here, which overrode that and produced a
        # 122K-char JSON repeat loop; freq/presence penalties ALSO shrink
        # Gemma's reasoning depth (shallow thinking on Pass 1). So set ONLY
        # llama-native anti-loop (repeat_penalty + min_p) that adds on top of
        # whatever each pass requests, plus Google's top_p/top_k. Deliberately
        # NO temperature / frequency_penalty / presence_penalty so each pass
        # keeps its own tuned values.
        "sampling_defaults": {
            "top_p": 0.95, "top_k": 64,
            "min_p": 0.05,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
        },
    },
}

# Build size_hint strings once at module load. Re-runs if you `import importlib;
# importlib.reload(llm_service)` after editing the registry.
for _repo_id, _info in MODEL_REGISTRY.items():
    _info["size_hint"] = _build_size_hint(_info)


# ── Curated public model catalog ────────────────────────────────────
# The local models that appear in the LLM picker, in this order. Any other
# MODEL_REGISTRY entry stays loadable by id but is HIDDEN from the dropdown:
# this covers functional-only entries (e.g. the Sulphur-2 dedicated prompt
# enhancer, referenced by gen models via prompt_enhancer_model) and
# deprecated / experimental variants. A repo id listed here that isn't
# currently in the registry is simply skipped.
_PUBLIC_MODEL_ORDER = [
    "Youssofal/Qwen3.6-27B-Abliterated-Heretic-Uncensored-GGUF",
    "Nesuwka/gemma-4-E2B-it-heretic-ara-Q4_K_M-GGUF",
    "Abhiray/gemma-4-E4B-it-heretic-GGUF",                         # default (Recommended)
    "Jiunsong/supergemma4-26b-uncensored-gguf-v2",
    "paperscarecrow/Gemma-4-31B-it-abliterated-gguf",
]


def get_available_models(provider: str = "local", remote_url: str = "", api_key: str = "") -> list:
    """Return list of available LLM model options for the UI.

    For local provider, returns the curated built-in catalog
    (_PUBLIC_MODEL_ORDER). For remote/openai, queries the server's
    /v1/models endpoint. For anthropic, returns a curated Claude list.
    """
    local_models = [
        {
            "id": repo_id,
            "label": MODEL_REGISTRY[repo_id]["label"],
            "size_hint": MODEL_REGISTRY[repo_id]["size_hint"],
            "provider": "local",
        }
        for repo_id in _PUBLIC_MODEL_ORDER
        if repo_id in MODEL_REGISTRY
    ]

    remote_models: list[dict] = []

    # Query remote OpenAI-compatible server (LM Studio, etc.)
    if provider in ("remote", "openai") and remote_url:
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            url = remote_url.rstrip("/")
            resp = requests.get(f"{url}/v1/models", headers=headers, timeout=10)
            if resp.ok:
                data = resp.json()
                for m in data.get("data", []):
                    mid = m.get("id", "")
                    if mid:
                        remote_models.append({
                            "id": mid,
                            "label": f"{mid} (Remote)" if provider == "remote" else f"{mid} (OpenAI)",
                            "size_hint": provider,
                            "provider": provider,
                        })
        except Exception as e:
            print(f"[LLM] Failed to query remote models at {remote_url}: {e}")

    # Anthropic models (curated list — no /models endpoint)
    if provider == "anthropic" and api_key:
        remote_models.extend([
            {"id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "size_hint": "anthropic", "provider": "anthropic"},
            {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "size_hint": "anthropic", "provider": "anthropic"},
        ])

    return local_models + remote_models


def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_model_dir() -> str:
    d = os.environ.get("MAESTRO_LLM_CACHE", DEFAULT_CACHE_DIR)
    os.makedirs(d, exist_ok=True)
    return d


PROVIDER_API_KEY_SETTING = {
    "remote": "llm_remote_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


def provider_api_key(provider: str, services: dict) -> str:
    """Return only the credential owned by the selected LLM provider."""

    setting = PROVIDER_API_KEY_SETTING.get(str(provider or "").lower())
    if not setting or not isinstance(services, dict):
        return ""
    return str(services.get(setting, "") or "")


_OPENAI_CHAT_FIELDS = frozenset({
    "messages",
    "model",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stream_options",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
    "user",
})


def _finalize_payload(payload: dict) -> dict:
    """Translate Maestro's llama.cpp request into OpenAI-compatible form.

    Local llama-server accepts additional sampler controls and serves only one
    model, so its native payload remains untouched. Hosted and third-party
    OpenAI-compatible gateways usually require ``model`` and often reject
    llama.cpp-only fields such as ``cache_prompt`` and ``min_p``.
    """

    if _provider not in ("remote", "openai"):
        return payload
    prepared = {
        key: value
        for key, value in payload.items()
        if key in _OPENAI_CHAT_FIELDS
    }
    dropped = sorted(set(payload) - set(prepared))
    if dropped:
        print(
            f"[LLM] Dropped {len(dropped)} llama.cpp-only field(s) not "
            f"accepted by provider={_provider}: {', '.join(dropped)}"
        )
    prepared["model"] = _model_id
    return prepared


def _server_url() -> str:
    if _provider in ("remote", "openai") and _remote_url:
        return _remote_url.rstrip("/")
    return f"http://127.0.0.1:{_server_port}"


def _api_headers() -> dict:
    """Build headers for API calls (adds auth for remote providers)."""
    headers = {"Content-Type": "application/json"}
    if _provider in ("remote", "openai", "anthropic") and _api_key:
        if _provider == "anthropic":
            headers["x-api-key"] = _api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {_api_key}"
    return headers


def _active_registry_entry() -> dict:
    """Return the MODEL_REGISTRY entry for the currently loaded model, or {}."""
    return MODEL_REGISTRY.get(_model_id, {})


def _apply_model_defaults(temperature: float, top_p: float, payload: dict) -> tuple[float, float]:
    """Apply per-model sampling defaults from the registry.

    Registry values WIN over caller values for any field the registry
    specifies. Rationale: caller values are pass-level heuristics (e.g.
    Pass 1 sends temperature=0.8 + frequency_penalty=0.15 because that
    works well for Qwen and is the historical default) but registry
    values are model-tuned (Gemma 4 was tuned at 1.0; the freq penalty
    that protects Qwen 3.x from repetition cascades shrinks Gemma's
    reasoning-vocabulary diversity and produces shallow thinking output).

    Models with no `sampling_defaults` entry pass through unchanged —
    caller's values stay. So adding registry tuning for one model never
    affects others.

    Returns adjusted (temperature, top_p) and mutates payload with
    top_k / frequency_penalty / presence_penalty when present.
    """
    entry = _active_registry_entry()
    defaults = entry.get("sampling_defaults", {})
    if not defaults:
        return temperature, top_p
    if "temperature" in defaults:
        temperature = defaults["temperature"]
    if "top_p" in defaults:
        top_p = defaults["top_p"]
    if "top_k" in defaults:
        payload["top_k"] = defaults["top_k"]
    # Penalty overrides — `0` is a valid value (means "explicitly off")
    # so we use `in defaults` rather than truthiness checks here. Caller's
    # frequency_penalty / presence_penalty (set later in generate_streaming)
    # need to be cleared if registry says off; otherwise they'd stick.
    if "frequency_penalty" in defaults:
        fp = defaults["frequency_penalty"]
        if fp > 0:
            payload["frequency_penalty"] = fp
        else:
            payload.pop("frequency_penalty", None)
    if "presence_penalty" in defaults:
        pp = defaults["presence_penalty"]
        if pp > 0:
            payload["presence_penalty"] = pp
        else:
            payload.pop("presence_penalty", None)
    # llama.cpp-native sampling parameters (forwarded by llama-server's
    # OpenAI-compatible endpoint as request extensions). These are NOT
    # standard OpenAI fields but llama-server passes them through to the
    # sampler. Necessary for models that need llama.cpp-native repetition
    # control rather than the weaker OpenAI-style presence/frequency
    # penalties — e.g. MoE Gemma fine-tunes that LM Studio handles via
    # its default `repeat_penalty: 1.1` + `min_p: 0.05` config.
    if "repeat_penalty" in defaults:
        rp = defaults["repeat_penalty"]
        if rp and rp != 1.0:
            payload["repeat_penalty"] = rp
        else:
            payload.pop("repeat_penalty", None)
    if "repeat_last_n" in defaults:
        # Width of the recent-tokens window that repeat_penalty applies
        # to. Default in llama.cpp is 64. Wider windows protect against
        # word-spaced repetition (where the model emits "X foo X bar X
        # baz" — within 64 tokens the repeated X stays inside the window
        # and gets penalized; widen to 256+ for protection against
        # paragraph-scale repetition).
        rln = defaults["repeat_last_n"]
        if rln and rln > 0:
            payload["repeat_last_n"] = rln
        else:
            payload.pop("repeat_last_n", None)
    if "min_p" in defaults:
        mp = defaults["min_p"]
        if mp and mp > 0:
            payload["min_p"] = mp
        else:
            payload.pop("min_p", None)
    return temperature, top_p


def _prepare_thinking(system_prompt: str, enable_thinking: Optional[bool], thinking_budget: int) -> tuple[str, Optional[bool], int]:
    """Handle model-specific thinking mode activation.

    Gemma 4 (incl. Heretic / abliterated fine-tunes): activate thinking
    by passing `enable_thinking=True` via `chat_template_kwargs`. The
    chat template embedded in the Heretic GGUF emits the literal
    `<|think|>` directive at the top of the system turn when this kwarg
    is true (verified by inspecting tokenizer.chat_template — it has an
    explicit `{%- if enable_thinking -%}{{- '<|think|>' -}}{%- endif -%}`
    block). Once activated, the model emits its reasoning inline as
    `<|channel>thought\\n...<channel|>` and switches to its actual
    answer after the closing marker.

    Do NOT inject `<|think|>` into the system message text WHEN THE
    CHAT TEMPLATE ALREADY EMITS IT — for Heretic/Abliterated fine-tunes
    whose templates honor enable_thinking, manual injection is
    redundant and risks double-tokenization. For fine-tunes whose
    templates ignore the kwarg entirely (e.g. SuperGemma's "Neutral"
    template), see the separate `gemma_prefix` style below — that path
    relies on llama.cpp's tokenizer recognizing the special token even
    when it appears in raw system-prompt text rather than from the
    chat template's emission.

    Qwen 3.x: same `enable_thinking` kwarg path. The Qwen chat template
    inserts `<think>` automatically when the kwarg is true.

    Returns (system_prompt, enable_thinking, thinking_budget).
    """
    entry = _active_registry_entry()
    # Force-off wins over everything else (caller's explicit value AND any
    # thinking_style setting). Used for models that auto-activate thinking
    # mode regardless of chat_template_kwargs — Gemma 4 fine-tunes are the
    # known offender. The model burns the entire max_new_tokens budget on
    # internal reasoning, stuffs the reasoning into `reasoning_content`,
    # and returns empty `content`. The result is a successful HTTP 200
    # with no useful output for the pipeline. Set `disable_thinking: True`
    # in the registry entry to suppress this — propagates as
    # enable_thinking=False to the chat template kwargs, and the
    # companion stop-tokens injection in generate() / generate_streaming()
    # catches any chat templates that ignore the kwarg.
    if entry.get("disable_thinking", False):
        return system_prompt, False, 0
    style = entry.get("thinking_style", "qwen")
    if style == "gemma":
        # Honor explicit opt-out from caller.
        if enable_thinking is False:
            return system_prompt, False, 0
        if thinking_budget <= 0:
            thinking_budget = 2048
        enable_thinking = True
        # Strip any leftover literal `<|think|>` prefix from previous
        # versions of this function — the chat template will inject the
        # real special-tokenized version. Leaving the literal text in
        # would result in the prefix appearing twice (once tokenized, once
        # as plain characters), which can confuse the model.
        stripped = system_prompt.lstrip()
        if stripped.startswith("<|think|>"):
            system_prompt = stripped[len("<|think|>"):].lstrip("\n")
    elif style == "gemma_prefix":
        # For Gemma 4 fine-tunes whose embedded chat template doesn't
        # honor chat_template_kwargs.enable_thinking — symptom: both
        # "gemma" and "qwen" styles produce 0 chars of reasoning_content
        # because the kwarg never reaches the model.
        #
        # Strategy: inject the canonical Gemma 4 thinking-activation
        # token literally at the top of the system prompt. The base
        # Gemma 4 chat templates do exactly the same thing when the
        # enable_thinking kwarg is honored — they emit the string
        # `<|think|>` at the top of the system turn — so injecting it
        # ourselves reproduces what the kwarg-aware path would have
        # done. llama.cpp's tokenizer recognizes `<|think|>` as a
        # special token from the model's added_tokens, and tokenizes
        # the injected text the same way regardless of whether it
        # came from a chat template or raw user text.
        #
        # The earlier "DO NOT inject <|think|>" comment in this file
        # was based on testing against the Heretic fine-tune, where
        # the kwarg already worked and manual injection was redundant
        # + risked double-tokenization. For models where the kwarg
        # doesn't work at all (SuperGemma's "Neutral" template), this
        # is the right and only option.
        #
        # Honor explicit opt-out from caller.
        if enable_thinking is False:
            return system_prompt, False, 0
        # Return None for enable_thinking — the calling code (generate /
        # generate_streaming) only includes the kwarg in the payload when
        # `enable_thinking is not None`. Returning False here would send
        # `enable_thinking: False` to the chat template, which on any
        # template that DOES honor the kwarg would EXPLICITLY DISABLE
        # thinking — actively fighting our `<|think|>` prefix injection.
        # Returning None skips the kwarg entirely, letting the literal
        # token activation be the sole signal.
        enable_thinking = None
        if thinking_budget <= 0:
            # Empirical sizing for SuperGemma-style verbose thinkers:
            # at 4096 the classification call (caller max_new_tokens=400)
            # produced 14,598 chars (~3650 tokens) of reasoning_content
            # and hit max_tokens still mid-thought — never reached the
            # closing `<channel|>` marker, parser put everything into
            # reasoning_content, content came back empty. Bumping the
            # default to 12288 gives the model room to wrap its
            # reasoning AND emit the final answer; if a future caller
            # needs more, it can override explicitly.
            #
            # Why so much higher than the "gemma" style's 2048 default:
            #   - "gemma" path uses framework-level activation; thinking
            #     streams via the separate `reasoning_content` channel
            #     and doesn't compete with the content for max_tokens.
            #   - "gemma_prefix" path uses inline activation; thinking
            #     and content share one max_tokens budget, so the
            #     budget must cover BOTH.
            #   - SuperGemma in particular is more verbose than the
            #     Heretic variants — observed 3500+ tokens for what
            #     should be a short classification task.
            thinking_budget = 12288
        # Avoid double-prefixing if the activator is already present
        # (e.g. a previous pass injected it, or the caller did manually).
        stripped = system_prompt.lstrip()
        if not stripped.startswith("<|think|>"):
            system_prompt = "<|think|>\n" + system_prompt
    return system_prompt, enable_thinking, thinking_budget


def is_loaded() -> bool:
    if _provider in ("remote", "openai", "anthropic"):
        return bool(_model_id)
    return _process is not None and _process.poll() is None


def get_status() -> dict:
    return {
        "loaded": is_loaded(),
        "model_id": _model_id or None,
        "device": _device if is_loaded() else None,
        "provider": _provider,
    }


def _download_gguf(repo_id: str, filename: str, cache_dir: str) -> str:
    """Download a GGUF file from HuggingFace and return the local path."""
    local_path = os.path.join(cache_dir, filename)
    if os.path.isfile(local_path):
        print(f"[LLM] GGUF file already cached: {local_path}")
        return local_path

    print(f"[LLM] Downloading {filename} from {repo_id}...")
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=cache_dir,
    )
    print(f"[LLM] Downloaded to: {downloaded}")
    return downloaded


# Minimum llama.cpp build Maestro requires. Builds below this lack correct
# support for newer model architectures we ship — notably Qwen3.5's hybrid
# attention/SSM arch ("qwen35") used by the Sulphur prompt enhancer, which
# crashes on older builds with: "error loading model: missing tensor
# 'blk.N.ssm_conv1d.weight'". Verified b9632 loads it; b9048 does not. Bump
# this (and FALLBACK_TAG below) when a newer model needs a newer runtime.
MIN_LLAMA_BUILD = 9632

_LLAMA_RUNTIME_RECEIPT = ".maestro_llama_runtime.json"
_WINDOWS_LLAMA_CUDA_FILES = (
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
)


def _positive_llama_build(value: str):
    """Extract a meaningful llama.cpp build number from text.

    Some official release binaries report ``version: 0 (unknown)`` even
    though the archive itself is tagged with a current ``bNNNN`` release.
    Build zero is missing metadata, not evidence that the binary predates
    every real llama.cpp release.
    """

    import re

    match = re.search(r"version:\s*b?(\d+)\b", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    build = int(match.group(1))
    return build if build > 0 else None


def _llama_release_build(tag: str):
    """Return the numeric build encoded by a llama.cpp release tag."""

    import re

    match = re.fullmatch(r"b?(\d+)", str(tag or "").strip(), re.IGNORECASE)
    if not match:
        return None
    build = int(match.group(1))
    return build if build > 0 else None


def _llama_runtime_receipt_path(bin_dir: str) -> str:
    return os.path.join(bin_dir, _LLAMA_RUNTIME_RECEIPT)


def _read_llama_runtime_receipt(bin_dir: str) -> dict:
    import json

    try:
        with open(_llama_runtime_receipt_path(bin_dir), "r", encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return receipt if isinstance(receipt, dict) else {}


def _write_llama_runtime_receipt(bin_dir: str, *, tag: str, build) -> None:
    """Atomically record which release supplied the installed executable."""

    import json

    path = _llama_runtime_receipt_path(bin_dir)
    temporary = f"{path}.{os.getpid()}.tmp"
    receipt = {
        "schema_version": 1,
        "release_tag": str(tag or ""),
        "build": int(build) if build else None,
        "installed_at": int(time.time()),
    }
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if os.path.isfile(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


def _llama_server_build(exe_path: str):
    """Return the installed llama-server's llama.cpp build number, or None if
    it can't be determined (e.g. unexpected --version format)."""
    try:
        import subprocess
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        out = subprocess.run(
            [exe_path, "--version"], capture_output=True, text=True, timeout=20, **kwargs
        )
        return _positive_llama_build((out.stdout or "") + (out.stderr or ""))
    except Exception:
        pass
    return None


def _ensure_llama_server(bin_dir: str) -> None:
    """Auto-download llama-server from llama.cpp GitHub releases if missing.

    Picks the appropriate prebuilt binary for the current platform:
        - Windows + CUDA (default for Maestro on NVIDIA):
            llama-b<tag>-bin-win-cuda-12.4-x64.zip
            cudart-llama-bin-win-cuda-12.4-x64.zip
        - Linux:
            llama-b<tag>-bin-ubuntu-x64.tar.gz

    llama.cpp's GitHub "latest release" is not guaranteed to be a binary
    bNNNNN release. It may point to a semantic/meta release such as v0.3.0
    which does not contain the Windows/Linux binaries.

    Therefore this function scans recent GitHub releases and selects the
    newest bNNNNN release that actually contains all assets required by the
    current platform.

    Falls back to a pinned known-good binary release if GitHub's API is
    unavailable.

    Side effect:
        Writes binaries/runtime libraries to bin_dir.

    Idempotent:
        Returns immediately if llama-server and the required CUDA runtime
        files are already present and sufficiently recent.
    """
    import os
    import sys
    import json
    import re
    import zipfile
    import tarfile
    import shutil

    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    is_windows = sys.platform.startswith("win")
    is_linux = sys.platform.startswith("linux")

    exe_name = "llama-server.exe" if is_windows else "llama-server"
    exe_path = os.path.join(bin_dir, exe_name)

    exe_exists = os.path.isfile(exe_path)

    reported_build = (
        _llama_server_build(exe_path)
        if exe_exists
        else None
    )

    receipt = _read_llama_runtime_receipt(bin_dir)

    receipt_build = _llama_release_build(
        receipt.get("release_tag", "")
    )

    if receipt_build is None:
        try:
            stored_build = int(receipt.get("build") or 0)
        except (TypeError, ValueError):
            stored_build = 0

        receipt_build = stored_build if stored_build > 0 else None

    known_build = reported_build or receipt_build

    needs_executable = not exe_exists

    if (
        exe_exists
        and known_build is not None
        and known_build < MIN_LLAMA_BUILD
    ):
        needs_executable = True

        print(
            f"[LLM] llama-server build {known_build} "
            f"< required {MIN_LLAMA_BUILD}; "
            "upgrading to the latest compatible llama.cpp release."
        )

    # ------------------------------------------------------------------
    # Windows CUDA runtime validation
    # ------------------------------------------------------------------

    missing_cuda_files = []

    if is_windows:
        missing_cuda_files = [
            filename
            for filename in _WINDOWS_LLAMA_CUDA_FILES
            if not os.path.isfile(
                os.path.join(bin_dir, filename)
            )
        ]

    needs_cudart = bool(missing_cuda_files)

    # Unknown/zero embedded version metadata is deliberately accepted.
    #
    # Official llama.cpp archives have occasionally shipped binaries whose
    # embedded version cannot be determined. Reinstalling the same binary
    # repeatedly won't improve that metadata.
    if not needs_executable and not needs_cudart:
        return

    # ------------------------------------------------------------------
    # Platform validation
    # ------------------------------------------------------------------

    if not (is_windows or is_linux):
        raise RuntimeError(
            "Auto-download of llama-server is supported on Windows "
            f"and Linux only. Detected platform: {sys.platform}. "
            "Download manually from "
            "https://github.com/ggml-org/llama.cpp/releases "
            f"and place {exe_name} in {bin_dir}."
        )

    os.makedirs(bin_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Required release assets
    # ------------------------------------------------------------------
    #
    # Windows needs TWO archives:
    #
    #   1. llama-bNNNNN-bin-win-cuda-12.4-x64.zip
    #      Contains llama-server.exe and related binaries.
    #
    #   2. cudart-llama-bin-win-cuda-12.4-x64.zip
    #      Contains CUDA runtime DLLs such as cudart/cublas.
    #
    # Putting the CUDA runtime DLLs next to llama-server.exe avoids
    # depending on a system-wide CUDA installation or PATH configuration.
    #
    # Linux uses one archive:
    #
    #   llama-bNNNNN-bin-ubuntu-x64.tar.gz
    #
    # Each tuple is:
    #
    #   (required filename prefix, required filename substring)
    #
    # Prefix matching prevents the llama archive and cudart archive from
    # being confused with one another.

    if is_windows:
        asset_specs = []

        if needs_executable:
            asset_specs.append(
                ("llama-", "bin-win-cuda-12.4-x64.zip")
            )

        if needs_cudart:
            asset_specs.append(
                ("cudart-", "bin-win-cuda-12.4-x64.zip")
            )

        archive_ext = ".zip"

    else:
        asset_specs = [
            ("llama-", "bin-ubuntu-x64.tar.gz")
        ]

        archive_ext = ".tar.gz"

    # ------------------------------------------------------------------
    # Release resolution
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Do NOT use:
    #
    #   /repos/ggml-org/llama.cpp/releases/latest
    #
    # llama.cpp may use that endpoint for a semantic/meta release which
    # does not contain platform binaries.
    #
    # Instead, inspect recent releases and find the newest bNNNNN release
    # that actually contains every asset we require.

    FALLBACK_TAG = "b10598"

    if needs_executable and not exe_exists:
        print(
            "[LLM] llama-server not found; "
            "resolving a compatible llama.cpp binary release..."
        )

    elif needs_cudart and not needs_executable:
        print(
            "[LLM] llama.cpp CUDA runtime is incomplete "
            f"(missing {', '.join(missing_cuda_files)}); "
            "repairing it..."
        )

    else:
        print(
            "[LLM] Resolving the latest compatible "
            "llama.cpp binary release..."
        )

    release_info = None
    tag = None

    api_url = (
        "https://api.github.com/repos/"
        "ggml-org/llama.cpp/releases?per_page=100"
    )

    try:
        req = Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Maestro-llama-runtime-installer",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        with urlopen(req, timeout=20) as response:
            releases = json.load(response)

        if not isinstance(releases, list):
            raise RuntimeError(
                "GitHub releases API returned unexpected data."
            )

        # The API normally returns newest releases first, but sort the binary
        # releases explicitly by bNNNNN number so we're not dependent on
        # GitHub's ordering semantics.
        binary_releases = []

        for candidate in releases:
            if candidate.get("draft"):
                continue

            candidate_tag = str(
                candidate.get("tag_name") or ""
            )

            match = re.fullmatch(
                r"b(\d+)",
                candidate_tag
            )

            if not match:
                continue

            binary_releases.append(
                (
                    int(match.group(1)),
                    candidate,
                )
            )

        binary_releases.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, candidate in binary_releases:
            candidate_tag = candidate.get(
                "tag_name",
                "",
            )

            assets = candidate.get("assets") or []

            has_everything = True

            for prefix, contains in asset_specs:
                matching_asset = next(
                    (
                        asset
                        for asset in assets
                        if str(
                            asset.get("name") or ""
                        ).startswith(prefix)
                        and contains
                        in str(
                            asset.get("name") or ""
                        )
                        and asset.get(
                            "browser_download_url"
                        )
                    ),
                    None,
                )

                if matching_asset is None:
                    has_everything = False
                    break

            if has_everything:
                release_info = candidate
                tag = candidate_tag
                break

        if release_info is None:
            raise RuntimeError(
                "No recent llama.cpp bNNNNN release contained "
                "all required platform assets."
            )

        print(
            f"[LLM] Using llama.cpp release {tag}"
        )

    except (
        URLError,
        HTTPError,
        json.JSONDecodeError,
        TimeoutError,
        RuntimeError,
        OSError,
    ) as e:
        print(
            "[LLM] Could not resolve the current llama.cpp "
            f"binary release ({e}); "
            f"falling back to pinned tag {FALLBACK_TAG}."
        )

        release_info = None
        tag = FALLBACK_TAG

    # ------------------------------------------------------------------
    # Resolve exact asset URLs
    # ------------------------------------------------------------------

    asset_urls = []

    api_assets = (
        (release_info or {}).get("assets")
        or []
    )

    for prefix, contains in asset_specs:
        asset_url = None
        asset_name = None

        # Prefer the exact URL supplied by GitHub.
        for asset in api_assets:
            name = str(
                asset.get("name") or ""
            )

            if (
                name.startswith(prefix)
                and contains in name
            ):
                candidate_url = asset.get(
                    "browser_download_url"
                )

                if candidate_url:
                    asset_url = candidate_url
                    asset_name = name
                    break

        # If API resolution wasn't available, construct the conventional
        # release URL using the pinned fallback tag.
        if not asset_url:
            if prefix == "llama-":
                asset_name = (
                    f"llama-{tag}-{contains}"
                )

            elif prefix == "cudart-":
                # The cudart archive does NOT contain the bNNNNN tag.
                asset_name = (
                    f"cudart-llama-{contains}"
                )

            else:
                raise RuntimeError(
                    f"Unknown llama.cpp asset prefix: {prefix}"
                )

            asset_url = (
                "https://github.com/"
                "ggml-org/llama.cpp/releases/download/"
                f"{tag}/{asset_name}"
            )

        print(
            f"[LLM] Resolved asset: {asset_name}"
        )

        asset_urls.append(
            (asset_name, asset_url)
        )

    # ------------------------------------------------------------------
    # Download and extract
    # ------------------------------------------------------------------

    for asset_index, (
        asset_name,
        asset_url,
    ) in enumerate(asset_urls):
        print(
            f"[LLM] Downloading {asset_name} "
            "(one-time setup)..."
        )

        # Unique name prevents weirdness if an interrupted run leaves
        # something behind.
        archive_path = os.path.join(
            bin_dir,
            f"_llama_download_{asset_index}{archive_ext}",
        )

        try:
            download_req = Request(
                asset_url,
                headers={
                    "User-Agent": (
                        "Maestro-llama-runtime-installer"
                    ),
                    "Accept": "application/octet-stream",
                },
            )

            with (
                urlopen(
                    download_req,
                    timeout=600,
                ) as response,
                open(
                    archive_path,
                    "wb",
                ) as output_file,
            ):
                content_length = response.headers.get(
                    "Content-Length"
                )

                try:
                    total_bytes = int(
                        content_length or 0
                    )
                except (TypeError, ValueError):
                    total_bytes = 0

                downloaded = 0

                chunk_size = (
                    1024 * 1024
                )  # 1 MiB

                last_pct = -10

                while True:
                    chunk = response.read(
                        chunk_size
                    )

                    if not chunk:
                        break

                    output_file.write(chunk)

                    downloaded += len(chunk)

                    if total_bytes:
                        pct = (
                            downloaded * 100
                        ) // total_bytes

                        if (
                            pct - last_pct >= 10
                            or downloaded
                            == total_bytes
                        ):
                            print(
                                "[LLM]   "
                                f"{pct}% "
                                f"({downloaded // (1024 * 1024)} / "
                                f"{total_bytes // (1024 * 1024)} MB)"
                            )

                            last_pct = pct

            # Sanity check: an HTML error page or truncated download should
            # not proceed silently into extraction.
            if not os.path.isfile(archive_path):
                raise RuntimeError(
                    "Download completed but archive file "
                    f"does not exist: {archive_path}"
                )

            archive_size = os.path.getsize(
                archive_path
            )

            if archive_size == 0:
                raise RuntimeError(
                    f"Downloaded empty archive: {asset_name}"
                )

            print(
                f"[LLM] Downloaded "
                f"{archive_size // (1024 * 1024)} MB"
            )

            print(
                f"[LLM] Extracting {asset_name}..."
            )

            # ----------------------------------------------------------
            # Windows ZIP extraction
            # ----------------------------------------------------------
            #
            # llama.cpp archives may contain binaries underneath directories
            # such as build/bin/. Maestro wants everything directly beside
            # llama-server, so flatten each archive member to its basename.

            if archive_ext == ".zip":
                if not zipfile.is_zipfile(
                    archive_path
                ):
                    raise RuntimeError(
                        "Downloaded llama.cpp asset is not "
                        f"a valid ZIP file: {asset_name}"
                    )

                with zipfile.ZipFile(
                    archive_path
                ) as archive:
                    for member in archive.infolist():
                        if member.is_dir():
                            continue

                        flat_name = os.path.basename(
                            member.filename
                        )

                        if not flat_name:
                            continue

                        target = os.path.join(
                            bin_dir,
                            flat_name,
                        )

                        with (
                            archive.open(
                                member
                            ) as src,
                            open(
                                target,
                                "wb",
                            ) as dst,
                        ):
                            shutil.copyfileobj(
                                src,
                                dst,
                            )

            # ----------------------------------------------------------
            # Linux tar.gz extraction
            # ----------------------------------------------------------

            else:
                with tarfile.open(
                    archive_path,
                    "r:gz",
                ) as archive:
                    for member in archive.getmembers():
                        if not member.isfile():
                            continue

                        flat_name = os.path.basename(
                            member.name
                        )

                        if not flat_name:
                            continue

                        target = os.path.join(
                            bin_dir,
                            flat_name,
                        )

                        src = archive.extractfile(
                            member
                        )

                        if src is None:
                            continue

                        with (
                            src,
                            open(
                                target,
                                "wb",
                            ) as dst,
                        ):
                            shutil.copyfileobj(
                                src,
                                dst,
                            )

                        # Preserve executable bit on Linux.
                        try:
                            os.chmod(
                                target,
                                member.mode,
                            )
                        except OSError:
                            pass

        except HTTPError as e:
            raise RuntimeError(
                "Failed to download llama.cpp asset "
                f"{asset_name}: HTTP {e.code} "
                f"from {asset_url}"
            ) from e

        except URLError as e:
            raise RuntimeError(
                "Failed to download llama.cpp asset "
                f"{asset_name}: {e.reason}. "
                f"URL: {asset_url}"
            ) from e

        except (
            zipfile.BadZipFile,
            tarfile.TarError,
        ) as e:
            raise RuntimeError(
                "Failed to extract llama.cpp asset "
                f"{asset_name}: {e}"
            ) from e

        finally:
            try:
                os.remove(
                    archive_path
                )
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Validate installation
    # ------------------------------------------------------------------

    if not os.path.isfile(exe_path):
        attempted_urls = [
            url
            for _, url in asset_urls
        ]

        raise FileNotFoundError(
            "Downloaded and extracted llama.cpp release "
            f"{tag}, but {exe_name} was not found in "
            f"{bin_dir}. Tried: {attempted_urls}"
        )

    # On Windows, make sure the CUDA runtime repair actually succeeded.
    if is_windows:
        still_missing_cuda_files = [
            filename
            for filename in _WINDOWS_LLAMA_CUDA_FILES
            if not os.path.isfile(
                os.path.join(
                    bin_dir,
                    filename,
                )
            )
        ]

        if still_missing_cuda_files:
            raise FileNotFoundError(
                "llama.cpp was downloaded, but required "
                "CUDA runtime files are still missing: "
                + ", ".join(
                    still_missing_cuda_files
                )
            )

    # ------------------------------------------------------------------
    # Store release/build receipt
    # ------------------------------------------------------------------

    if needs_executable:
        installed_build = (
            _llama_server_build(
                exe_path
            )
            or _llama_release_build(
                tag
            )
        )

        _write_llama_runtime_receipt(
            bin_dir,
            tag=tag,
            build=installed_build,
        )

    print(
        f"[LLM] llama-server installed to {exe_path}"
    )



def _get_server_exe() -> str:
    """Find the llama-server executable, downloading it on first use if missing.

    Lazy-download pattern matches the model-weights flow: nothing is
    fetched until the user actually triggers their first LLM call. The
    download is one-time (~50-100 MB) and cached in bin_dir, so
    subsequent loads are instant.
    """
    bin_dir = os.environ.get("MAESTRO_LLAMA_BIN", DEFAULT_BIN_DIR)
    _ensure_llama_server(bin_dir)
    if os.name == "nt":
        return os.path.join(bin_dir, "llama-server.exe")
    return os.path.join(bin_dir, "llama-server")


def load_model(
    model_id: str = "",
    device: str = "cpu",
    force_reload: bool = False,
    provider: str = "local",
    remote_url: str = "",
    api_key: str = "",
) -> None:
    """Load an LLM model. Supports local (llama-server), remote (OpenAI-compatible),
    OpenAI API, and Anthropic API providers.

    Args:
        model_id: Model ID (HF repo for local, model name for remote/API)
        device: "cpu" or "cuda" (local only)
        force_reload: If True, restart even if already running
        provider: "local" | "remote" | "openai" | "anthropic"
        remote_url: Base URL for remote/openai servers (e.g. http://192.168.1.100:1234)
        api_key: API key for openai/anthropic providers
    """
    global _process, _model_id, _device, _server_port, _vision_available
    global _provider, _remote_url, _api_key

    # Handle remote/API providers — no subprocess needed
    if provider in ("remote", "openai", "anthropic"):
        with _lock:
            if (
                is_loaded()
                and _model_id == model_id
                and _provider == provider
                and _remote_url == remote_url
                and _api_key == api_key
                and not force_reload
            ):
                return
            if _process is not None:
                _unload_inner()
            _provider = provider
            _remote_url = remote_url
            _api_key = api_key
            _model_id = model_id
            _device = provider
            _vision_available = False
            print(f"[LLM] Connected to {provider} provider: model={model_id}, url={remote_url or 'API'}")
            _reset_idle_timer()
        return

    repo_id = model_id or DEFAULT_HF_REPO
    _provider = "local"
    _remote_url = ""
    _api_key = ""

    with _lock:
        if is_loaded() and _model_id == repo_id and not force_reload:
            return

        if is_loaded():
            _unload_inner()

        print(f"[LLM] Loading model: {repo_id} on {device}")

        base_cache_dir = get_model_dir()

        # Look up GGUF filename from registry, fall back to convention
        repo_basename = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        model_stem = repo_basename.replace("-GGUF", "")
        if repo_id in MODEL_REGISTRY:
            gguf_file = MODEL_REGISTRY[repo_id]["gguf_file"]
        else:
            gguf_file = f"{model_stem}-Q4_K_S.gguf"

        # Use model-specific subdirectory to avoid cache collisions between models
        dir_override = MODEL_REGISTRY.get(repo_id, {}).get("cache_dir_override")
        cache_dir = os.path.join(base_cache_dir, dir_override or model_stem)
        os.makedirs(cache_dir, exist_ok=True)

        gguf_path = _download_gguf(repo_id, gguf_file, cache_dir)
        gguf_path = os.path.normpath(gguf_path)

        # Try to download mmproj for vision support (optional — not all models have it)
        registry_entry = MODEL_REGISTRY.get(repo_id, {})
        mmproj_file = registry_entry.get("mmproj_file", DEFAULT_MMPROJ_FILE)
        mmproj_repo = registry_entry.get("mmproj_repo", repo_id)  # allow mmproj from different repo
        mmproj_path = None
        try:
            mmproj_path = _download_gguf(mmproj_repo, mmproj_file, cache_dir)
            mmproj_path = os.path.normpath(mmproj_path)
            print(f"[LLM] Vision support: mmproj loaded from {mmproj_repo}")
        except Exception as e:
            print(f"[LLM] No mmproj available (vision disabled): {e}")

        _vision_available = mmproj_path is not None or registry_entry.get("native_vision", False)

        server_exe = _get_server_exe()

        _server_port = _find_free_port()

        extra_flags = registry_entry.get("extra_flags", [])

        # Most models ship a sane jinja chat template inside the GGUF and
        # benefit from --jinja so llama-server honors thinking-mode kwargs
        # like enable_thinking. A small number of fine-tunes ship a
        # "neutral"/generic embedded template that breaks the model's
        # expected role markers — those entries can opt out by setting
        # `disable_jinja: True` in the registry, typically alongside a
        # `--chat-template <name>` override in extra_flags to force the
        # canonical built-in template (e.g. "gemma" for Gemma 4 fine-tunes
        # with broken embedded templates).
        cmd = [
            server_exe,
            "--model", gguf_path,
            "--host", "127.0.0.1",
            "--port", str(_server_port),
            "--threads", str(max(os.cpu_count() // 2, 2)),
        ]
        if not registry_entry.get("disable_jinja", False):
            cmd.append("--jinja")

        # Use model-specific context size if provided, otherwise default
        if "-c" in extra_flags:
            cmd += extra_flags
        else:
            cmd += ["--ctx-size", "65536"] + extra_flags

        if mmproj_path:
            cmd += ["--mmproj", mmproj_path]
            # Force ONE image per encode batch. llama-server's
            # clip_image_batch_encode sizes its output buffer for a single
            # image, but the mtmd batcher groups same-processed-shape images
            # from one request into one batch — two identically-sized images
            # (e.g. Director's start-frame references, both 432x768) then
            # abort the server ("Output buffer size mismatch", build 9632)
            # and the client sees a bare connection reset. A cap of 1 token
            # per batch means every image always exceeds it and is encoded
            # alone (the batcher always admits at least one image).
            # Verified against the exact crashing request.
            if "--mtmd-batch-max-tokens" not in extra_flags:
                cmd += ["--mtmd-batch-max-tokens", "1"]

        if device == "cuda":
            # Use -ngl from extra_flags if present, otherwise default to all layers
            if "-ngl" in extra_flags:
                pass  # already included via extra_flags
            else:
                cmd += ["--n-gpu-layers", "-1"]
        else:
            cmd += ["--n-gpu-layers", "0"]

        print(f"[LLM] Starting llama-server on port {_server_port}")
        _process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        # Wait for server to be ready (poll /health)
        # Scale timeout with model size — large models need more time to load
        file_size_gb = os.path.getsize(gguf_path) / 1e9
        load_timeout = max(60, int(file_size_gb * 15))  # ~15s per GB, min 60s
        ready = False
        for i in range(load_timeout):
            if _process.poll() is not None:
                # Process exited — read output for error details
                exit_code = _process.returncode
                output = _process.stdout.read().decode(errors="replace") if _process.stdout else ""
                _process = None
                _model_id = ""
                raise RuntimeError(f"llama-server exited with code {exit_code}:\n{output[-2000:]}")
            try:
                resp = requests.get(f"{_server_url()}/health", timeout=1)
                if resp.status_code == 200:
                    data = resp.json() if resp.text else {}
                    status = data.get("status", "ok")
                    if status == "ok":
                        ready = True
                        break
                    elif status == "loading model":
                        pass  # still loading
            except (requests.ConnectionError, requests.Timeout):
                pass
            time.sleep(1)

        if not ready:
            # Capture process output for debugging
            server_output = ""
            if _process and _process.stdout:
                import select
                try:
                    # Non-blocking read of whatever output is available
                    _process.stdout.flush()
                    import threading
                    lines = []
                    def _read():
                        try:
                            for line in iter(_process.stdout.readline, b''):
                                lines.append(line.decode(errors="replace"))
                                if len(lines) > 50:
                                    break
                        except Exception:
                            pass
                    t = threading.Thread(target=_read, daemon=True)
                    t.start()
                    t.join(timeout=2)
                    server_output = "".join(lines[-20:])
                except Exception:
                    pass
            _unload_inner()
            raise RuntimeError(
                f"llama-server did not become ready within {load_timeout}s (model: {file_size_gb:.1f}GB)\n"
                f"Server output:\n{server_output}"
            )

        _model_id = repo_id
        _device = device
        file_size = os.path.getsize(gguf_path) / 1e6
        print(f"[LLM] Model loaded: {repo_id} ({file_size:.0f}MB) on {device}, port {_server_port}")

        # Start draining the server's output now that it's up. Without this
        # the PIPE fills on a long run and the server blocks on write.
        _start_log_reader(_process)


def _cancel_idle_timer():
    """Cancel any pending idle-unload timer."""
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def _reset_idle_timer():
    """Reset the idle-unload timer. Called after each LLM request."""
    global _idle_timer
    _cancel_idle_timer()
    _idle_timer = threading.Timer(_idle_timeout, _auto_unload)
    _idle_timer.daemon = True
    _idle_timer.start()


def _auto_unload():
    """Called by the idle timer to unload the LLM after inactivity."""
    global _idle_timer
    _idle_timer = None
    if is_loaded():
        print("[LLM] Auto-unloading after idle timeout")
        unload_model()


def _start_log_reader(proc: subprocess.Popen) -> None:
    """Drain llama-server's stdout into `_server_log` in the background.

    Prevents the OS pipe from filling (which deadlocks the server) and
    keeps a rolling tail for crash diagnosis. The thread ends on its own
    when the pipe closes (i.e. the process exits).

    Also mirrors every line to logs/llm/llama-server.log (fresh file per
    server launch) — the in-memory tail dies with the process, and a
    server crash mid-request is exactly the moment a postmortem needs
    the full output."""
    global _log_reader
    _server_log.clear()
    log_path = None
    try:
        log_dir = os.path.join(_BASE_DIR, "..", "..", "logs", "llm")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "llama-server.log")
    except Exception:
        pass

    def _drain():
        log_file = None
        gemma_compat_noted = False
        if log_path:
            try:
                log_file = open(log_path, "w", encoding="utf-8", errors="replace")
            except Exception:
                log_file = None
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode(errors="replace").rstrip("\n")
                if _is_benign_gemma_template_warning(line):
                    # llama.cpp applies its compatibility workaround and keeps
                    # serving. Recording the raw line as a WARNING made users
                    # and crash-tail diagnostics mistake it for the failure.
                    if gemma_compat_noted:
                        continue
                    line = (
                        "[LLM] Gemma 4 embedded chat template is older; "
                        "llama.cpp compatibility mode is active (non-fatal)."
                    )
                    gemma_compat_noted = True
                _server_log.append(line)
                if log_file:
                    try:
                        log_file.write(line + "\n")
                        log_file.flush()
                    except Exception:
                        log_file = None
        except Exception:
            pass
        finally:
            if log_file:
                try:
                    log_file.close()
                except Exception:
                    pass

    _log_reader = threading.Thread(target=_drain, name="llama-log-reader", daemon=True)
    _log_reader.start()


def _server_log_tail(n: int = 20) -> str:
    # Be defensive with logs captured before the reader learned to normalize
    # this warning. It is not evidence of the subsequent request failure.
    lines = [
        line for line in list(_server_log)
        if not _is_benign_gemma_template_warning(line)
    ][-n:]
    return "\n".join(lines) if lines else "(no server output captured)"


def _diagnose_llm_request_failure(exc: Exception) -> "RuntimeError":
    """Turn a raw request failure into an actionable error.

    Distinguishes "the llama-server subprocess died" (the common cause of a
    frozen Director run — bad GGUF quant, VRAM OOM at load, wrong binary)
    from a transient network blip, and quotes the server's last output so
    the pipeline error the user sees names the real cause.
    """
    proc = _process
    if _provider == "local" and proc is not None:
        # A reset socket usually means the subprocess is mid-death; poll()
        # can race the actual exit by a moment. Give it a beat to finish
        # dying so a crash is reported as a crash (with the server's last
        # words) instead of a generic connection error.
        try:
            proc.wait(timeout=3)
        except Exception:
            pass
    if _provider == "local" and proc is not None and proc.poll() is not None:
        code = proc.returncode
        tail = _server_log_tail(40)
        _unload_inner()  # reset singleton so the next call relaunches cleanly
        # Only use OOM wording when the server log actually shows an OOM —
        # services/oom_detect.py substring-matches "out of memory" on error
        # text, so speculative OOM wording here made every server crash pop
        # the "lower VRAM headroom?" recovery banner even when the GPU was
        # nearly empty (e.g. the clip.cpp image-batch abort).
        tail_l = tail.lower()
        if any(s in tail_l for s in ("out of memory", "cudamalloc", "erralloc", "alloc failed")):
            cause = "The GPU ran out of memory mid-request (e.g. a video/image model was still resident)."
        else:
            cause = "This is an internal llama-server failure; see its last output below."
        return RuntimeError(
            f"The local LLM server (llama-server) crashed while generating "
            f"(exit code {code}). {cause} "
            f"Full log: logs/llm/llama-server.log. "
            f"Last server output:\n{tail}"
        )
    if _provider == "local" and proc is None:
        return RuntimeError(
            "The local LLM server is not running. It may have been unloaded "
            "or failed to start — retry, or check the Services settings."
        )
    if _provider == "local":
        # Server still alive — include its recent output anyway; CUDA errors
        # can surface as dropped requests without killing the process.
        tail = _server_log_tail(15)
        return RuntimeError(
            f"LLM request failed: {exc}\nRecent llama-server output:\n{tail}"
        )
    # Remote provider — include the endpoint's explanation. requests'
    # HTTPError string normally contains only the status line, while strict
    # gateways put the actionable "unknown field" / "model not found" detail
    # in their response body.
    detail = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = (response.text or "").strip()
        except Exception:
            body = ""
        if body:
            if len(body) > 800:
                body = body[:800] + "... (truncated)"
            detail = f"\nEndpoint response: {body}"
    return RuntimeError(f"LLM request failed: {exc}{detail}")


def _unload_inner():
    global _process, _model_id, _device, _server_port, _vision_available
    _cancel_idle_timer()
    if _process is not None:
        try:
            _process.terminate()
            _process.wait(timeout=10)
        except Exception:
            try:
                _process.kill()
            except Exception:
                pass
    _process = None
    _model_id = ""
    _device = ""
    _server_port = 0
    _vision_available = False
    gc.collect()


def unload_model() -> None:
    with _lock:
        _unload_inner()
        print("[LLM] Model unloaded")


def _image_to_data_url(image_path: str, max_size: int = 768) -> Optional[str]:
    """Read an image file, resize if needed, and return a data URL (base64-encoded).

    Large images are resized so the longest edge is at most *max_size* pixels
    and re-encoded as JPEG to keep the data URL compact for LLM context.
    """
    import base64
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(image_path)
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            # ASCII arrow on purpose: a cp1252 console (plain cmd, some CI
            # shells) can't encode U+2192 and the print would crash the
            # whole vision request mid-flight.
            print(f"[LLM] Resized image for LLM: {w}x{h} -> {img.size[0]}x{img.size[1]}")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    except ImportError:
        # PIL not available — send raw file (no resize)
        import mimetypes
        mime, _ = mimetypes.guess_type(image_path)
        if not mime:
            mime = "image/png"
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{data}"


def generate(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    image_paths: Optional[list] = None,
    thinking_budget: int = 0,
    enable_thinking: Optional[bool] = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[list[str]] = None,
    json_schema: Optional[dict] = None,
) -> str:
    """Generate text via llama-server's OpenAI-compatible chat endpoint.

    Args:
        image_paths: Optional list of file paths to images. If provided and
            the model supports vision, images are sent as multimodal content.
        thinking_budget: Extra tokens reserved for model reasoning/thinking.
            Added on top of max_new_tokens so thinking doesn't eat into
            the content budget.
        enable_thinking: If False, disables Qwen3.5's thinking mode via the
            --jinja chat template. If None, uses model default (thinking on).
        json_schema: Optional JSON Schema dict. When set (local llama-server
            only), the output is grammar-constrained to schema-valid JSON —
            the sampler masks every token that would break the schema, so
            the model physically cannot emit prose, markdown fences, or the
            repeat-loop garbage that breaks structured planning passes.
    """
    if not is_loaded():
        raise RuntimeError("LLM not loaded. Call load_model() first.")

    # Cancel idle timer during active request — prevents auto-unload mid-generation.
    # Timer is reset at the END of the request (after response is received).
    _cancel_idle_timer()

    # Grammar-constrained JSON mode requires thinking OFF. The grammar
    # constrains sampling from the FIRST token, so any thinking the chat
    # template force-opens (Gemma's `<|think|>`, Qwen's `<think>`) would
    # trap the model: it could only emit schema-JSON, never the think-close
    # marker, and the parser would file the entire output under
    # reasoning_content with empty content. Forcing enable_thinking=False
    # here makes _prepare_thinking skip every activation path.
    if json_schema is not None:
        enable_thinking = False
        thinking_budget = 0

    # Per-model thinking mode (Gemma vs Qwen)
    system_prompt, enable_thinking, thinking_budget = _prepare_thinking(system_prompt, enable_thinking, thinking_budget)

    total_tokens = max_new_tokens + thinking_budget

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message — multimodal if images provided and vision is available
    if image_paths and _vision_available:
        content_parts = []
        for img_path in image_paths:
            data_url = _image_to_data_url(img_path)
            if data_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
        content_parts.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_tokens": total_tokens,
        "cache_prompt": False,  # Disable prompt caching — system prompt changes between calls (LoRA hints, etc.)
    }
    # Per-model sampling defaults (e.g. Gemma 4 wants temp=1.0, top_k=64)
    temperature, top_p = _apply_model_defaults(temperature, top_p, payload)
    payload["temperature"] = max(temperature, 0.01)
    payload["top_p"] = top_p

    if frequency_penalty > 0:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty > 0:
        payload["presence_penalty"] = presence_penalty
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    # Qwen thinking mode via chat template kwargs (Gemma handled by _prepare_thinking)
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    # Hard stop sequences. The Director Pass 3 polish path uses this with
    # `<think>` to abort generation the moment a Qwen3.5/3.6 model tries
    # to enter thinking mode despite enable_thinking=False being requested.
    # That cap caps wasted tokens at ~1 (the `<think>` token itself) instead
    # of the previous ~1024 the model would burn before producing nothing.
    #
    # For registry entries with `disable_thinking: True`, automatically
    # inject thinking-marker stop tokens so every call gets the same
    # protection — covers Gemma 4 fine-tunes that auto-activate thinking
    # mode despite chat_template_kwargs saying otherwise. Both Qwen-style
    # (`<think>`) and Gemma-style (`<channel>`, `<|think|>`) markers are
    # included since the supergemma fine-tune emits the latter format.
    combined_stop = list(stop) if stop else []
    if _active_registry_entry().get("disable_thinking", False):
        for tok in ("<think>", "<thinking>", "<|think|>", "<channel>", "<|channel|>"):
            if tok not in combined_stop:
                combined_stop.append(tok)
    if combined_stop:
        payload["stop"] = combined_stop

    # Grammar-constrained JSON output. llama-server compiles the schema to
    # a GBNF grammar server-side ({"type": "json_object", "schema": ...} is
    # the long-standing llama.cpp extension form). Local provider only —
    # remote OpenAI-compatible endpoints vary in which response_format
    # flavor they accept, so we degrade to an unconstrained call there.
    if json_schema is not None:
        if _provider == "local":
            payload["response_format"] = {"type": "json_object", "schema": json_schema}
        else:
            print(f"[LLM] json_schema requested but provider={_provider} — sending unconstrained (grammar is local llama-server only)")

    if _provider == "anthropic":
        return _generate_anthropic(messages, total_tokens, max(temperature, 0.01), top_p)

    try:
        resp = requests.post(
            f"{_server_url()}/v1/chat/completions",
            json=_finalize_payload(payload),
            headers=_api_headers(),
            # (connect, read): fail fast if the server socket is gone;
            # allow a long read for actual generation.
            timeout=(10, 600),
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        # A dead subprocess surfaces here as a ConnectionError; translate it
        # into an actionable error naming the real cause (see the helper).
        raise _diagnose_llm_request_failure(e) from e
    data = resp.json()

    raw_content = data["choices"][0]["message"]["content"] or ""
    finish_reason = data["choices"][0].get("finish_reason", "unknown")
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", "?")
    completion_tokens = usage.get("completion_tokens", "?")
    print(f"[LLM] Response: {completion_tokens} tokens generated (prompt={prompt_tokens}, finish={finish_reason})")
    if not raw_content:
        print(f"[LLM] WARNING: Server returned empty content despite generating {completion_tokens} tokens (model likely consumed all tokens on internal reasoning)")
        # Check if reasoning_content is available (llama-server may separate it)
        reasoning = data["choices"][0]["message"].get("reasoning_content", "")
        if reasoning:
            print(f"[LLM] Reasoning content detected ({len(reasoning)} chars) — model used thinking mode. reasoning_budget=0 may not be active.")

    content = _strip_thinking_tags(raw_content)

    if not content.strip() and raw_content:
        print(f"[LLM] WARNING: Model spent all {completion_tokens} tokens on <think> reasoning with nothing left for the answer. Raw starts with: {raw_content[:200]!r}")

    _reset_idle_timer()
    return content.strip()


def get_stream_status() -> dict:
    """Return current streaming state for polling."""
    with _stream_lock:
        return {"text": _stream_buffer, "done": _stream_done}


def generate_streaming(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = -1,
    image_paths: list = None,
    thinking_budget: int = 0,
    enable_thinking: bool = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    json_schema: Optional[dict] = None,
) -> str:
    """Generate text using SSE streaming, populating the stream buffer in real-time.

    Same interface as generate(), but tokens appear in _stream_buffer as they arrive.
    Returns the final stripped content (same as generate()).

    Args:
        thinking_budget: Extra tokens reserved for model reasoning/thinking.
            Added on top of max_new_tokens so thinking doesn't eat into
            the content budget. Set to 0 to use max_new_tokens as-is.
        json_schema: Optional JSON Schema dict — grammar-constrains the
            output to schema-valid JSON on local llama-server. Forces
            thinking OFF (see generate() for the rationale).
    """
    global _stream_buffer, _stream_done, _last_system_prompt, _last_user_prompt, _last_thinking_text
    import re as _re

    if not is_loaded():
        raise RuntimeError("LLM not loaded. Call load_model() first.")

    # Grammar-constrained JSON mode requires thinking OFF — same rationale
    # as the matching block in generate(): the grammar masks sampling from
    # the first token, so a force-opened think block could never close.
    if json_schema is not None:
        enable_thinking = False
        thinking_budget = 0

    # Per-model thinking mode (Gemma vs Qwen)
    system_prompt, enable_thinking, thinking_budget = _prepare_thinking(system_prompt, enable_thinking, thinking_budget)

    # Store for pipeline dashboard capture (system + user prompt both,
    # so the dashboard can render the full LLM input).
    _last_system_prompt = system_prompt
    _last_user_prompt = prompt
    _last_thinking_text = ""

    # Cancel idle timer during active request — prevents auto-unload mid-streaming.
    # Timer is reset at the END of the request (after streaming completes).
    _cancel_idle_timer()

    total_tokens = max_new_tokens + thinking_budget

    with _stream_lock:
        _stream_buffer = ""
        _stream_done = False

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build user message — multimodal if images provided and vision is available
    if image_paths and _vision_available:
        content_parts = []
        for img_path in image_paths:
            data_url = _image_to_data_url(img_path)
            if data_url:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url},
                })
        content_parts.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "max_tokens": total_tokens,
        "stream": True,
        "cache_prompt": False,  # Disable prompt caching — system prompt changes between calls
    }
    # Apply caller's penalty values FIRST so they're in the payload
    # before _apply_model_defaults runs. The registry-defaults pass below
    # then overrides them when the active model has tuned values
    # (Gemma 4 → no penalty; Qwen 3.x → penalty stays as caller suggested).
    payload["temperature"] = max(temperature, 0.01)
    payload["top_p"] = top_p
    if frequency_penalty > 0:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty > 0:
        payload["presence_penalty"] = presence_penalty
    # Per-model sampling defaults — registry wins over caller for any
    # field it specifies. Models without sampling_defaults (e.g. Qwen
    # 3.x) pass through unchanged. See _apply_model_defaults().
    temperature, top_p = _apply_model_defaults(temperature, top_p, payload)
    payload["temperature"] = max(temperature, 0.01)
    payload["top_p"] = top_p
    if seed is not None and seed >= 0:
        payload["seed"] = seed
    # Qwen thinking mode via chat template kwargs (Gemma handled by _prepare_thinking)
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    # Auto-inject thinking-marker stop tokens for `disable_thinking: True`
    # models. Mirrors the same protection in generate() — see comment there.
    # Critical for Gemma 4 fine-tunes whose embedded chat templates ignore
    # the enable_thinking=false kwarg and auto-enter reasoning mode anyway,
    # burning the entire token budget on `reasoning_content` and returning
    # empty `content`. Stopping on the marker token caps wasted tokens at 1.
    if _active_registry_entry().get("disable_thinking", False):
        stop_tokens = ["<think>", "<thinking>", "<|think|>", "<channel>", "<|channel|>"]
        existing = payload.get("stop") or []
        payload["stop"] = list(existing) + [t for t in stop_tokens if t not in existing]

    # Grammar-constrained JSON output — local llama-server only (see the
    # matching block in generate() for the full rationale).
    if json_schema is not None:
        if _provider == "local":
            payload["response_format"] = {"type": "json_object", "schema": json_schema}
        else:
            print(f"[LLM] json_schema requested but provider={_provider} — sending unconstrained (grammar is local llama-server only)")

    # Diagnostic — log every payload field except `messages` so we can
    # compare what Maestro sends to llama-server vs what LM Studio sends
    # for the same model. The messages array gets summarized (length per
    # role) instead of dumped, since system prompts can be multi-KB and
    # multimodal content includes base64-encoded images.
    try:
        _diag = {k: v for k, v in payload.items() if k != "messages"}
        # The compiled schema can be multi-KB — log its size, not its body.
        if "response_format" in _diag:
            _diag["response_format"] = f"<json grammar, {len(str(_diag['response_format']))} chars>"
        _msg_summary = []
        for _m in messages:
            _role = _m.get("role", "?")
            _content = _m.get("content")
            if isinstance(_content, str):
                _msg_summary.append(f"{_role}({len(_content)}c)")
            elif isinstance(_content, list):
                _parts = []
                for _p in _content:
                    _t = _p.get("type", "?")
                    if _t == "text":
                        _parts.append(f"text({len(_p.get('text',''))}c)")
                    elif _t == "image_url":
                        _parts.append("image")
                    else:
                        _parts.append(_t)
                _msg_summary.append(f"{_role}[{','.join(_parts)}]")
            else:
                _msg_summary.append(_role)
        print(f"[LLM] Payload to llama-server: {_diag} | messages=[{', '.join(_msg_summary)}]")
    except Exception:
        pass

    if _provider == "anthropic":
        return _generate_streaming_anthropic(messages, total_tokens, max(temperature, 0.01), top_p)

    raw_content = ""
    reasoning_content = ""
    in_reasoning = False
    try:
        resp = requests.post(
            f"{_server_url()}/v1/chat/completions",
            json=_finalize_payload(payload),
            headers=_api_headers(),
            timeout=(10, 600),
            stream=True,
        )
        resp.raise_for_status()

        import json as _json_mod
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # strip "data: "
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = _json_mod.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # With --jinja, Qwen3.5 may send reasoning via separate field
                reasoning_token = delta.get("reasoning_content", "")
                if reasoning_token:
                    reasoning_content += reasoning_token
                    if not in_reasoning:
                        in_reasoning = True
                    # Show reasoning in the stream buffer wrapped in <think> tags
                    with _stream_lock:
                        _stream_buffer = f"<think>{reasoning_content}</think>"

                token = delta.get("content", "")
                if token:
                    raw_content += token
                    # Build display: thinking (if any) + content so far
                    display = ""
                    if reasoning_content:
                        display = f"<think>{reasoning_content}</think>\n"
                    display += raw_content
                    with _stream_lock:
                        _stream_buffer = display
            except Exception:
                continue

    except requests.exceptions.RequestException as e:
        # Server socket died mid-stream (common: subprocess crash). Surface
        # the real cause so the Director run reports it instead of hanging.
        with _stream_lock:
            _stream_done = True
        raise _diagnose_llm_request_failure(e) from e
    except Exception:
        with _stream_lock:
            _stream_done = True
        raise

    # Capture thinking text for the pipeline dashboard. Two sources:
    #   1. reasoning_content — populated by chat templates that emit
    #      thinking via the OpenAI-style `reasoning_content` delta field
    #      (Qwen 3.x with --jinja, base Gemma 4 if the template fires).
    #   2. Inline <|channel>thought\n...<channel|> markers in raw_content
    #      — emitted by Gemma 4 Heretic and similar fine-tunes whose
    #      chat templates don't extract thinking into reasoning_content.
    # Prefer (1) when present, fall back to (2).
    inline_gemma_match = _GEMMA_THINKING_INNER_RE.search(raw_content)
    inline_gemma_thinking = inline_gemma_match.group(1) if inline_gemma_match else ""
    _last_thinking_text = reasoning_content or inline_gemma_thinking
    print(
        f"[LLM] Streaming complete: {len(raw_content)} chars, "
        f"reasoning_content: {len(reasoning_content)} chars, "
        f"gemma_inline_thinking: {len(inline_gemma_thinking)} chars"
    )

    # Build full raw for the UI (includes thinking)
    full_raw = ""
    if reasoning_content:
        full_raw = f"<think>{reasoning_content}</think>\n"
    full_raw += raw_content

    # Strip thinking/reasoning blocks for the return value
    content = _strip_thinking_tags(full_raw)

    with _stream_lock:
        _stream_buffer = full_raw  # keep full raw for the UI to show thinking
        _stream_done = True

    _reset_idle_timer()
    return content.strip()


def _generate_anthropic(messages: list, max_tokens: int, temperature: float, top_p: float) -> str:
    """Non-streaming generation via Anthropic Messages API."""
    import re as _re
    # Anthropic uses system as a top-level param, not in messages
    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append(m)

    payload = {
        "model": _model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": api_messages,
    }
    if system_text:
        payload["system"] = system_text

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        json=payload,
        headers=_api_headers(),
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    # Anthropic response: {"content": [{"type": "text", "text": "..."}], ...}
    raw_content = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            raw_content += block.get("text", "")

    usage = data.get("usage", {})
    print(f"[LLM/Anthropic] Response: {usage.get('output_tokens', '?')} tokens (prompt={usage.get('input_tokens', '?')})")

    content = _strip_thinking_tags(raw_content)
    _reset_idle_timer()
    return content.strip()


def _generate_streaming_anthropic(messages: list, max_tokens: int, temperature: float, top_p: float) -> str:
    """Streaming generation via Anthropic Messages API with SSE."""
    global _stream_buffer, _stream_done
    import re as _re

    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append(m)

    payload = {
        "model": _model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": api_messages,
        "stream": True,
    }
    if system_text:
        payload["system"] = system_text

    raw_content = ""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=_api_headers(),
            timeout=600,
            stream=True,
        )
        resp.raise_for_status()

        import json
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8", errors="replace")
            if not line_str.startswith("data: "):
                continue
            json_str = line_str[6:]
            if json_str.strip() == "[DONE]":
                break
            try:
                event = json.loads(json_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    raw_content += text
                    with _stream_lock:
                        _stream_buffer = raw_content

    except Exception as e:
        print(f"[LLM/Anthropic] Streaming error: {e}")
        with _stream_lock:
            _stream_buffer = raw_content or f"Error: {e}"
            _stream_done = True
        return ""

    content = _strip_thinking_tags(raw_content)

    with _stream_lock:
        _stream_buffer = raw_content
        _stream_done = True

    _reset_idle_timer()
    return content.strip()


def _build_enhance_user_prompt(
    prompt,
    mode,
    duration_seconds,
    window_count,
    window_size_seconds,
    model_type="",
):
    """Prefix the user prompt with the app's structural context (duration +
    sliding-window / paragraph count) so the LLM writes one paragraph per
    window. Shared by the guide-based path and the raw per-model-enhancer
    path — the dedicated enhancer gets no system guide, so without this it has
    no idea how many window-paragraphs to produce."""
    if duration_seconds and mode in ("video", "avatar"):
        parts = [f"Duration: {duration_seconds} seconds"]
        if window_count and window_count > 1:
            parts.append(f"{window_count} sliding windows of ~{window_size_seconds}s each")
            # State the COUNT explicitly, not just the ratio — a fine-tuned
            # enhancer (e.g. Sulphur) can read "one paragraph per window" as
            # "one paragraph" and collapse a 2-window prompt into a single one.
            parts.append(
                f"Write EXACTLY {window_count} paragraphs (one per window), "
                "separated by newlines"
            )
        context = f"[{', '.join(parts)}]"
        if (
            window_count
            and window_count > 1
            and str(model_type or "").lower().startswith("ltx")
        ):
            context += (
                "\n[LTX STANDALONE-WINDOW CONTRACT: Each paragraph is sent "
                "alone to one native video pass. A later pass receives only a "
                "short audiovisual overlap, never any previous prompt text. "
                "First identify the sequence-wide capture format, camera "
                "behavior, speed/pacing, visual style or era, persistent "
                "identity/world rules, continuous-shot rules, and audio rules. "
                "Repeat every applicable sequence-wide rule explicitly and "
                "with near-identical wording in EVERY paragraph. Keep only "
                "the window-local action, environment, and physical handoff "
                "unique. If the request is endless, looping, nonstop, or "
                "never-stopping, the final paragraph must remain in motion "
                "through its last frame and must not slow, settle, stop, or "
                "resolve.]"
            )
        return f"{context}\n\n{prompt}"
    return prompt


def _clean_enhancer_output(text):
    """Strip a fine-tuned enhancer's spurious leading style-framing clause —
    e.g. "The video, rendered in a high-quality 3D animation style," — which
    the Sulphur enhancer prepends even to live-action / real-people content,
    wrongly forcing a 3D/animation look. Applied per line so it also cleans
    multi-paragraph (multi-window) output. Conservative: only fires on a
    leading "The <video/clip/...> ... rendered in ... <style keyword> ...,"."""
    if not text:
        return text
    import re
    pat = re.compile(
        r'^\s*the\s+(?:video|clip|scene|footage|animation)\b[^.]*?'
        r'\brendered\s+in\b[^.]*?'
        r'\b(?:style|animation|cgi|3d|cg|render(?:ing)?)\b[^.]*?[,.]\s+',
        re.IGNORECASE,
    )
    out = []
    for line in text.split("\n"):
        new = pat.sub("", line, count=1)
        if new != line and new:
            new = new[0].upper() + new[1:]  # re-capitalize after the strip
        out.append(new)
    return "\n".join(out).strip()


def enhance_prompt(
    prompt: str,
    mode: str = "video",
    max_new_tokens: int = 200,
    temperature: float = 0.6,
    nsfw: bool = False,
    model_type: str = "",
    image_paths: Optional[list] = None,
    duration_seconds: Optional[int] = None,
    window_count: Optional[int] = None,
    window_size_seconds: Optional[int] = None,
    system_override: Optional[str] = None,
    tts_enhance_mode: Optional[str] = None,
    tts_voice_count: int = 2,
    lora_system_hint: str = "",
    raw_enhancer_mode: bool = False,
    reference_context: Optional[str] = None,
) -> str:
    # Repair legacy Windows/code-page damage before model-specific parsers
    # copy user-authored international text into an immutable prompt contract.
    prompt = repair_text(prompt)
    system_override = repair_text(system_override) if system_override else system_override
    reference_context = repair_text(reference_context) if reference_context else reference_context
    lora_system_hint = repair_text(lora_system_hint)
    is_h3_ref2va = (
        mode in ("video", "avatar")
        and (model_type or "").lower().startswith("minimax_h3_ref2va")
    )
    is_h3_context_ir = (
        mode in ("video", "avatar")
        and (model_type or "").lower().startswith("minimax_h3")
        and not is_h3_ref2va
    )
    is_h3_structured = is_h3_context_ir or is_h3_ref2va
    # If caller provides a system prompt override, use it directly (e.g., Director third-pass)
    if system_override:
        # Do NOT append the full model-specific enhance guide — the override is self-contained.
        # The enhance guides are designed for Studio mode (expand brief prompts) and would
        # contradict Director overrides (refine, don't expand).
        system = system_override
        # Inject content guidance
        from services.director.nsfw_guidance import inject_content_guidance
        system = inject_content_guidance(system, nsfw, "enhance")
        # Inject LoRA hints into system prompt
        if lora_system_hint:
            system = f"{system}\n\n{lora_system_hint}"

        # Disable thinking mode for system_override callers. The override path
        # is used exclusively by the Director third-pass polish, which is a
        # REFINEMENT task — there's no creative reasoning the LLM needs to do.
        # On Qwen3.5/3.6, thinking consumes the entire token budget before
        # any content is produced:
        #   [LLM] WARNING: Server returned empty content despite generating
        #         1024 tokens (model likely consumed all tokens on internal reasoning)
        #
        # THREE redundant suppression mechanisms — Qwen3.5/3.6 chat templates
        # in current llama.cpp builds ignore the first two, so the third
        # (hard stop sequence) is the actual safety net:
        #   1. enable_thinking=False     → chat_template_kwargs route (often ignored)
        #   2. /no_think user prefix     → Qwen Jinja template marker (often ignored)
        #   3. stop=["<think>", ...]     → llama-server stops generation when
        #      the model tries to enter thinking mode. Wastes ~1 token per call
        #      instead of the 1024-token budget. The polish pipeline's
        #      "unchanged passthrough" handler then falls back to the original
        #      Pass-2 prompt, so the user gets usable output even when polish
        #      is silently no-op'd by a stubborn thinking template.
        #
        # Net effect on a thinking-stubborn Qwen build: Pass 3 polish becomes
        # a ~free no-op (per-call cost drops from ~1024 tokens of waste to
        # ~1 token), and the user gets the unmodified Pass-2 prompt. Better
        # than the previous behavior of burning 26k+ tokens producing nothing.
        prompt_with_marker = f"/no_think\n\n{prompt}" if prompt else "/no_think"
        result = generate(
            prompt=prompt_with_marker,
            system_prompt=system,
            max_new_tokens=max(max_new_tokens, 1024),
            temperature=temperature,
            enable_thinking=False,
            stop=["<think>", "<thinking>"],
        )
        return repair_text(result).strip() if result else prompt

    # Dedicated per-model enhancer (e.g. Sulphur's uncensored enhancer): the
    # model is trained to enhance directly. Send the user's prompt (+ optional
    # image) with NO system prompt / guide and no thinking.
    if raw_enhancer_mode:
        # The fine-tuned enhancer (a) doesn't reliably honor a "write N
        # paragraphs" instruction and (b) likes to prepend a bogus "rendered
        # in a 3D animation style" clause. Always run the style cleanup; and
        # when the user gave one line per window, enhance each window
        # independently and collapse it to a single paragraph — that makes the
        # output EXACTLY window_count paragraphs regardless of the model.
        gen_kw = dict(
            system_prompt="", max_new_tokens=max(max_new_tokens, 256),
            temperature=temperature, enable_thinking=False,
        )
        lines = [ln.strip() for ln in (prompt or "").split("\n") if ln.strip()]
        if window_count and window_count > 1 and len(lines) == window_count:
            print(f"[Enhance] Raw enhancer: per-window x{window_count} ({model_type})")
            outs = []
            for i, ln in enumerate(lines):
                # Image only informs window 1; later windows continue from it.
                w_prompt = _build_enhance_user_prompt(
                    ln,
                    mode,
                    window_size_seconds,
                    1,
                    window_size_seconds,
                    model_type,
                )
                r = generate(prompt=w_prompt, image_paths=(image_paths if i == 0 else None), **gen_kw)
                r = _clean_enhancer_output(r)
                r = " ".join(r.split()) if r else ln  # collapse to one paragraph
                outs.append(r or ln)
            return repair_text("\n".join(outs))
        # Single call: 1-line "expand into N windows", or a line/window
        # mismatch. Falls back to the explicit-count instruction.
        raw_prompt = _build_enhance_user_prompt(
            prompt,
            mode,
            duration_seconds,
            window_count,
            window_size_seconds,
            model_type,
        )
        print(f"[Enhance] Raw enhancer ({model_type}, images={bool(image_paths)}, windows={window_count})")
        result = generate(prompt=raw_prompt, image_paths=image_paths, **gen_kw)
        return repair_text(_clean_enhancer_output(result) or prompt)

    # Try to load a model-specific guide
    system = None
    if model_type:
        try:
            from services.enhance_guides import get_enhance_guide
            has_images = bool(image_paths)
            system = get_enhance_guide(model_type, mode, has_images=has_images)
            print(f"[Enhance] Using model-specific guide for {model_type} ({mode}, images={has_images})")
        except Exception:
            pass

    # Fallback to generic prompts if no guide loaded
    if not system:
        system_prompts = {
            "video": (
                "You are an expert cinematic director. Enhance the user's video prompt "
                "with detailed descriptions of movements, camera angles, lighting, and "
                "environment. Keep under 150 words. Output only the enhanced prompt."
            ),
            "image": (
                "You are an expert photographer. Enhance the user's image prompt with "
                "detailed descriptions of composition, lighting, colors, and mood. "
                "The output is a STILL PHOTOGRAPH — describe static poses only, no motion verbs "
                "(no walking, running, reaching, turning, dancing). "
                "Keep under 150 words. Output only the enhanced prompt."
            ),
            "audio": (
                "You are an expert audio producer. Enhance the user's audio description "
                "with detailed descriptions of tone, pace, emotion, and sound qualities. "
                "Keep under 100 words. Output only the enhanced prompt."
            ),
        }
        system = system_prompts.get(mode, system_prompts["video"])

    # TTS-specific enhance: override system prompt with monologue/dialogue templates.
    #
    # Model handlers can supply their own per-mode enhancer prompts via the
    # `text_prompt_enhancer_instructions` (monologue) and
    # `text_prompt_enhancer_instructions1` (dialogue) keys on their model_def.
    # We check the model_def first — if the active model provides a custom
    # prompt, it takes precedence over the generic TTS_*_PROMPT defaults.
    #
    # This is the path Scenema relies on: its handler points
    # text_prompt_enhancer_instructions[1] at the rich markdown guides under
    # llm_guides/prompt_enhancer/ (single-speaker speech rules and two-speaker
    # dialogue rules). Before this lookup existed, the generic
    # TTS_QWEN3_DIALOGUE_PROMPT below always won regardless of which model the
    # user picked — producing "Peter:"/"Sarah:" character-labeled dialogue
    # instead of Scenema's `Speaker N{voice=..., gender=..., scene=...}: [cue]`
    # format. Result: Scenema's parser saw no `Speaker N:` headers and
    # collapsed the entire output into a single-voice block.
    if mode == "audio" and tts_enhance_mode:
        from models.TTS.prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT

        # Look up model-specific enhancer prompts (Scenema, Kugel, Qwen3-TTS,
        # Index-TTS2, Chatterbox, IndexTTS2 all set these on their model_def).
        # Lazy import keeps llm_service.py importable in environments where
        # wgp.py is unavailable (e.g. lightweight tooling, tests).
        model_specific_monologue = None
        model_specific_dialogue = None
        if model_type:
            try:
                from wgp import get_model_def
                md = get_model_def(model_type)
                if md:
                    model_specific_monologue = md.get("text_prompt_enhancer_instructions")
                    model_specific_dialogue = md.get("text_prompt_enhancer_instructions1")
            except Exception as e:
                print(f"[Enhance] Could not load model_def for {model_type}: {e}")

        if tts_enhance_mode in ("dialogue", "dialogue_fast"):
            # Check for voice count to customize speaker count
            voice_count = tts_voice_count
            if voice_count <= 1:
                system = model_specific_monologue or TTS_MONOLOGUE_PROMPT
            elif voice_count == 2:
                # Two-speaker path: prefer the model's dialogue prompt
                # (Scenema's markdown guide, Kugel's NotebookLM rules, etc.)
                # before falling back to the generic Qwen-style template.
                system = model_specific_dialogue or TTS_QWEN3_DIALOGUE_PROMPT
            else:
                # Multi-speaker (3-6): adapt the dialogue template. Both
                # Scenema and Kugel cap at 2 voices, so this branch is for
                # legacy TTS engines that support 3+ speakers (Qwen3-TTS et al).
                system = (
                    f"You are a creative dialogue writer for a text-to-speech model. "
                    f"Write an engaging, natural conversation with exactly {voice_count} characters based on the user prompt.\n\n"
                    f"Output rules:\n"
                    f"- Output ONLY dialogue lines. No explanations, stage directions, or narration.\n"
                    f"- Every line must start with a character name followed by a colon.\n"
                    f"- Use exactly {voice_count} characters. Use names from the prompt, or invent fitting names.\n"
                    f"- NO brackets, NO emotion tags, NO parenthetical directions.\n"
                    f"- Write NATURAL, REALISTIC dialogue with interruptions, reactions, varied sentence lengths.\n"
                    f"- Write a FULL conversation (20-40+ lines) unless the user specifies shorter.\n"
                    f"- Use clear punctuation. Commas and periods create natural pauses.\n"
                    f"- Make sure all {voice_count} characters participate meaningfully — don't let any character fade out.\n"
                )
        else:
            # Monologue path. Model-specific prompt wins over the generic
            # TTS_MONOLOGUE_PROMPT (so Scenema speech rules get used).
            system = model_specific_monologue or TTS_MONOLOGUE_PROMPT

        if model_specific_dialogue or model_specific_monologue:
            picked = "dialogue" if (tts_enhance_mode in ("dialogue", "dialogue_fast") and tts_voice_count == 2 and model_specific_dialogue) else ("monologue" if model_specific_monologue else "generic")
            print(f"[Enhance] Using model-specific {picked} prompt for {model_type}")

    # Inject NSFW enhance guidance when mature mode is on. Uses a SHARED,
    # VERSION-CONTROLLED guide (llm_guides/enhance/nsfw_shared.md) so it ships
    # via git to every install — the previous path read from the gitignored
    # supplement pack, which never travels through `git pull` (so edits never
    # reached the runtime). The guide is semi-clean: it LICENSES and DIRECTS
    # explicit anatomy, natural mature dialogue, and intensifier preservation
    # without containing graphic examples; the uncensored LLM supplies the
    # actual words. Applies to every nsfw_only model (Sulphur, 10Eros, ...) on
    # top of its clean per-model enhance_guide.
    if nsfw:
        from services.guide_loader import load_guide as _load_nsfw_guide
        nsfw_block = _load_nsfw_guide("enhance", "nsfw_shared")
        if nsfw_block:
            system = f"{system}\n\n{nsfw_block}"

    # Shared video-enhance rules — e.g. reference characters by a stable visual
    # appearance, not by name/relationship/pronoun (the model has no memory of
    # who anyone is). Appended for video so EVERY path gets it: per-model guides
    # (Sulphur, 10Eros) that don't include the generic LTX video guide, plus the
    # generic guide itself. Mirrors the Director-mode character-reference rule.
    # H3's guide already carries its own identity, pacing, and silence rules.
    # The generic appendix says to remove all character names and to write one
    # paragraph per sliding window, both of which conflict with H3's
    # knowledge-aware Context-IR format and single native timeline.
    if mode in ("video", "avatar") and not is_h3_structured:
        from services.guide_loader import load_guide as _load_vid_guide
        vid_block = _load_vid_guide("enhance", "video_shared")
        if vid_block:
            system = f"{system}\n\n{vid_block}"

    # Build the user prompt with context (duration + sliding-window count).
    # Shared with the raw per-model-enhancer path via the helper above.
    user_prompt = _build_enhance_user_prompt(
        prompt,
        mode,
        duration_seconds,
        window_count,
        window_size_seconds,
        model_type,
    )

    # Add image context
    if image_paths:
        if is_h3_ref2va:
            user_prompt = (
                "I have attached the image references from an ordered MiniMax H3 Omni-reference request. "
                "Use what you can see together with the exact label map below; references are identity/style/motion "
                "evidence and are not automatically an opening frame.\n\n"
                f"{reference_context or 'Use the supplied ordered reference labels.'}\n\n{user_prompt}"
            )
        elif is_h3_context_ir and reference_context:
            user_prompt = (
                "I have attached the ordered timeline images for a MiniMax H3 "
                "First / Last request. Preserve the exact Picture numbering and "
                "time alignment map below before the three Context-IR fields. "
                "Each injected picture is a frame-exact visual destination, not "
                "a general identity reference.\n\n"
                f"{reference_context}\n\n{user_prompt}"
            )
        elif mode == "image":
            user_prompt = f"I have attached a reference image. Enhance this prompt based on what you see in the image:\n\n{user_prompt}"
        else:
            user_prompt = f"I have attached a start frame image. Enhance this video prompt to match what you see in the image and describe what should happen:\n\n{user_prompt}"
        print(f"[Enhance] Sending {len(image_paths)} image(s) to vision LLM")
    elif is_h3_ref2va and reference_context:
        user_prompt = f"Ordered Omni-reference label map:\n{reference_context}\n\n{user_prompt}"
    elif is_h3_context_ir and reference_context:
        user_prompt = f"Ordered H3 timeline-picture map:\n{reference_context}\n\n{user_prompt}"

    # Inject LoRA hints into system prompt (NOT user prompt) so LLM treats them as instructions
    if lora_system_hint:
        system += f"\n\n{lora_system_hint}"

    # Preserve structural elements in image prompts
    if mode == "image":
        system += (
            '\n\nSTRUCTURAL RULES for image prompts:'
            '\n- If the prompt starts with "create new scene", keep that prefix.'
            '\n- If the prompt ends with "Use original reference images" or similar, keep that suffix.'
            '\n- ALWAYS end the prompt with: "Preserve character identity, attire, body attributes, and the art style of the reference image."'
            '\n- NEVER include LoRA names or filenames in the output.'
        )

    # Reinforce the output constraint. MiniMax H3 is intentionally different:
    # its field labels and <d> blocks are part of the model input, not prose
    # headers to strip. The generic "no labels" rule previously contradicted
    # the H3 guide and encouraged ordinary quote-mark dialogue.
    if is_h3_ref2va:
        system += (
            "\n\nCRITICAL MINIMAX H3 REF2VA OUTPUT CONTRACT: Output ONLY the six required fields, "
            "in order: subject_definitions:, summary:, retention_analysis:, detailed_description:, "
            "overall_soundscape:, and non_diegetic_music:. Use only the supplied <Picture n>, <Video n>, "
            "and <Audio n> labels. These labels and fields are model syntax, not explanatory headings. "
            "Every VOICE REFERENCE must be bound inside subject_definitions to its matching <Subject n> "
            "and stable (S1), (S2), etc. speaker ID. Spoken lines require that same ID and <d>[Language] literal "
            "words</d>. No markdown, "
            "explanation, filenames, or LoRA names."
        )
    elif is_h3_context_ir:
        system += (
            "\n\nCRITICAL MINIMAX H3 OUTPUT CONTRACT: Output ONLY the structured "
            "H3 prompt, with the exact field labels "
            "integrated_multimodal_description:, overall_soundscape:, and "
            "non_diegetic_music:. These labels are required model syntax, not "
            "explanatory headers. Every spoken line must have a stable (S1), "
            "(S2), etc. speaker ID and use <d>[Language] literal words</d>. "
            "When the user requests a discussion without supplying lines, write "
            "short meaningful dialogue that fits the supplied Duration. Once the "
            "last line ends, describe silent visible action and closed mouths; do "
            "not invent more speech. No markdown, explanation, or LoRA filenames."
        )
    else:
        system += "\n\nCRITICAL: Output ONLY the enhanced prompt text. No headers, no labels, no markdown, no explanation, no \"Enhancement Logic\", no \"Edit Prompt:\". No LoRA filenames (.safetensors). Just the raw prompt text."

    if is_h3_structured:
        dialogue_requirement = _build_h3_dialogue_requirement(prompt, duration_seconds)
        if dialogue_requirement:
            # Keep this adjacent to the output contract so a long vision guide
            # cannot demote literal dialogue into a vague "speaks" action.
            system += f"\n\n{dialogue_requirement}"

    # Scale max tokens for multi-window video prompts
    effective_max_tokens = max_new_tokens
    if window_count and window_count > 1:
        effective_max_tokens = max(max_new_tokens, window_count * 300 + 256)
    if is_h3_ref2va:
        effective_max_tokens = max(effective_max_tokens, 1200)
    elif is_h3_context_ir:
        # Leave enough room for the three required fields plus a compact timed
        # dialogue. Most H3 prompts finish well below this ceiling, but 512 can
        # truncate a vision-assisted 15-second rewrite before its sound fields.
        effective_max_tokens = max(effective_max_tokens, 768)

    # TTS: thinking mode for creative dialogue, disabled for fast mode
    is_tts = bool(tts_enhance_mode)
    is_fast = tts_enhance_mode and tts_enhance_mode.endswith('_fast')
    use_thinking = is_tts and not is_fast

    result = generate(
        prompt=user_prompt,
        system_prompt=system,
        max_new_tokens=effective_max_tokens,
        temperature=temperature,
        image_paths=image_paths,
        enable_thinking=use_thinking,
        thinking_budget=16384 if use_thinking else 4096,
        frequency_penalty=0.3,  # prevent repetition loops
        presence_penalty=0.1,   # encourage variety
    )
    result = repair_text(result)

    # Post-process ordinary prose aggressively, but preserve H3's required field
    # labels and media tags. The old substring-loop cleaner could truncate a
    # valid Context-IR response at its first repeated <Picture>/<Audio> mapping.
    if result:
        result = _clean_enhance_output(result, preserve_structure=is_h3_structured)

    structure_is_valid = (
        _has_complete_h3_ref2va_structure(result)
        if is_h3_ref2va
        else _has_complete_h3_context_structure(result)
    ) if is_h3_structured else True
    dialogue_is_valid = _h3_dialogue_contract_satisfied(prompt, result) if is_h3_structured else True
    timed_silence_is_valid = (
        _h3_timed_silence_contract_satisfied(prompt, result, duration_seconds)
        if is_h3_structured
        else True
    )
    voice_binding_is_valid = (
        _h3_voice_binding_contract_satisfied(result, reference_context)
        if is_h3_ref2va
        else True
    )

    # Small local LLMs can either repeat the first Ref2VA mapping or summarize
    # quoted dialogue as the word "speaks". Retry malformed H3 output once with
    # the immutable dialogue contract adjacent to the shape constraint.
    if is_h3_structured and not (
        structure_is_valid
        and dialogue_is_valid
        and timed_silence_is_valid
        and voice_binding_is_valid
    ):
        failures = []
        if not structure_is_valid:
            failures.append("structure")
        if not dialogue_is_valid:
            failures.append("dialogue")
        if not timed_silence_is_valid:
            failures.append("timed silence")
        if not voice_binding_is_valid:
            failures.append("voice binding")
        print(f"[Enhance] Invalid MiniMax H3 {'/'.join(failures)}; retrying once.")
        field_requirement = (
            "Emit each of the six required field labels exactly once, in order."
            if is_h3_ref2va
            else "Emit each of the three required field labels exactly once, in order."
        )
        retry = generate(
            prompt=user_prompt,
            system_prompt=(
                system
                + f"\n\nRETRY REQUIREMENT: Be concise. {field_requirement} "
                "Do not repeat a subject definition or reference mapping. Never replace a requested "
                "spoken line with the words 'speaks', 'talks', or 'dialogue'; write the actual <d> block."
            ),
            max_new_tokens=effective_max_tokens,
            temperature=min(float(temperature), 0.35),
            image_paths=image_paths,
            enable_thinking=False,
            thinking_budget=4096,
            frequency_penalty=0.6,
            presence_penalty=0.15,
        )
        retry = repair_text(retry)
        retry = _clean_enhance_output(retry, preserve_structure=True) if retry else ""
        retry_structure_is_valid = (
            _has_complete_h3_ref2va_structure(retry)
            if is_h3_ref2va
            else _has_complete_h3_context_structure(retry)
        )
        retry_dialogue_is_valid = _h3_dialogue_contract_satisfied(prompt, retry)
        retry_timed_silence_is_valid = _h3_timed_silence_contract_satisfied(
            prompt,
            retry,
            duration_seconds,
        )
        retry_voice_binding_is_valid = (
            _h3_voice_binding_contract_satisfied(retry, reference_context)
            if is_h3_ref2va
            else True
        )
        if (
            retry_structure_is_valid
            and retry_dialogue_is_valid
            and retry_timed_silence_is_valid
            and retry_voice_binding_is_valid
        ):
            result = retry
        else:
            print("[Enhance] H3 retry was incomplete; using deterministic structured fallback.")
            result = (
                _build_h3_ref2va_tagged_fallback(
                    prompt,
                    reference_context,
                    duration_seconds=duration_seconds,
                )
                if is_h3_ref2va
                else _build_h3_context_fallback(
                    prompt,
                    has_start_image=(
                        bool(image_paths)
                        and not bool(reference_context)
                    ),
                    reference_context=reference_context,
                    duration_seconds=duration_seconds,
                )
            )

    # If two full rewrites still summarize a vague request as "they discuss",
    # ask the local LLM for only the missing exchange. This rare focused pass is
    # cheaper and more reliable than accepting a prompt that makes H3 improvise.
    if (
        is_h3_structured
        and _h3_requests_speech(prompt)
        and not _extract_h3_quoted_dialogue(prompt)
        and not _h3_dialogue_contract_satisfied(prompt, result)
    ):
        word_budget = max(4, int(duration_seconds or 8))
        dialogue_language = _detect_h3_dialogue_language(prompt)
        print("[Enhance] H3 discussion still has no dialogue; generating a focused exchange.")
        dialogue_fragment = generate(
            prompt=(
                f"Duration: {duration_seconds or 8} seconds. Total dialogue budget: at most "
                f"{word_budget} spoken words. Request: {prompt}"
            ),
            system_prompt=(
                "Write only the concise dialogue requested by the user. Output one to three lines in "
                f"the exact form 'Speaker description (S1): <d>[{dialogue_language}] Literal words.</d>', using "
                "stable sequential speaker IDs. Communicate the requested topic. No narration, "
                "markdown, quotation marks, headings, or dialogue beyond the word budget."
            ),
            max_new_tokens=min(320, effective_max_tokens),
            temperature=min(float(temperature), 0.5),
            image_paths=None,
            enable_thinking=False,
            thinking_budget=2048,
            frequency_penalty=0.4,
            presence_penalty=0.1,
        )
        dialogue_fragment = repair_text(dialogue_fragment)
        dialogue_fragment = (
            _clean_enhance_output(dialogue_fragment, preserve_structure=True)
            if dialogue_fragment
            else ""
        )
        if _extract_h3_dialogue_blocks(dialogue_fragment):
            result = _inject_h3_generated_dialogue(
                result,
                dialogue_fragment,
                ref2va=is_h3_ref2va,
            )
        else:
            print("[Enhance] Focused H3 dialogue pass returned no valid <d> block.")

    # Explicit user dialogue is immutable. Even if both LLM attempts omit it,
    # compile every quoted line into H3 syntax before returning the prompt.
    if is_h3_structured and not _h3_dialogue_contract_satisfied(prompt, result):
        result = _inject_missing_h3_dialogue(result, prompt, ref2va=is_h3_ref2va)
    if is_h3_structured:
        result = _strip_h3_untagged_dialogue_duplicates(result, prompt)
        result = _enforce_h3_soundscape_silence(result, prompt)
        result = _enforce_h3_music_request(result, prompt, reference_context)
    if is_h3_context_ir and image_paths:
        result = _ensure_h3_visual_grounding(
            result,
            prompt,
            image_paths,
            generate_fn=generate,
        )
    return repair_text(result)


_H3_REF2VA_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
_H3_CONTEXT_FIELDS = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_H3_LANGUAGE_ALIASES = (
    ("mandarin chinese", "Mandarin Chinese"),
    ("mandarin", "Mandarin Chinese"),
    ("cantonese", "Cantonese"),
    ("brazilian portuguese", "Brazilian Portuguese"),
    ("portuguese", "Portuguese"),
    ("french", "French"),
    ("spanish", "Spanish"),
    ("german", "German"),
    ("italian", "Italian"),
    ("japanese", "Japanese"),
    ("korean", "Korean"),
    ("chinese", "Chinese"),
    ("hindi", "Hindi"),
    ("arabic", "Arabic"),
    ("russian", "Russian"),
    ("dutch", "Dutch"),
    ("polish", "Polish"),
    ("turkish", "Turkish"),
    ("swedish", "Swedish"),
    ("norwegian", "Norwegian"),
    ("danish", "Danish"),
    ("finnish", "Finnish"),
    ("greek", "Greek"),
    ("hebrew", "Hebrew"),
    ("ukrainian", "Ukrainian"),
    ("czech", "Czech"),
    ("romanian", "Romanian"),
    ("hungarian", "Hungarian"),
    ("thai", "Thai"),
    ("vietnamese", "Vietnamese"),
    ("indonesian", "Indonesian"),
    ("filipino", "Filipino"),
    ("tagalog", "Tagalog"),
    ("english", "English"),
)


def _canonical_h3_language_tag(value: str) -> str:
    normalized = " ".join(str(value or "").strip().casefold().split())
    for alias, canonical in _H3_LANGUAGE_ALIASES:
        if normalized == alias or normalized == canonical.casefold():
            return canonical
    return str(value or "").strip()


def _detect_h3_dialogue_language(prompt: str) -> str:
    """Return the explicitly requested H3 speech language, else English.

    A language word only counts near speech/language wording, so a setting
    such as "a French restaurant" does not accidentally change the dialogue.
    Existing ``<d>[Language]`` syntax is authoritative.
    """

    import re

    text = repair_text(prompt)
    explicit = re.search(r"<d>\s*\[([^\]\r\n]+)\]", text, flags=re.IGNORECASE)
    if explicit:
        return _canonical_h3_language_tag(explicit.group(1)) or "English"

    context_word = re.compile(
        r"\b(?:in|speak|speaks|speaking|spoken|say|says|saying|talk|talks|"
        r"talking|dialogue|sentence|line|words?|language|speech|voice|voiced)\b",
        flags=re.IGNORECASE,
    )
    suffix_word = re.compile(
        r"\b(?:dialogue|language|sentence|line|words?|speech|voice|speaking|spoken)\b",
        flags=re.IGNORECASE,
    )
    lowered = text.casefold()
    for alias, canonical in _H3_LANGUAGE_ALIASES:
        for match in re.finditer(rf"\b{re.escape(alias)}\b", lowered):
            before = lowered[max(0, match.start() - 60):match.start()]
            after = lowered[match.end():match.end() + 40]
            if context_word.search(before) or suffix_word.search(after):
                return canonical
    return "English"


def _extract_h3_quoted_dialogue(text: str) -> list[str]:
    """Extract explicit straight- or curly-quoted speech in source order."""
    import re
    matches = []
    for match in re.finditer(r'"([^"\r\n]{1,500})"|“([^”\r\n]{1,500})”', str(text or "")):
        value = (match.group(1) or match.group(2) or "").strip()
        if value:
            matches.append(value)
    return matches


def _h3_requests_speech(text: str) -> bool:
    import re
    return bool(
        _extract_h3_quoted_dialogue(text)
        or re.search(
            r"\b(?:say|says|speak|speaks|talk|talks|discuss|discusses|discussion|"
            r"argue|argues|announce|announces|ask|asks|reply|replies|tell|tells|"
            r"conversation|dialogue)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _extract_h3_dialogue_blocks(text: str) -> list[str]:
    import re
    return [
        match.strip()
        for match in re.findall(
            r"<d>\s*\[[^\]]+\]\s*(.*?)\s*</d>",
            str(text or ""),
            flags=re.DOTALL,
        )
        if match.strip()
    ]


def _extract_h3_dialogue_entries(text: str) -> list[tuple[str, str]]:
    import re
    return [
        (_canonical_h3_language_tag(language), words.strip())
        for language, words in re.findall(
            r"<d>\s*\[([^\]]+)\]\s*(.*?)\s*</d>",
            str(text or ""),
            flags=re.DOTALL | re.IGNORECASE,
        )
        if words.strip()
    ]


def _h3_dialogue_schedule(prompt: str, duration_seconds: Optional[float]) -> tuple[float, float, float]:
    """Choose an early bounded speech interval and leave useful silent action around it."""
    duration = max(2.0, float(duration_seconds or 8.0))
    quotes = _extract_h3_quoted_dialogue(prompt)
    if quotes:
        word_count = sum(len(line.split()) for line in quotes)
    else:
        # Vague discussion requests still need room for reactions and action.
        word_count = max(4, int(duration))
    speech_duration = max(1.0, word_count / 2.0)
    speech_duration = min(speech_duration, max(1.0, duration * 0.55))
    start = max(0.5, duration * 0.2)
    start = min(start, max(0.25, duration - speech_duration - 0.75))
    end = min(duration - 0.25, start + speech_duration)
    return duration, start, end


def _build_h3_timed_silence_clause(prompt: str, duration_seconds: Optional[float]) -> str:
    if not _h3_requests_speech(prompt):
        return ""
    duration, start, end = _h3_dialogue_schedule(prompt, duration_seconds)
    return (
        f"From 0.00 to {start:.2f} seconds, show active scene-appropriate nonverbal action rather "
        "than idle staring; every mouth stays completely closed and the audio contains no human "
        "voice. Begin the first tagged line at approximately "
        f"{start:.2f} seconds and finish all <d> dialogue by approximately {end:.2f} seconds. "
        f"From {end:.2f} to {duration:.2f} seconds, fill the remaining timeline with concrete "
        "nonverbal action, reactions, camera development, ambience, and synchronized practical "
        "effects. Outside the tagged interval there are no voices, whispers, grunts, audible "
        "breathing, or speech-like vocalizations, and every mouth remains closed."
    )


def _build_h3_dialogue_requirement(
    prompt: str,
    duration_seconds: Optional[float] = None,
) -> str:
    quotes = _extract_h3_quoted_dialogue(prompt)
    language = _detect_h3_dialogue_language(prompt)
    timed_clause = _build_h3_timed_silence_clause(prompt, duration_seconds)
    if quotes:
        required = "\n".join(
            f"- REQUIRED VERBATIM: <d>[{language}] {line}</d>" for line in quotes
        )
        return (
            "IMMUTABLE H3 DIALOGUE CONTRACT: The user supplied the spoken lines below. "
            "Every line must appear verbatim inside a <d> block in the output; do not summarize, "
            "paraphrase, censor, omit, or add speech. Give each line a stable (S1), (S2), etc. "
            f"speaker outside its tag. Never repeat these words as ordinary quoted text in summary "
            f"or any other field.\n{required}\n{timed_clause}"
        )
    if _h3_requests_speech(prompt):
        return (
            "MANDATORY H3 DIALOGUE CONTRACT: The user explicitly requests speech but supplied no "
            "script. Write concise, meaningful dialogue that communicates the requested subject, "
            f"using stable speaker IDs and one or more <d>[{language}] literal words</d> blocks. "
            "Writing only 'speaks', 'talks', or 'they discuss' makes the output invalid. "
            f"{timed_clause}"
        )
    return ""


def _h3_dialogue_contract_satisfied(prompt: str, result: str) -> bool:
    import re
    quotes = _extract_h3_quoted_dialogue(prompt)
    blocks = _extract_h3_dialogue_blocks(result)
    entries = _extract_h3_dialogue_entries(result)
    language = _detect_h3_dialogue_language(prompt)
    has_speaker_id = bool(re.search(r"\(S\d+\)", str(result or "")))
    if quotes:
        return has_speaker_id and all(
            any(entry_language == language and words == line for entry_language, words in entries)
            for line in quotes
        )
    if _h3_requests_speech(prompt):
        return has_speaker_id and bool(blocks) and all(
            entry_language == language for entry_language, _words in entries
        )
    return True


_H3_VISUAL_CATEGORY_PATTERNS = (
    r"\b(?:camera|shot|frame|framing|foreground|midground|background|screen[- ](?:left|right)|"
    r"close[- ]?up|medium[- ]?shot|wide[- ]?shot|angle|lens|composition)\b",
    r"\b(?:woman|man|person|people|child|face|hair|eyes?|build|silhouette|subject|character)\b",
    r"\b(?:wearing|dressed|wardrobe|shirt|jacket|coat|dress|suit|trousers|pants|skirt|shoes?|hat)\b",
    r"\b(?:interior|exterior|room|street|kitchen|office|building|wall|window|door|furniture|"
    r"table|desk|landscape|environment|setting)\b",
    r"\b(?:light|lighting|lit|shadow|sunlight|neon|warm|cool|bright|dim|color|palette|contrast)\b",
)


def _h3_visual_category_count(text: str) -> int:
    import re
    return sum(
        1 for pattern in _H3_VISUAL_CATEGORY_PATTERNS
        if re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    )


def _h3_visual_grounding_contract_satisfied(prompt: str, result: str) -> bool:
    """Reject H3 start-frame rewrites that merely repackage user prose."""

    import re
    match = re.search(
        r"(?ms)^\s*integrated_multimodal_description\s*:(.*?)"
        r"(?=^\s*overall_soundscape\s*:)",
        str(result or ""),
    )
    if not match:
        return False
    description = match.group(1)
    category_count = _h3_visual_category_count(description)
    if category_count < 3:
        return False

    source_words = {
        word for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}", str(prompt or "").casefold())
    }
    description_words = {
        word for word in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{4,}", description.casefold())
    }
    novel_words = description_words - source_words - {
        "shot", "video", "seconds", "target", "picture", "requested",
        "scene", "shows", "visible", "camera", "final", "frame",
    }
    # A user's already-rich prompt needs less invention; otherwise require
    # enough genuinely new visual vocabulary to show that the image was used.
    return len(novel_words) >= 8 or _h3_visual_category_count(prompt) >= 3


def _inject_h3_visual_anchor(result: str, anchor: str) -> str:
    """Insert a compact observed-frame description into the H3 visual field."""

    import re
    anchor = repair_text(anchor)
    # A vision-only repair is never allowed to introduce extra speech.
    anchor = re.sub(r"(?is)<\s*d\s*>.*?<\s*/\s*d\s*>", "", anchor)
    anchor = re.sub(r"(?is)<\s*/?\s*d\s*>", "", anchor)
    anchor = re.sub(r"(?m)^\s*(?:#{1,4}\s*)?(?:visual anchor|description)\s*:\s*", "", anchor)
    anchor = " ".join(anchor.replace("**", "").split()).strip()
    if not anchor or _h3_visual_category_count(anchor) < 3:
        return result
    anchor = anchor[:1800].rstrip()

    match = re.search(
        r"(?mi)^\s*integrated_multimodal_description\s*:\s*",
        str(result or ""),
    )
    if not match:
        return result
    insert_at = match.end()
    remainder = result[insert_at:]
    shot = re.match(r"\[Shot\s+1\]\s*", remainder, flags=re.IGNORECASE)
    if shot:
        insert_at += shot.end()
    prefix = "Attached-frame visual evidence in supplied timeline order: "
    return result[:insert_at] + prefix + anchor + ". " + result[insert_at:]


def _ensure_h3_visual_grounding(
    result: str,
    prompt: str,
    image_paths: Optional[list],
    *,
    generate_fn,
) -> str:
    """Run one focused vision pass when the main H3 rewrite ignored images."""

    if not image_paths or _h3_visual_grounding_contract_satisfied(prompt, result):
        return result
    print("[Enhance] H3 rewrite lacked concrete frame evidence; grounding from attached image(s).")
    try:
        anchor = generate_fn(
            prompt=(
                f"There are {len(image_paths)} attached target-frame image(s), in order. "
                "Describe only concrete visible facts needed to preserve them in a generated video. "
                f"The separate action request is: {prompt}"
            ),
            system_prompt=(
                "Act as a visual continuity observer. Return one compact factual paragraph. "
                "For every attached image, describe visible subject count and appearance, wardrobe, "
                "screen position and composition, setting and important objects, lighting and color, "
                "and camera framing. Do not invent identity, dialogue, action, emotion, or unseen facts. "
                "Do not output markdown, field labels, quotation marks, or model instructions."
            ),
            max_new_tokens=max(240, min(700, len(image_paths) * 220)),
            temperature=0.15,
            image_paths=image_paths,
            enable_thinking=False,
            thinking_budget=2048,
            frequency_penalty=0.2,
            presence_penalty=0.0,
        )
    except Exception as exc:
        print(f"[Enhance] Focused H3 visual grounding was unavailable: {exc}")
        return result
    grounded = _inject_h3_visual_anchor(result, repair_text(anchor))
    if grounded == result:
        print("[Enhance] Focused H3 visual pass returned no usable grounding details.")
    return grounded


def _h3_timed_silence_contract_satisfied(
    prompt: str,
    result: str,
    duration_seconds: Optional[float],
) -> bool:
    """Require explicit non-vocal time allocation around requested speech."""
    if not _h3_requests_speech(prompt):
        return True
    import re
    text = str(result or "")
    has_opening_interval = bool(re.search(r"(?i)\bfrom\s+0(?:\.0+)?\s+(?:to|until)", text))
    has_closed_mouths = bool(re.search(r"(?i)\bmouths?\b.{0,50}\bclosed\b", text))
    has_no_voice = bool(
        re.search(r"(?i)\b(?:no|without)\s+(?:human\s+)?(?:voices?|speech|vocal)", text)
    )
    has_remaining_interval = bool(
        re.search(r"(?i)\bfrom\s+\d+(?:\.\d+)?\s+(?:to|until)\s+\d+(?:\.\d+)?\s+seconds", text)
    )
    return has_opening_interval and has_closed_mouths and has_no_voice and has_remaining_interval


def _h3_voice_binding_contract_satisfied(
    result: str,
    reference_context: Optional[str],
) -> bool:
    """Require every Omni voice reference inside the subject/speaker mapping."""
    import re
    voice_labels = re.findall(
        r"(?mi)^(<Audio\s+\d+>).*?intent=VOICE REFERENCE.*$",
        str(reference_context or ""),
    )
    if not voice_labels:
        return True
    match = re.search(
        r"(?ms)^\s*subject_definitions\s*:(.*?)(?=^\s*summary\s*:)",
        str(result or ""),
    )
    if not match:
        return False
    definitions = match.group(1)
    return all(label in definitions for label in voice_labels) and bool(
        re.search(r"\(S\d+\)", definitions)
    )


def _has_complete_h3_ref2va_structure(text: str) -> bool:
    """Return true only for one complete, ordered six-field Ref2VA prompt."""
    if not text:
        return False
    import re
    positions = []
    for field in _H3_REF2VA_FIELDS:
        matches = list(re.finditer(rf"(?mi)^\s*{re.escape(field)}\s*:", text))
        if len(matches) != 1:
            return False
        positions.append(matches[0].start())
    return positions == sorted(positions)


def _has_complete_h3_context_structure(text: str) -> bool:
    """Return true only for one complete, ordered three-field H3 prompt."""
    if not text:
        return False
    import re
    positions = []
    for field in _H3_CONTEXT_FIELDS:
        matches = list(re.finditer(rf"(?mi)^\s*{re.escape(field)}\s*:", text))
        if len(matches) != 1:
            return False
        positions.append(matches[0].start())
    return positions == sorted(positions)


def _compile_h3_explicit_dialogue(prompt: str) -> str:
    """Replace user quotation marks with literal H3 dialogue blocks."""
    import re
    counter = 0
    language = _detect_h3_dialogue_language(prompt)

    def replace(match):
        nonlocal counter
        counter += 1
        value = (match.group(1) or match.group(2) or "").strip()
        return f"(S{counter}) <d>[{language}] {value}</d>"

    return re.sub(
        r'"([^"\r\n]{1,500})"|“([^”\r\n]{1,500})”',
        replace,
        str(prompt or ""),
    )


def _inject_missing_h3_dialogue(result: str, prompt: str, *, ref2va: bool) -> str:
    """Deterministically append omitted literal dialogue to the correct H3 field."""
    quotes = _extract_h3_quoted_dialogue(prompt)
    if not quotes:
        return result
    existing = set(_extract_h3_dialogue_blocks(result))
    missing = [line for line in quotes if line not in existing]
    if not missing:
        return result
    language = _detect_h3_dialogue_language(prompt)
    additions = " ".join(
        f"The intended speaker (S{index}) says exactly once: <d>[{language}] {line}</d>."
        for index, line in enumerate(missing, start=1)
    )
    additions += (
        " These are the only spoken words in the video; before and after them, everyone remains "
        "silent with mouths closed, with no other voices or speech-like vocalization."
    )
    field = "detailed_description" if ref2va else "integrated_multimodal_description"
    next_field = "overall_soundscape"
    import re
    pattern = re.compile(
        rf"(?ms)(^\s*{re.escape(field)}\s*:.*?)(?=^\s*{re.escape(next_field)}\s*:)",
    )
    if pattern.search(result or ""):
        return pattern.sub(
            lambda match: match.group(1).rstrip() + " " + additions + "\n",
            result,
            count=1,
        )
    return f"{result or ''}\n{field}: {additions}".strip()


def _inject_h3_generated_dialogue(result: str, fragment: str, *, ref2va: bool) -> str:
    """Insert a focused generated exchange while discarding non-dialogue prose."""
    valid_lines = [
        line.strip()
        for line in str(fragment or "").splitlines()
        if "<d>" in line and "</d>" in line
    ][:3]
    if not valid_lines:
        return result
    addition = (
        " The complete requested exchange is: "
        + " ".join(valid_lines)
        + " No other words are spoken; afterward everyone remains silent with mouths closed."
    )
    field = "detailed_description" if ref2va else "integrated_multimodal_description"
    import re
    pattern = re.compile(
        rf"(?ms)(^\s*{re.escape(field)}\s*:.*?)(?=^\s*overall_soundscape\s*:)",
    )
    if pattern.search(result or ""):
        return pattern.sub(
            lambda match: match.group(1).rstrip() + addition + "\n",
            result,
            count=1,
        )
    return f"{result or ''}\n{field}:{addition}".strip()


def _strip_h3_untagged_dialogue_duplicates(result: str, prompt: str) -> str:
    """Remove summary/narration copies of dialogue that already belongs in <d>."""
    source_lines = _extract_h3_quoted_dialogue(prompt)
    if not source_lines:
        return result
    import re

    def normalized(value: str) -> str:
        return " ".join(value.strip().rstrip(".,!?;:").casefold().split())

    source_values = {normalized(line) for line in source_lines}
    protected: list[str] = []

    def stash(match):
        protected.append(match.group(0))
        return f"@@MAESTRO_H3_DIALOGUE_{len(protected) - 1}@@"

    text = re.sub(r"<d>.*?</d>", stash, str(result or ""), flags=re.DOTALL)

    def replace_quote(match):
        value = match.group(1) or match.group(2) or ""
        return "the scripted line" if normalized(value) in source_values else match.group(0)

    text = re.sub(
        r'"([^"\r\n]{1,500})"|\u201c([^\u201d\r\n]{1,500})\u201d',
        replace_quote,
        text,
    )
    for index, block in enumerate(protected):
        text = text.replace(f"@@MAESTRO_H3_DIALOGUE_{index}@@", block)
    return text


def _enforce_h3_soundscape_silence(result: str, prompt: str) -> str:
    """Keep the model from filling dialogue gaps with invented human noises."""
    if not _h3_requests_speech(prompt):
        return result
    import re
    if re.search(
        r"(?i)\b(?:grunt|gasp|scream|laugh|sob|cry|audible breathing|nonverbal vocal)\w*\b",
        str(prompt or ""),
    ):
        return result

    pattern = re.compile(
        r"(?ms)(^\s*overall_soundscape\s*:)(.*?)(?=^\s*non_diegetic_music\s*:)",
    )
    match = pattern.search(str(result or ""))
    if not match:
        return result
    content = match.group(2).strip()
    sentences = re.split(r"(?<=[.!?])\s+", content)
    kept = []
    for sentence in sentences:
        has_negation = re.search(r"(?i)\b(?:no|not|without|never)\b", sentence)
        if has_negation:
            kept.append(sentence.strip())
            continue
        # Remove only comma/semicolon clauses that introduce vocal filler so a
        # mixed sentence keeps its impacts, ambience, and debris sounds.
        safe_segments = [
            segment.strip()
            for segment in re.split(r"[,;]", sentence)
            if segment.strip()
            and not re.search(
                r"(?i)\b(?:grunt|gasp|whisper|scream|laugh|breath|voice|speech-like)\w*\b",
                segment,
            )
        ]
        if safe_segments:
            kept.append(", ".join(safe_segments))
    kept.append(
        "Outside the tagged dialogue, no human voices, whispers, grunts, audible breathing, "
        "or speech-like vocalizations occur"
    )
    replacement = match.group(1) + " " + ". ".join(kept).rstrip(".") + ".\n"
    return result[:match.start()] + replacement + result[match.end():]


def _enforce_h3_music_request(
    result: str,
    prompt: str,
    reference_context: Optional[str],
) -> str:
    """Do not turn visual words such as 'cinematic' into an invented score."""
    import re
    requests_music = bool(
        re.search(
            r"(?i)\b(?:music|song|score|soundtrack|orchestra|orchestral|instrumental|melody|theme)\b",
            str(prompt or ""),
        )
    )
    mapped_music = bool(
        re.search(
            r"(?i)(?:AUDIO REUSE / PERFORMANCE DRIVER|Sound / music style|intent=AUDIO REFERENCE)",
            str(reference_context or ""),
        )
    )
    if requests_music or mapped_music:
        return result
    return re.sub(
        r"(?ms)^\s*non_diegetic_music\s*:.*\Z",
        "non_diegetic_music: N/A",
        str(result or ""),
    )


def _build_h3_ref2va_tagged_fallback(
    prompt: str,
    reference_context: Optional[str],
    *,
    duration_seconds: Optional[float] = None,
) -> str:
    """Create a deterministic six-field fallback when the local LLM loops."""
    raw_mapping = reference_context or "Use the supplied ordered references according to their roles."
    mapping = " ".join(raw_mapping.split())
    import re
    picture_labels = re.findall(r"<Picture\s+\d+>", raw_mapping)
    voice_labels = re.findall(
        r"(?mi)^(<Audio\s+\d+>).*?intent=VOICE REFERENCE.*$",
        raw_mapping,
    )
    subject_bindings = []
    for index, picture_label in enumerate(picture_labels, start=1):
        subject_bindings.append(
            f"<Subject {index}> (S{index}) takes visual identity from {picture_label}."
        )
    for index, audio_label in enumerate(voice_labels, start=1):
        subject_index = min(index, max(1, len(picture_labels)))
        subject_bindings.append(
            f"{audio_label} is the voice-timbre reference for <Subject {subject_index}> (S{subject_index})."
        )
    subject_mapping = " ".join(subject_bindings + [mapping])
    request = _compile_h3_explicit_dialogue(prompt)
    timed_clause = _build_h3_timed_silence_clause(prompt, duration_seconds)
    return (
        f"subject_definitions: {subject_mapping}\n"
        "summary: A finished video matching the requested action, identity, setting, and explicitly "
        "tagged dialogue.\n"
        f"retention_analysis: Preserve the mapped identity, motion, and audio roles exactly: {mapping}\n"
        f"detailed_description: The finished target video follows this request: {request} "
        "Reference pictures provide identity and appearance only, never their original background, "
        "framing, pose, or an opening still. The scripted dialogue is the only speech; all mouths "
        f"remain closed before and after it. {timed_clause}\n"
        "overall_soundscape: Continuous scene-appropriate stereo ambience and synchronized practical "
        "sound effects begin at the first frame and continue naturally underneath dialogue. Outside "
        "tagged dialogue there are no human voices, whispers, grunts, audible breathing, or "
        "speech-like vocalizations.\n"
        "non_diegetic_music: N/A"
    )


def _build_h3_context_fallback(
    prompt: str,
    *,
    has_start_image: bool,
    reference_context: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> str:
    """Create a deterministic three-field fallback for ordinary H3 Base."""
    alignment = str(reference_context or "").strip()
    if alignment:
        alignment += "\n\n"
    elif has_start_image:
        alignment = (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> "
            "(from [Shot 1]) is fully referenced.\n\n"
        )
    request = _compile_h3_explicit_dialogue(prompt)
    timed_clause = _build_h3_timed_silence_clause(prompt, duration_seconds)
    return (
        f"{alignment}integrated_multimodal_description: [Shot 1] {request} The scripted dialogue "
        f"is the only speech; all mouths remain closed before and after it. {timed_clause}\n\n"
        "overall_soundscape: Continuous scene-appropriate ambience and synchronized practical "
        "sound effects begin at the first frame and continue naturally underneath dialogue. Outside "
        "tagged dialogue there are no human voices, whispers, grunts, audible breathing, or "
        "speech-like vocalizations.\n\n"
        "non_diegetic_music: N/A"
    )


def _clean_enhance_output(text: str, preserve_structure: bool = False) -> str:
    """Strip markdown formatting, headers, explanation, and repetition loops from enhance output."""
    import re
    # Remove markdown bold/headers
    if preserve_structure:
        text = text.replace('**', '')
    else:
        text = re.sub(r'\*\*.*?\*\*:?\s*', '', text)
    # Remove markdown headers
    if preserve_structure:
        text = re.sub(r'^\s*#{1,4}\s*', '', text, flags=re.MULTILINE)
    else:
        text = re.sub(r'^#{1,4}\s+.*$', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    # Remove common label prefixes the model adds
    text = re.sub(r'^(?:Edit Prompt|Enhanced Prompt|Prompt|Output|Result|Enhancement Logic|Here is)[:\s]*', '', text, flags=re.IGNORECASE)
    # Remove leading/trailing quotes if the entire output is quoted
    text = text.strip()
    if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
        text = text[1:-1]
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # H3 Context-IR legitimately repeats subject and reference labels across
    # sections. Structure validation/retry handles malformed H3 output without
    # destroying those mappings.
    cleaned = text.strip()
    if preserve_structure:
        return cleaned

    # Detect and truncate repetition loops: if any 20+ char substring repeats 3+ times, keep only the first occurrence
    for chunk_len in range(30, 15, -1):
        if len(cleaned) < chunk_len * 3:
            continue
        for start in range(len(cleaned) - chunk_len * 2):
            chunk = cleaned[start:start + chunk_len]
            if chunk in cleaned[start + chunk_len:start + chunk_len * 3]:
                # Found a repeating pattern — truncate at the first repetition
                cleaned = cleaned[:start + chunk_len].rstrip('. ,;') + '.'
                print(f"[Enhance] Truncated repetition loop at position {start} (pattern: '{chunk[:40]}...')")
                return cleaned

    return cleaned


def describe_image(
    image_path: str,
    prompt: str = "Describe this image in detail for use as a video generation prompt.",
    max_new_tokens: int = 256,
) -> str:
    """Describe an image. Vision support requires multimodal GGUF (future)."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    basename = os.path.basename(image_path)
    return generate(
        prompt=f"The user has an image file named '{basename}'. {prompt}",
        system_prompt="You are a helpful assistant that generates creative, detailed video prompts.",
        max_new_tokens=max_new_tokens,
        temperature=0.4,
    )


def _section_hints(section: str) -> str:
    """Return cinematic variation hints based on section type."""
    hints = {
        "intro": "establishing wide shot, slow reveal, atmospheric, moody lighting",
        "verse": "medium shots, storytelling, character focus, steady camera",
        "chorus": "dynamic angles, fast cuts, peak energy, bold colors, wide and close-up mix",
        "bridge": "change of scenery, dreamy or surreal, unique angle, slow motion",
        "outro": "pulling back, reflective, fading light, wide shot",
        "instrumental": "abstract visuals, dramatic camera sweep, focus on environment",
    }
    return hints.get(section, "creative angle, vivid scene")


def _build_clip_description(clip: dict, index: int, lyrics: Optional[list] = None) -> str:
    """Build a rich description for a single clip."""
    start = clip.get("start", 0)
    end = clip.get("end", 0)
    section = clip.get("section_label", "verse")
    energy = clip.get("energy", 0.5)

    if energy > 0.7:
        energy_word = "very high energy, intense"
    elif energy > 0.5:
        energy_word = "high energy, dynamic"
    elif energy > 0.3:
        energy_word = "moderate energy, steady"
    else:
        energy_word = "low energy, calm"

    clip_lyrics = ""
    if lyrics:
        matching = [l["text"] for l in lyrics
                   if l["start"] < end and l["end"] > start]
        if matching:
            clip_lyrics = f'\n   Lyrics: "{" ".join(matching)}"'

    hints = _section_hints(section)
    return (
        f"Clip {index + 1} ({start:.1f}s-{end:.1f}s): {section} section, {energy_word}.\n"
        f"   Cinematic direction: {hints}{clip_lyrics}"
    )


def plan_clip_prompts(
    clips: list,
    style_prompt: str,
    lyrics: Optional[list] = None,
    bpm: float = 120.0,
    max_new_tokens: int = 150,
) -> list:
    """Generate per-clip video prompts based on audio analysis and style.

    Processes clips in small batches so the small LLM can produce varied output.

    Args:
        clips: List of dicts with keys: start, end, section_label, energy, suggested_prompt_hint
        style_prompt: Overall style/concept for the video
        lyrics: Optional list of dicts with keys: start, end, text
        bpm: Song BPM
        max_new_tokens: Max tokens per clip prompt

    Returns:
        List of prompt strings, one per clip
    """
    if not clips:
        return []

    import re

    BATCH_SIZE = 100  # Process all clips at once so the LLM sees the full song arc
    all_prompts = []

    # Camera/action variety pool to inject into prompts
    camera_angles = [
        "low angle looking up", "overhead bird's eye view", "close-up detail shot",
        "wide establishing shot", "Dutch angle tilted frame", "tracking shot following action",
        "slow dolly push-in", "handheld shaky cam energy", "profile silhouette shot",
        "over-the-shoulder perspective", "sweeping crane shot", "ground-level shot",
    ]

    system_prompt = (
        "You are an expert music video director. Write vivid, UNIQUE video generation prompts. "
        "Each prompt must describe a DIFFERENT visual scene with specific camera angle, lighting, "
        "action, and composition. NEVER repeat the same description twice. "
        "Vary camera movements, subject framing, and visual mood between clips. "
        "Output numbered prompts like '1. prompt text'. Keep each under 40 words. "
        "Output ONLY the numbered prompts, nothing else."
    )

    for batch_start in range(0, len(clips), BATCH_SIZE):
        batch = clips[batch_start:batch_start + BATCH_SIZE]
        batch_size = len(batch)

        clip_descriptions = []
        for j, clip in enumerate(batch):
            global_idx = batch_start + j
            desc = _build_clip_description(clip, global_idx, lyrics)
            # Suggest a specific camera angle to encourage variety
            angle = camera_angles[global_idx % len(camera_angles)]
            desc += f"\n   Suggested camera: {angle}"
            clip_descriptions.append(desc)

        clips_text = "\n".join(clip_descriptions)

        user_prompt = (
            f"Music video concept: {style_prompt}\n"
            f"Song tempo: {bpm:.0f} BPM\n\n"
            f"Write {batch_size} UNIQUE video prompts for these clips:\n\n"
            f"{clips_text}\n\n"
            f"Each prompt must be visually DIFFERENT. Use the suggested camera angles. Go:"
        )

        tokens_for_batch = max(max_new_tokens, batch_size * 80 + 1024)

        raw = generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=tokens_for_batch,
            temperature=0.8,
        )

        # Parse numbered lines
        batch_prompts = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and len(cleaned) > 10:
                batch_prompts.append(cleaned)

        # Pad short batches with style-based fallbacks
        while len(batch_prompts) < batch_size:
            idx = batch_start + len(batch_prompts)
            clip = clips[idx] if idx < len(clips) else {}
            section = clip.get("section_label", "verse")
            angle = camera_angles[idx % len(camera_angles)]
            batch_prompts.append(
                f"{style_prompt}, {section} section, {angle}, cinematic lighting"
            )
        batch_prompts = batch_prompts[:batch_size]

        all_prompts.extend(batch_prompts)
        print(f"[LLM] Planned prompts for clips {batch_start + 1}-{batch_start + batch_size}")

    return all_prompts[:len(clips)]


# Fixed angle categories for start image generation
ANGLE_CATEGORIES = [
    ("wide_establishing", "wide establishing shot, full body visible, environment and background prominent"),
    ("medium_subject", "medium shot, waist-up framing, subject centered, moderate background detail"),
    ("close_up", "close-up shot, face and upper body, shallow depth of field, intimate framing"),
    ("dynamic_angle", "dramatic low angle or Dutch tilt, dynamic perspective, bold composition"),
]


def plan_angle_prompts(
    style_prompt: str,
    num_angles: int = 4,
) -> list:
    """Generate image-edit prompts for camera angle variations of a reference photo.

    Uses the LLM to refine each angle category into a short prompt that
    incorporates the user's style description.

    Args:
        style_prompt: Overall visual style/concept from the user
        num_angles: Number of angle variations (default 4)

    Returns:
        List of image-edit prompt strings, one per angle
    """
    angles = ANGLE_CATEGORIES[:num_angles]

    angle_list = "\n".join(
        f"{i + 1}. {name}: {desc}"
        for i, (name, desc) in enumerate(angles)
    )

    system_prompt = (
        "You are a photography director. Write short image-edit prompts that describe "
        "how to reframe a reference photo into different camera angles. "
        "Each prompt should be under 30 words and describe the framing, angle, and mood. "
        "Incorporate the user's style into each prompt. "
        "Output numbered prompts like '1. prompt text'. Output ONLY the numbered prompts."
    )

    user_prompt = (
        f"Visual style: {style_prompt}\n\n"
        f"Write {len(angles)} image-edit prompts to create these camera angle variations "
        f"of a reference photo:\n\n{angle_list}\n\n"
        f"Each prompt should describe the camera angle and incorporate the visual style. Go:"
    )

    raw = generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=len(angles) * 60,
        temperature=0.7,
    )

    import re
    prompts = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
        if cleaned and len(cleaned) > 10:
            prompts.append(cleaned)

    # Pad with descriptive fallbacks if LLM output was too short
    while len(prompts) < len(angles):
        idx = len(prompts)
        name, desc = angles[idx]
        prompts.append(f"{style_prompt}, {desc}")
    prompts = prompts[:len(angles)]

    print(f"[LLM] Planned {len(prompts)} angle prompts")
    return prompts


# ---------------------------------------------------------------------------
# Song section classification via LLM
# ---------------------------------------------------------------------------

_VALID_SECTION_LABELS = {"intro", "verse", "chorus", "bridge", "outro", "instrumental"}

# Label normalization: maps common LLM output keywords to valid labels.
# Checked in order — "pre-chorus"/"pre chorus" must match before "chorus".
_LABEL_MAP = [
    ("intro", "intro"),
    ("outro", "outro"),
    ("pre-chorus", "bridge"),
    ("pre chorus", "bridge"),
    ("bridge", "bridge"),
    ("hook", "chorus"),
    ("chorus", "chorus"),
    ("verse", "verse"),
    ("instrumental", "instrumental"),
]


def _format_transcript(lyrics: list) -> str:
    """Format lyrics as a timestamped list with repetition and speaker markers.

    Detects lines that appear multiple times in the song (strong chorus signal)
    and includes speaker tags from diarization when available.
    """
    import re as _re

    # Collect texts with timestamps and speaker
    entries = []
    for lyr in lyrics:
        text = lyr.get("text", "").strip()
        if not text:
            continue
        start = int(lyr.get("start", 0))
        speaker = lyr.get("speaker")
        entries.append((start, text, speaker))

    if not entries:
        return ""

    # Count normalized occurrences to find repeated lyrics
    def _normalize(t: str) -> str:
        return _re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    norm_counts: dict = {}
    for _, text, _ in entries:
        key = _normalize(text)
        if key:
            norm_counts[key] = norm_counts.get(key, 0) + 1

    # Build formatted lines with speaker tag, repetition marker
    lines = []
    for start, text, speaker in entries:
        m, s = divmod(start, 60)
        key = _normalize(text)
        tags = []
        if speaker:
            tags.append(f"[{speaker}]")
        if norm_counts.get(key, 0) >= 2:
            tags.append("[REPEATS]")
        suffix = "  " + " ".join(tags) if tags else ""
        lines.append(f"{m}:{s:02d}  {text}{suffix}")
    return "\n".join(lines)


def _deloop_sections(llm_sections: list) -> list:
    """Truncate if the LLM falls into a repeating pattern (length 2-4).

    Requires 3+ repetitions — 2x is normal in real songs (e.g.
    Verse→Chorus→Verse→Chorus or Pre-Chorus→Chorus→Bridge→Chorus).
    """
    if len(llm_sections) <= 6:
        return llm_sections

    labels = [s["label"] for s in llm_sections]
    for pattern_len in range(2, 5):
        for start_idx in range(3, len(labels) - pattern_len * 3 + 1):
            pattern = labels[start_idx:start_idx + pattern_len]
            repeats = 1
            pos = start_idx + pattern_len
            while pos + pattern_len <= len(labels):
                if labels[pos:pos + pattern_len] == pattern:
                    repeats += 1
                    pos += pattern_len
                else:
                    break
            if repeats >= 3:
                print(f"[LLM] Detected {repeats}x repeating pattern {pattern} "
                      f"at section {start_idx + 1}, truncating")
                return llm_sections[:start_idx]
    return llm_sections


def _normalize_label(raw: str) -> str:
    """Map a raw LLM section label to a valid label."""
    raw = raw.lower()
    for keyword, label in _LABEL_MAP:
        if keyword in raw:
            return label
    return "verse"


# ---------------------------------------------------------------------------
# Speaker diarization helpers for section refinement
# ---------------------------------------------------------------------------

def _build_speaker_runs(lyrics):
    """Find sustained speaker blocks from diarized lyrics.

    Groups consecutive lines by speaker.  Single-line interjections
    (ad-libs, backing vocals) are absorbed into the surrounding block
    to avoid spurious section splits.

    Returns list of dicts [{speaker, start, end, lines}, ...] sorted by time.
    """
    sorted_lyrs = sorted(
        [l for l in lyrics if l.get("speaker") and l.get("text", "").strip()],
        key=lambda l: float(l.get("start", 0)),
    )
    if not sorted_lyrs:
        return []

    # Phase 1: raw consecutive runs
    runs = []
    for lyr in sorted_lyrs:
        spk = lyr["speaker"]
        t = float(lyr.get("start", 0))
        if runs and runs[-1]["speaker"] == spk:
            runs[-1]["end"] = t
            runs[-1]["lines"] += 1
        else:
            runs.append({"speaker": spk, "start": t, "end": t, "lines": 1})

    if len(runs) <= 1:
        return runs

    # Phase 2: absorb single-line interjections into the previous block
    merged = [runs[0]]
    for r in runs[1:]:
        if r["lines"] <= 1:
            merged[-1]["end"] = r["end"]
            merged[-1]["lines"] += r["lines"]
        else:
            merged.append(r)

    # Phase 3: merge consecutive same-speaker blocks created by absorption
    final = [merged[0]]
    for r in merged[1:]:
        if r["speaker"] == final[-1]["speaker"]:
            final[-1]["end"] = r["end"]
            final[-1]["lines"] += r["lines"]
        else:
            final.append(r)

    return final


def _identify_chorus_speaker(lyrics):
    """Identify which speaker tends to sing repeated/hook lines.

    Returns (speaker_id, repeat_ratio) or (None, 0) if no clear signal.
    The speaker with the highest fraction of globally-repeated lines
    is the likely chorus/hook performer.
    """
    import re as _re

    def _norm(t):
        return _re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    # Count all normalised lines globally
    global_counts: dict = {}
    for lyr in lyrics:
        text = lyr.get("text", "").strip()
        if not text:
            continue
        key = _norm(text)
        if len(key) > 5:
            global_counts[key] = global_counts.get(key, 0) + 1

    # Per-speaker stats
    speaker_stats: dict = {}
    for lyr in lyrics:
        spk = lyr.get("speaker")
        text = lyr.get("text", "").strip()
        if not spk or not text:
            continue
        key = _norm(text)
        if spk not in speaker_stats:
            speaker_stats[spk] = {"total": 0, "repeated": 0}
        speaker_stats[spk]["total"] += 1
        if len(key) > 5 and global_counts.get(key, 0) >= 2:
            speaker_stats[spk]["repeated"] += 1

    if len(speaker_stats) < 2:
        return None, 0

    for s in speaker_stats.values():
        s["ratio"] = s["repeated"] / s["total"] if s["total"] > 0 else 0

    best = max(speaker_stats, key=lambda k: speaker_stats[k]["ratio"])

    # Need a meaningful repeat ratio (>= 10%) to call someone the hook singer
    if speaker_stats[best]["ratio"] < 0.10:
        return None, 0

    # Must be noticeably higher than the next speaker
    ratios = sorted(speaker_stats.values(), key=lambda s: s["ratio"], reverse=True)
    if len(ratios) >= 2 and ratios[0]["ratio"] - ratios[1]["ratio"] < 0.05:
        return None, 0

    return best, speaker_stats[best]["ratio"]


def _refine_sections_with_speakers(sections, lyrics, duration):
    """Split long verse/bridge sections at sustained speaker-change points.

    After the initial repetition-based structure is built, this function
    looks for sections longer than 20 s that contain multiple speaker
    blocks.  When found it subdivides them — the speaker with the most
    repeated lines is labelled as singing hooks/choruses.
    """
    chorus_spk, chorus_ratio = _identify_chorus_speaker(lyrics)

    runs = _build_speaker_runs(lyrics)
    unique_speakers = set(r["speaker"] for r in runs)

    if len(unique_speakers) < 2:
        return sections

    if chorus_spk:
        print(f"[Sections] Hook speaker: {chorus_spk} "
              f"({chorus_ratio:.0%} repeat ratio)")

    refined = []
    verse_num = 0

    for i, sec in enumerate(sections):
        sec_start = sec["start"]
        sec_end = sections[i + 1]["start"] if i + 1 < len(sections) else duration
        sec_label = sec["label"]

        # Only split long verse / bridge sections
        if sec_label not in ("verse", "bridge") or (sec_end - sec_start) < 20:
            if sec_label == "verse":
                verse_num += 1
                sec = dict(sec)
                sec["display_label"] = f"Verse {verse_num}"
            refined.append(sec)
            continue

        # Speaker runs overlapping this section (>= 2 lines each)
        sec_runs = [
            r for r in runs
            if r["end"] > sec_start - 1 and r["start"] < sec_end
            and r["lines"] >= 2
        ]

        if len(sec_runs) < 2:
            if sec_label == "verse":
                verse_num += 1
                sec = dict(sec)
                sec["display_label"] = f"Verse {verse_num}"
            refined.append(sec)
            continue

        print(f"[Sections] Splitting {sec.get('display_label', sec_label)} "
              f"({sec_end - sec_start:.0f}s) into {len(sec_runs)} sub-sections")

        for j, run in enumerate(sec_runs):
            # First sub-section keeps the original section start time
            sub_start = sec_start if j == 0 else int(run["start"])
            sub_end = (
                int(sec_runs[j + 1]["start"]) if j + 1 < len(sec_runs)
                else sec_end
            )

            if sub_end - sub_start < 5:
                continue  # too short — replace_sections_with_structure merges < 5 s

            if chorus_spk and run["speaker"] == chorus_spk:
                refined.append({
                    "label": "chorus",
                    "display_label": "Hook",
                    "start": sub_start,
                })
            else:
                verse_num += 1
                refined.append({
                    "label": "verse",
                    "display_label": f"Verse {verse_num}",
                    "start": sub_start,
                })

    return refined if len(refined) >= 3 else sections


def _classify_by_speakers(lyrics, duration):
    """Build song structure purely from speaker diarization.

    Fallback when repetition detection finds no chorus pattern but
    we have 2+ speakers.  Creates section boundaries at sustained
    speaker changes and uses repetition ratio to label hooks.
    """
    runs = _build_speaker_runs(lyrics)
    unique_speakers = set(r["speaker"] for r in runs)

    if len(unique_speakers) < 2 or len(runs) < 3:
        return []

    chorus_spk, _ = _identify_chorus_speaker(lyrics)

    sections = []
    verse_num = 0

    # Detect intro (instrumental gap before first lyric)
    first_lyric_time = None
    for lyr in sorted(lyrics, key=lambda l: float(l.get("start", 0))):
        if lyr.get("text", "").strip():
            first_lyric_time = float(lyr.get("start", 0))
            break
    if first_lyric_time and first_lyric_time > 3:
        sections.append({"label": "intro", "display_label": "Intro", "start": 0})

    last_lyric_time = max(
        float(l.get("start", 0)) for l in lyrics if l.get("text", "").strip()
    )

    for i, run in enumerate(runs):
        run_end = runs[i + 1]["start"] if i + 1 < len(runs) else (last_lyric_time + 2)
        if run_end - run["start"] < 5:
            continue

        if chorus_spk and run["speaker"] == chorus_spk:
            sections.append({
                "label": "chorus",
                "display_label": "Hook",
                "start": int(run["start"]),
            })
        else:
            verse_num += 1
            sections.append({
                "label": "verse",
                "display_label": f"Verse {verse_num}",
                "start": int(run["start"]),
            })

    # Detect outro
    if duration - last_lyric_time > 10:
        sections.append({
            "label": "outro",
            "display_label": "Outro",
            "start": int(last_lyric_time) + 2,
        })

    if sections:
        print(f"[Classification] Speaker-based structure ({len(sections)} sections):")
        for s in sections:
            m, sec = divmod(s["start"], 60)
            print(f"  [{s['display_label']}] {int(m)}:{sec:02d}")

    return sections if len(sections) >= 3 else []


def _classify_by_repetition(lyrics: list, duration: float) -> list:
    """Build song structure from lyric repetition patterns.

    Finds chorus sections by detecting clusters of repeated lyric lines.
    More reliable than the 2B LLM for verse/chorus identification.

    Returns list of dicts [{label, display_label, start}, ...] or empty list.
    """
    import re as _re

    def _norm(t: str) -> str:
        return _re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    entries = []
    counts: dict = {}
    for lyr in lyrics:
        text = lyr.get("text", "").strip()
        if not text:
            continue
        key = _norm(text)
        start = float(lyr.get("start", 0))
        if len(key) > 5:
            counts[key] = counts.get(key, 0) + 1
        entries.append({"start": start, "key": key})

    if len(entries) < 6:
        return []

    n = len(entries)

    # Mark each line: repeated if substantial and appears 2+ times
    for e in entries:
        e["rep"] = len(e["key"]) > 5 and counts.get(e["key"], 0) >= 2

    # Chorus mask: 5-line sliding window, chorus zone if >= 3 repeats
    chorus_mask = []
    for i in range(n):
        window = entries[max(0, i - 2):min(n, i + 3)]
        chorus_mask.append(sum(1 for w in window if w["rep"]) >= 3)

    # Extract contiguous chorus regions
    regions = []
    rstart = None
    for i, c in enumerate(chorus_mask):
        if c and rstart is None:
            rstart = i
        elif not c and rstart is not None:
            regions.append((rstart, i - 1))
            rstart = None
    if rstart is not None:
        regions.append((rstart, n - 1))

    if not regions:
        return []

    print(f"[Repetition] Detected {len(regions)} chorus region(s):")
    for cs, ce in regions:
        print(f"  {entries[cs]['start']:.0f}s - {entries[ce]['start']:.0f}s")

    # --- Build section list ---
    sections = []
    verse_num = 0

    first_chorus_time = entries[regions[0][0]]["start"]

    # Intro: look for a gap before substantial lyrics start
    first_substantial = next((e for e in entries if len(e["key"]) > 10), None)
    intro_end = first_substantial["start"] if first_substantial else 0

    if intro_end > 3:
        sections.append({"label": "intro", "display_label": "Intro", "start": 0})

    # Verse before first chorus
    if first_chorus_time > (intro_end + 5):
        verse_num += 1
        sections.append({
            "label": "verse", "display_label": f"Verse {verse_num}",
            "start": int(intro_end),
        })

    # Choruses and inter-chorus verses
    for i, (cs, ce) in enumerate(regions):
        sections.append({
            "label": "chorus", "display_label": "Chorus",
            "start": int(entries[cs]["start"]),
        })

        if i + 1 < len(regions):
            gap_start = ce + 1
            gap_end = regions[i + 1][0]
            if gap_end - gap_start >= 3:
                verse_num += 1
                sections.append({
                    "label": "verse", "display_label": f"Verse {verse_num}",
                    "start": int(entries[gap_start]["start"]),
                })
        else:
            # After last chorus
            if ce + 1 < n:
                post_lines = n - (ce + 1)
                post_time = duration - entries[ce + 1]["start"]
                if post_lines >= 5 and post_time > 15:
                    verse_num += 1
                    sections.append({
                        "label": "verse", "display_label": f"Verse {verse_num}",
                        "start": int(entries[ce + 1]["start"]),
                    })
                elif post_time > 5:
                    sections.append({
                        "label": "outro", "display_label": "Outro",
                        "start": int(entries[ce + 1]["start"]),
                    })

    # Refine with speaker diarization: split long sections at speaker changes
    if len(sections) >= 3:
        sections = _refine_sections_with_speakers(sections, lyrics, duration)

    return sections if len(sections) >= 3 else []


def _map_labels_to_sections(sections: list, structure: list) -> list:
    """Map each audio section to the structure entry containing its midpoint."""
    labels = []
    for sec in sections:
        mid = (sec.get("start", 0) + sec.get("end", 0)) / 2
        best_label = structure[0]["label"]
        for s in structure:
            if s["start"] <= mid:
                best_label = s["label"]
            else:
                break
        labels.append(best_label)
    return labels


def classify_song_sections(
    sections: list,
    lyrics: list,
    duration: float,
) -> dict:
    """Classify song sections using repetition detection, with LLM fallback.

    Primary: programmatic detection of repeated lyric clusters (chorus).
    Fallback: LLM-based classification if no repetition pattern is found.

    Returns:
        Dict with:
          labels: List of label strings, one per audio section
          song_structure: List of dicts [{label, display_label, start}, ...]
    """
    empty_result = {"labels": [], "song_structure": []}
    if not sections:
        return empty_result

    fallback_labels = [s.get("label", "verse") for s in sections]

    if not lyrics:
        return {"labels": fallback_labels, "song_structure": []}

    transcript_text = _format_transcript(lyrics)
    if not transcript_text:
        return {"labels": fallback_labels, "song_structure": []}

    print(f"[Classification] Transcript ({len(transcript_text.splitlines())} lines):\n{transcript_text}")

    # --- Primary: repetition-based classification ---
    rep_structure = _classify_by_repetition(lyrics, duration)
    if rep_structure:
        labels = _map_labels_to_sections(sections, rep_structure)
        print(f"[Classification] Repetition-based structure:")
        for s in rep_structure:
            m, sec = divmod(s["start"], 60)
            print(f"  [{s['display_label']}] {m}:{sec:02d}")
        print(f"[Classification] Mapped to {len(sections)} sections: {labels}")
        return {"labels": labels, "song_structure": rep_structure}

    # --- Secondary: speaker diarization-based classification ---
    spk_structure = _classify_by_speakers(lyrics, duration)
    if spk_structure:
        labels = _map_labels_to_sections(sections, spk_structure)
        print(f"[Classification] Mapped to {len(sections)} sections: {labels}")
        return {"labels": labels, "song_structure": spk_structure}

    # --- Fallback: LLM-based classification ---
    import re

    # Strip [REPEATS] markers — they overwhelm the 2B model
    clean_transcript = transcript_text.replace("  [REPEATS]", "")

    print(f"[LLM] No repetition or speaker pattern found, using LLM classification")

    system_prompt = (
        "You are a music analyst. Given a timestamped song transcript, "
        "identify the song sections.\n"
        "Valid sections: [Intro], [Verse 1], [Verse 2], [Pre-Chorus], "
        "[Chorus], [Bridge], [Outro], [Instrumental].\n"
        "Rules:\n"
        "- Chorus: repeated hook/refrain\n"
        "- Verse: narrative lyrics, different each time\n"
        "- Bridge: contrasting section, usually appears once\n"
        "- Intro/Outro: beginning/end\n"
        "- Instrumental: no lyrics\n\n"
        "Output ONLY section labels with start times:\n"
        "[Intro] 0:00\n[Verse 1] 0:10\n[Chorus] 0:55\n"
        "No lyrics, no explanations."
    )

    user_prompt = (
        f"Song transcript ({duration:.0f}s total):\n\n"
        f"{clean_transcript}\n\n"
        f"Song sections:"
    )

    raw = generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=400,
        temperature=0.2,
    )

    print(f"[LLM] Raw classification output:\n{raw}")

    llm_sections = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[([^\]]+)\]\s*(\d+):(\d+)", line)
        if match:
            label = _normalize_label(match.group(1))
            start_time = int(match.group(2)) * 60 + int(match.group(3))
            llm_sections.append({
                "label": label,
                "display_label": match.group(1).strip(),
                "start": start_time,
            })

    if not llm_sections:
        print("[LLM] Could not parse classification output, using heuristic")
        return {"labels": fallback_labels, "song_structure": []}

    llm_sections = _deloop_sections(llm_sections)

    labels = _map_labels_to_sections(sections, llm_sections)

    print(f"[LLM] Structure: "
          + ", ".join(f"[{s['display_label']}] {s['start']//60}:{s['start']%60:02d}" for s in llm_sections))
    print(f"[LLM] Mapped to {len(sections)} sections: {labels}")

    song_structure = [
        {"label": s["label"], "display_label": s["display_label"], "start": s["start"]}
        for s in llm_sections
    ]

    return {"labels": labels[:len(sections)], "song_structure": song_structure}


# ---------------------------------------------------------------------------
# Unified per-clip video + image prompt planning
# ---------------------------------------------------------------------------

def _parse_performer_map(scene_description: str) -> dict:
    """Extract section→performer mapping from scene description.

    Looks for patterns like "man raps the verses", "woman sings chorus",
    "he performs the bridge", etc. Returns e.g. {"verse": "man", "chorus": "woman"}.
    """
    import re
    mapping = {}
    scene_lower = scene_description.lower()

    # Patterns: "<person> <verb> <section>" or "<section> by <person>"
    _PERSONS = r"(man|woman|guy|girl|boy|he|she|male|female|rapper|singer)"
    _SECTIONS = r"(verse|chorus|bridge|intro|outro|hook)"
    _VERBS = r"(?:raps?|sings?|signs?|performs?|does|delivers?|handles?)"

    # "man raps the verses"
    for m in re.finditer(
        _PERSONS + r"\s+" + _VERBS + r"\s+(?:the\s+)?" + _SECTIONS + r"s?",
        scene_lower,
    ):
        person, section = m.group(1), m.group(2)
        mapping[section] = person

    # "chorus by the woman"
    for m in re.finditer(
        _SECTIONS + r"s?\s+(?:are |is )?(?:by |from )(?:the\s+)?" + _PERSONS,
        scene_lower,
    ):
        section, person = m.group(1), m.group(2)
        mapping[section] = person

    # Normalize pronouns to gendered nouns
    _PRONOUN_MAP = {
        "he": "man", "she": "woman", "guy": "man", "girl": "woman",
        "boy": "man", "male": "man", "female": "woman",
        "rapper": "man", "singer": "woman",
    }
    return {k: _PRONOUN_MAP.get(v, v) for k, v in mapping.items()}


def _dominant_speaker(lyrics: list, start: float, end: float) -> Optional[str]:
    """Find the speaker with the most lines in a time range."""
    if not lyrics:
        return None
    counts: dict = {}
    for l in lyrics:
        if l["start"] < end and l["end"] > start and l.get("speaker"):
            s = l["speaker"]
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _build_clip_description_v2(
    clip: dict, index: int, lyrics: Optional[list] = None,
    performer_map: Optional[dict] = None,
    speaker_names: Optional[dict] = None,
    speaker_roles: Optional[dict] = None,
) -> str:
    """Build a clip description with lyrics context and performer info.

    Uses diarization speaker tags (if available) to tell the LLM exactly
    who to show. Falls back to section-based performer_map otherwise.
    Camera/shot choices are left to the LLM's creative judgment.
    """
    start = clip.get("start", 0)
    end = clip.get("end", 0)
    section = clip.get("section_label", "verse")
    beat_count = clip.get("beat_count", 16)

    # Gather overlapping lyrics
    lyrics_snippet = ""
    if lyrics:
        matching = [l["text"] for l in lyrics if l["start"] < end and l["end"] > start]
        if matching:
            lyrics_snippet = " ".join(matching)

    # Identify who is performing in this clip
    speaker = _dominant_speaker(lyrics, start, end) if lyrics else None
    role = ""
    if speaker and speaker_roles:
        role = speaker_roles.get(speaker, "")

    vocal_info = f'lyrics: "{lyrics_snippet}"' if lyrics_snippet else "instrumental"

    # Performer hint: prefer diarization speaker, fall back to section performer_map
    performer_hint = ""
    if speaker and speaker_names and speaker in speaker_names:
        name = speaker_names[speaker]
        role_suffix = f" ({role})" if role else ""
        performer_hint = f" Performer: the {name}{role_suffix}."
    elif performer_map and section in performer_map:
        performer_hint = f" Performer: the {performer_map[section]}."

    return (
        f"Clip {index + 1}: {section}, {beat_count} beats, {vocal_info}.{performer_hint}"
    )


def _build_fallback_prompt(
    clip: dict, index: int, section: str,
    lyrics: Optional[list] = None,
    speaker_names: Optional[dict] = None,
    prompt_type: str = "image",
) -> str:
    """Build a clip-specific fallback prompt when LLM parsing fails."""
    start = clip.get("start", 0)
    end = clip.get("end", 0)

    # Get performer name for this clip
    performer = ""
    if lyrics and speaker_names:
        speaker = _dominant_speaker(lyrics, start, end)
        if speaker and speaker in speaker_names:
            performer = speaker_names[speaker]

    # Get a lyrics snippet for context
    snippet = ""
    if lyrics:
        matching = [l["text"] for l in lyrics if l["start"] < end and l["end"] > start]
        if matching:
            snippet = " ".join(matching[:2])

    if prompt_type == "video":
        parts = []
        if performer:
            parts.append(f"{performer} performing")
        parts.append(f"{section} section")
        if snippet:
            parts.append(f"matching lyrics about: {snippet[:60]}")
        return ", ".join(parts)
    else:
        parts = []
        if performer:
            parts.append(performer)
        parts.append(f"{section} scene")
        if snippet:
            parts.append(f"inspired by: {snippet[:60]}")
        return ", ".join(parts)


def plan_clip_prompts_and_images(
    clips: list,
    scene_description: str,
    lyrics: Optional[list] = None,
    bpm: float = 120.0,
    max_new_tokens: int = 512,
    reference_image_path: Optional[str] = None,
    speaker_mappings: Optional[dict] = None,
    prompt_type: str = "both",
    existing_image_prompts: Optional[list] = None,
) -> list:
    """Generate per-clip prompts.  Supports three modes via *prompt_type*:

    - ``"image"`` — generate only starting-frame (image) descriptions.
    - ``"video"`` — generate only video-motion prompts (may use *existing_image_prompts* as context).
    - ``"both"``  — legacy mode, generates V + I in one pass.

    Returns:
        ``prompt_type="image"``  → ``[{"image_prompt": str}, ...]``
        ``prompt_type="video"``  → ``[{"video_prompt": str}, ...]``
        ``prompt_type="both"``   → ``[{"video_prompt": str, "image_prompt": str}, ...]``
    """
    if not clips:
        return []

    import re

    BATCH_SIZE = 100  # Process all clips at once so the LLM sees the full song arc
    all_plans: list = []
    performer_map = _parse_performer_map(scene_description)
    if performer_map:
        print(f"[LLM] Performer map from scene: {performer_map}")

    # Build speaker→name and speaker→role mappings.
    # Prefer explicit UI mappings; fall back to auto-detection via repeat ratio.
    speaker_names: dict = {}
    speaker_roles: dict = {}

    if speaker_mappings:
        # User provided explicit mappings from the UI
        for spk_id, info in speaker_mappings.items():
            if info.get("name"):
                speaker_names[spk_id] = info["name"]
            if info.get("role"):
                speaker_roles[spk_id] = info["role"]
        print(f"[LLM] Speaker mappings from UI: {speaker_names} roles={speaker_roles}")

    elif lyrics and any(l.get("speaker") for l in lyrics):
        # Auto-detect: identify chorus vs verse speaker by repeat ratio
        import re as _re
        def _norm(t): return _re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
        norm_counts: dict = {}
        for l in lyrics:
            key = _norm(l.get("text", ""))
            if key and len(key) > 5:
                norm_counts[key] = norm_counts.get(key, 0) + 1

        speaker_repeat: dict = {}  # speaker → [repeating_lines, total_lines]
        for l in lyrics:
            s = l.get("speaker")
            if not s:
                continue
            key = _norm(l.get("text", ""))
            if len(key) <= 5:
                continue
            if s not in speaker_repeat:
                speaker_repeat[s] = [0, 0]
            speaker_repeat[s][1] += 1
            if norm_counts.get(key, 0) >= 2:
                speaker_repeat[s][0] += 1

        if len(speaker_repeat) >= 2:
            ranked = sorted(speaker_repeat.items(),
                            key=lambda x: x[1][0] / max(x[1][1], 1), reverse=True)
            chorus_speaker = ranked[0][0]
            verse_speaker = ranked[-1][0]

            chorus_name = performer_map.get("chorus", "woman")
            verse_name = performer_map.get("verse", "man")

            speaker_names[chorus_speaker] = chorus_name
            speaker_names[verse_speaker] = verse_name

            print(f"[LLM] Speaker mapping (auto): {chorus_speaker}→{chorus_name} (chorus), "
                  f"{verse_speaker}→{verse_name} (verse)")
        elif len(speaker_repeat) == 1:
            only_speaker = list(speaker_repeat.keys())[0]
            speaker_names[only_speaker] = performer_map.get("verse", "performer")
            print(f"[LLM] Single speaker: {only_speaker}")


    has_image = reference_image_path and os.path.isfile(reference_image_path)

    # ── Load LLM guides ──────────────────────────────────────────────
    guides_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_guides")
    def _load_guide(filename):
        p = os.path.join(guides_dir, filename)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"[LLM] Loaded guide: {filename} ({len(content)} chars)")
            return content
        return ""

    video_guide = _load_guide("LTX-2_PROMPTING_GUIDE_Embedded_Audio.MD")
    image_guide = _load_guide("QWEN IMAGE EDIT PROMPTING GUIDE.md")

    guide_sections = ""
    if video_guide and prompt_type in ("video", "both"):
        guide_sections += (
            "\n\nVIDEO PROMPTING GUIDE — follow this when writing video prompts:\n"
            "---\n"
            f"{video_guide}\n"
            "---\n"
        )
    # NOTE: The Qwen image edit guide is NOT included here — it was designed for
    # direct user prompting (single edits) and its examples ("Edit the provided image",
    # "Preserve identity") actively conflict with Director mode requirements.
    # The system prompt instructions above are sufficient for Director image prompts.

    # ── Build system prompt per prompt_type ──────────────────────────
    photo_line = (
        "You are given a REFERENCE PHOTO — use it to identify the people, "
        "their clothing, and the setting.\n"
    ) if has_image else ""

    char_rule = (
        "- NEVER use character names in image OR video prompts — neither the image "
        "editor nor the video model can identify people by name. Instead, describe "
        "each person by their VISIBLE appearance: clothing, hair, position in frame, "
        "based on what you visually see in the attached reference image. For example, "
        "write 'the woman in the white lab coat' instead of a character name.\n"
        "- Character names ARE only allowed in spoken dialogue in the video prompt.\n"
        "- Describe people using what you SEE in the photo + Scene description."
    ) if has_image else (
        "- NEVER use character names in image OR video prompts. Instead, describe each "
        "person by their visual appearance: clothing, hair color, position.\n"
        "- Character names ARE only allowed in spoken dialogue in the video prompt.\n"
        "- ONLY use characters and clothing described in the Scene."
    )

    # Shared rules for all music video modes
    shared_rules = (
        "- Use the Scene Concept as your PRIMARY guide for locations, outfits, "
        "props, and activities.\n"
        "- LOCATIONS ARE BINDING: if the Scene Concept names a specific location "
        "or setting, EVERY clip stays in that location unless the Scene Concept "
        "itself calls for a move. Do NOT invent new locations for visual variety — "
        "vary the camera angle, framing, distance, and lighting instead.\n"
        "- Do NOT add subjects, creatures, or objects the Scene Concept and "
        "reference photos don't contain. Any examples in these instructions "
        "show FORMAT only — never copy their content into prompts.\n"
        "- Use the lyrics to inspire mood and visual metaphors, NOT literal text.\n"
        "- If a clip says 'Performer:', that person must appear.\n"
        f"{char_rule}\n"
        "- Do NOT put lyrics or spoken words in prompts.\n"
        "- CRITICAL: Each image and video prompt is generated INDEPENDENTLY with "
        "NO context from other clips. The image generator does NOT know who any "
        "character is. Every prompt must be FULLY SELF-CONTAINED — "
        "re-describe each character's clothing, hair, and appearance, plus the "
        "setting, lighting, and atmosphere in EVERY prompt. Never assume the "
        "generator 'remembers' anything from prior clips.\n"
    )

    if prompt_type == "image":
        if has_image:
            system_prompt = (
                f"You are a creative music video scene designer.\n{photo_line}\n"
                "Each clip's image prompt EDITS the reference photo to create a visually "
                "distinct starting frame. Think like a cinematographer choosing each shot — "
                "vary camera angle, framing, or lighting when it serves the scene, but "
                "maintain continuity when that makes more sense.\n\n"
                "Focus your prompt on WHAT TO CHANGE — new camera angles, different framing, "
                "new settings, repositioned characters, lighting shifts. You do NOT need to "
                "re-describe things that stay the same as the reference photo.\n\n"
                "WHEN TO DESCRIBE CHARACTERS: When you change the setting to a new location, "
                "re-describe characters by clothing/appearance so the editor knows who to place. "
                "When the setting stays the same, you only need to describe characters whose "
                "position changes.\n\n"
                "GOOD: 'change to a dramatic close-up of the woman in the red dress, soft backlight, shallow depth of field'\n"
                "GOOD: 'change the setting to a neon-lit club, the woman in the red dress stands center stage under purple lights'\n"
                "GOOD: 'change to an over-the-shoulder shot from behind the man in the dark jacket, looking at the stage'\n"
                "BAD: 'Edit the provided image. Show Sarah dancing. Preserve character identity.' — uses name, meta-instructions\n"
                "BAD: re-describing the entire reference photo when nothing changed\n\n"
                "RULES:\n"
                f"{shared_rules}"
                "- Use a mix of shot types (close-ups, wide shots, over-shoulder, etc.) "
                "where appropriate — but continuity between consecutive clips is fine when it fits.\n"
                "- The PERFORMER IS the person/character in the reference photo. Anchor them "
                "explicitly in EVERY prompt ('the [descriptor] from the reference image') — "
                "describing them loosely as a new character makes the image model invent a "
                "different-looking one.\n"
                "- NO motion blur, speed lines, or long-exposure effects — the image is a sharp "
                "still frame; motion belongs to the video prompt.\n"
                "- Focus on WHAT TO CHANGE from the reference. Do not re-describe things that stay the same.\n"
                "- Do NOT start with 'Edit the provided image' — just describe the changes.\n"
                "- Do NOT use preservation meta-language ('preserve', 'maintain', 'keep unchanged').\n"
                f"{guide_sections}\n"
                "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
            )
        else:
            system_prompt = (
                "You are a creative music video scene designer.\n\n"
                "For each clip, write a SCENE DESCRIPTION — describe the starting frame: "
                "where the characters are, what they are doing, the setting, lighting, "
                "mood, and composition. Be vivid and specific.\n\n"
                "RULES:\n"
                f"{shared_rules}"
                "- Vary shots creatively — mix close-ups, wide shots, different angles.\n"
                "- Consecutive clips should feel visually DIFFERENT.\n"
                "- Instrumental clips can use establishing shots, environment details, "
                "or abstract visuals.\n"
                f"{guide_sections}\n"
                "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
            )
    elif prompt_type == "video":
        system_prompt = (
            "You write video motion prompts for music video clips.\n"
            "Each clip is a SINGLE CONTINUOUS SHOT — one unbroken camera take with "
            "no cuts or edits. Write a flowing paragraph describing WHO is on screen, "
            "WHAT they are doing, the SETTING with lighting and atmosphere, and how "
            "the CAMERA moves during this one take.\n\n"
            "RULES:\n"
            "- NEVER say 'montage', 'quick cuts', 'cut to', 'series of shots', or "
            "'multiple angles' — these are impossible in a single take.\n"
            f"{shared_rules}"
            "- Include specific camera movement (slow dolly in, tracking shot, "
            "orbit, pan, handheld follow).\n"
            "- Describe character action, body language, and energy matching the "
            "music's mood for that section.\n"
            "- CRITICAL: Each video prompt is generated INDEPENDENTLY — the video "
            "model has NO memory of previous clips and does NOT know who any character "
            "is. You MUST re-describe every character's clothing, hair, and appearance "
            "in EVERY video prompt, even if you described them in the previous prompt.\n"
            "- For performance clips: describe how the performer moves, gestures, "
            "and engages with the camera.\n"
            "- For instrumental/atmospheric clips: focus on environment, lighting "
            "shifts, textures, and cinematic camera moves.\n"
            f"{guide_sections}\n"
            "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
        )
    else:
        # Combined "both" mode — LLM has full creative control
        image_instruction = (
            "- I (image edit): EDITS the reference photo to create the FIRST FRAME BEFORE action begins. "
            "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
            "show the room without them. Focus on WHAT TO CHANGE from reference. "
            "Describe POSES as static states (standing, seated, leaning). "
            "No motion verbs (walking, running, reaching, heaving, turning). "
            "No motion blur, speed lines, or long-exposure effects — the frame is sharp. "
            "Anchor the performer as 'the [descriptor] from the reference image'. "
            "NEVER use names. Actions belong ONLY in V, never in I.\n"
        ) if has_image else (
            "- I (image): the FIRST FRAME BEFORE action begins — a frozen still photograph. "
            "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
            "the room is empty. Describe static poses only (standing, seated, leaning). "
            "No motion verbs (walking, running, reaching, heaving, turning). "
            "Characters (by appearance, not name), positions, setting, lighting, mood.\n"
        )
        system_prompt = (
            f"You are a music video director.\n{photo_line}\n"
            "The user gives you a SCENE CONCEPT describing the overall vibe, locations, "
            "outfits, and activities. Your job is to bring THIS CONCEPT to life across "
            "all the clips.\n\n"
            "PROMPT TYPES:\n"
            "- V (video): a SINGLE CONTINUOUS SHOT — this is where ALL motion, action, "
            "dancing, gestures, and camera movement happen. Describe who is on screen (by "
            "clothing/appearance — re-describe in EVERY prompt), what they DO, setting "
            "with lighting, and camera movement. NEVER say 'montage', 'quick cuts', "
            "'cut to', or 'series of shots'. Character names ARE only allowed in spoken dialogue.\n"
            f"{image_instruction}"
            "- I prompts describe ONE SINGLE STATIC IMAGE — the starting frame that the "
            "video animates FROM. No motion, no actions, no dancing. Just a frozen "
            "establishing shot: where everyone is, what the setting looks like.\n\n"
            "CREATIVE DIRECTION:\n"
            f"{shared_rules}"
            "- YOU choose camera angles, movements, and shot composition.\n"
            "- Vary shots creatively — mix close-ups, wide shots, tracking shots, etc.\n"
            "- Consecutive clips should feel visually DIFFERENT.\n"
            "- Instrumental clips can use establishing shots, environment details, "
            "or abstract visuals.\n"
            "- Do NOT start I prompts with 'Edit the provided image'.\n"
            "- Do NOT use meta-language ('preserve', 'maintain', 'keep unchanged') in I prompts.\n"
            "- DO use action verbs in I prompts: 'change', 'make', 'move to', 'add'.\n"
            f"{guide_sections}\n"
            "Format: '1V. prompt' then '1I. prompt'. Output ONLY numbered prompts."
        )

    # Send the reference image with every batch so the LLM can see who's in the scene
    batch_images = [reference_image_path] if has_image else None

    print(f"[LLM] Planning prompts: prompt_type={prompt_type}, {len(clips)} clips")

    for batch_start in range(0, len(clips), BATCH_SIZE):
        batch = clips[batch_start:batch_start + BATCH_SIZE]
        batch_size = len(batch)

        clip_descriptions = []
        for j, clip in enumerate(batch):
            global_idx = batch_start + j
            desc = _build_clip_description_v2(clip, global_idx, lyrics, performer_map, speaker_names, speaker_roles)
            clip_descriptions.append(desc)

        clips_text = "\n".join(clip_descriptions)

        # Build user prompt with context appropriate to the prompt type
        if prompt_type == "image":
            user_prompt = (
                f"Scene Concept: {scene_description}\n\n"
                f"Clips:\n{clips_text}\n\n"
                f"Design scenes based on the Scene Concept above. "
                f"For each clip, write a detailed, self-contained image prompt "
                f"describing the setting, characters, lighting, and composition. "
                f"Output format: 1. prompt, 2. prompt, etc."
            )
            tokens_for_batch = max(max_new_tokens, batch_size * 200 + 512)
        elif prompt_type == "video":
            # Include existing image prompts as context so video prompts are consistent
            context_lines = ""
            if existing_image_prompts:
                img_context = []
                for j in range(batch_size):
                    global_idx = batch_start + j
                    if global_idx < len(existing_image_prompts):
                        ip = existing_image_prompts[global_idx]
                        img_context.append(f"Clip {global_idx + 1} starts as: {ip}")
                if img_context:
                    context_lines = "\nStarting frames:\n" + "\n".join(img_context) + "\n"

            user_prompt = (
                f"Scene Concept: {scene_description}\n{context_lines}\n"
                f"Clips:\n{clips_text}\n\n"
                f"Write a flowing video prompt paragraph for each clip. "
                f"Each prompt should describe the full scene: setting, lighting, "
                f"character action, and camera movement. "
                f"Output format: 1. prompt, 2. prompt, etc."
            )
            tokens_for_batch = max(max_new_tokens, batch_size * 200 + 1024)
        else:
            user_prompt = (
                f"Scene Concept: {scene_description}\n\n"
                f"Clips:\n{clips_text}\n\n"
                f"Direct a music video based on the Scene Concept above. "
                f"Write detailed V and I prompts for each clip. "
                f"Output format: 1V. ... then 1I. ... for each clip."
            )
            tokens_for_batch = max(max_new_tokens, batch_size * 300 + 1024)

        # Add thinking budget so reasoning tokens don't eat into the content budget
        thinking_budget = 8192

        print(f"[LLM] --- Batch clips {batch_start + 1}-{batch_start + batch_size} ({prompt_type}) ---")
        print(f"[LLM] Token budget: {tokens_for_batch} content + {thinking_budget} thinking")
        print(f"[LLM] User prompt:\n{user_prompt}")

        raw = generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_new_tokens=tokens_for_batch,
            temperature=0.8,
            image_paths=batch_images,
            thinking_budget=thinking_budget,
        )

        print(f"[LLM] Output:\n{raw}")

        if prompt_type in ("image", "video"):
            # Parse simple numbered list: "1. prompt" or "1) prompt"
            prompts_list: list = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned and len(cleaned) > 5:
                    prompts_list.append(cleaned)

            key = "image_prompt" if prompt_type == "image" else "video_prompt"
            if len(prompts_list) < batch_size:
                print(f"[LLM] WARNING: Parsed {len(prompts_list)} prompts but need {batch_size} — using fallback for remaining clips")
            for k in range(batch_size):
                global_idx = batch_start + k
                clip = clips[global_idx] if global_idx < len(clips) else {}
                section = clip.get("section_label", "verse")
                if k < len(prompts_list):
                    p = prompts_list[k]
                else:
                    # Build a clip-specific fallback from clip context
                    p = _build_fallback_prompt(clip, global_idx, section, lyrics, speaker_names, prompt_type)
                all_plans.append({key: p})

        else:
            # Parse V/I pairs (legacy "both" mode)
            video_prompts: list = []
            image_prompts: list = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                v_match = re.match(r"^\d+\s*V[\.\)]\s*(.*)", line, re.IGNORECASE)
                i_match = re.match(r"^\d+\s*I[\.\)]\s*(.*)", line, re.IGNORECASE)
                if v_match and v_match.group(1).strip():
                    video_prompts.append(v_match.group(1).strip())
                elif i_match and i_match.group(1).strip():
                    image_prompts.append(i_match.group(1).strip())
                else:
                    cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                    if cleaned and len(cleaned) > 10:
                        video_prompts.append(cleaned)

            for k in range(batch_size):
                global_idx = batch_start + k
                clip = clips[global_idx] if global_idx < len(clips) else {}
                section = clip.get("section_label", "verse")

                vp = video_prompts[k] if k < len(video_prompts) else (
                    _build_fallback_prompt(clip, global_idx, section, lyrics, speaker_names, "video")
                )
                ip = image_prompts[k] if k < len(image_prompts) else (
                    _build_fallback_prompt(clip, global_idx, section, lyrics, speaker_names, "image")
                )
                all_plans.append({"video_prompt": vp, "image_prompt": ip})

        print(f"[LLM] Planned {prompt_type} prompts for clips {batch_start + 1}-{batch_start + batch_size}")

    return all_plans[:len(clips)]


# ---------------------------------------------------------------------------
# Short Film: cinematic prompt generation
# ---------------------------------------------------------------------------


def _build_short_film_clip_description(
    clip: dict,
    index: int,
    lyrics: Optional[list] = None,
    speaker_names: Optional[dict] = None,
    characters: Optional[list] = None,
) -> str:
    """Build a clip description for short film mode.

    Focuses on dialogue content and character presence rather than
    musical concepts like beats and sections.
    """
    start = clip.get("start", 0)
    end = clip.get("end", 0)
    duration = end - start

    # Gather dialogue lines in this clip
    dialogue_lines = clip.get("dialogue_lines", [])
    if not dialogue_lines and lyrics:
        dialogue_lines = [
            l["text"] for l in lyrics
            if l["start"] < end and l["end"] > start
        ]

    # Identify speakers in this clip
    speakers_in_clip: list = []
    if lyrics:
        seen = set()
        for l in sorted(lyrics, key=lambda x: x.get("start", 0)):
            if l["start"] < end and l["end"] > start:
                spk = l.get("speaker")
                if spk and spk not in seen:
                    seen.add(spk)
                    name = speaker_names.get(spk, spk) if speaker_names else spk
                    speakers_in_clip.append(name)

    # Build character info
    char_info = ""
    if characters:
        char_names = [c.get("name", "") for c in characters if c.get("name")]
        if char_names:
            char_info = f" Characters: {', '.join(char_names)}."

    if speakers_in_clip:
        char_info = f" On screen: {', '.join(speakers_in_clip)}."

    # Build dialogue snippet
    dialogue_text = ""
    if dialogue_lines:
        snippet = " / ".join(dialogue_lines[:4])
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."
        dialogue_text = f' Dialogue: "{snippet}"'

    scene_label = clip.get("section_label", "scene")
    return (
        f"Shot {index + 1}: {scene_label}, {duration:.1f}s.{char_info}{dialogue_text}"
    )


def plan_short_film_prompts(
    clips: list,
    scene_description: str,
    lyrics: Optional[list] = None,
    max_new_tokens: int = 512,
    reference_image_path: Optional[str] = None,
    speaker_mappings: Optional[dict] = None,
    characters: Optional[list] = None,
    prompt_type: str = "both",
    existing_image_prompts: Optional[list] = None,
) -> list:
    """Generate per-clip prompts for short film mode.

    Similar to ``plan_clip_prompts_and_images`` but uses cinematic/narrative
    system prompts instead of music video prompts. Focuses on dialogue,
    character acting, and cinematic camera work.

    Returns same format as plan_clip_prompts_and_images.
    """
    if not clips:
        return []

    import re

    all_plans: list = []

    # Build speaker name map
    speaker_names: dict = {}
    if speaker_mappings:
        for spk_id, info in speaker_mappings.items():
            if info.get("name"):
                speaker_names[spk_id] = info["name"]

    has_image = reference_image_path and os.path.isfile(reference_image_path)

    # ── Character context for system prompt ──────────────────────
    char_context = ""
    if characters:
        char_lines = []
        for c in characters:
            name = c.get("name", "")
            desc = c.get("description", "")
            if name:
                char_lines.append(f"  - {name}" + (f": {desc}" if desc else ""))
        if char_lines:
            char_context = "Characters:\n" + "\n".join(char_lines) + "\n\n"

    # ── Build system prompt ──────────────────────────────────────
    photo_line = (
        "You are given a REFERENCE PHOTO showing the characters. "
        "Use it to identify the people, their appearance, clothing, and setting.\n"
    ) if has_image else ""

    char_rule = (
        "- NEVER use character names in image OR video prompts — neither the image "
        "editor nor the video model can identify people by name. Instead, describe "
        "each person by their VISIBLE appearance: clothing, hair, position in frame, "
        "based on what you visually see in the attached reference image. For example, "
        "write 'the woman in the white lab coat' instead of 'Dr Ava', or "
        "'the man in the blue shirt' instead of 'Mr Johnson'.\n"
        "- Character names ARE only allowed in spoken dialogue in the video prompt.\n"
        "- Describe characters using what you SEE in the photo + the character descriptions."
    ) if has_image else (
        "- NEVER use character names in image OR video prompts. Instead, describe each "
        "person by their visual appearance: clothing, hair color, position. "
        "For example, 'the tall woman in the red dress' not 'Sarah'.\n"
        "- Character names ARE only allowed in spoken dialogue in the video prompt.\n"
        "- Describe characters using only the character descriptions provided."
    )

    if prompt_type == "image":
        if has_image:
            system_prompt = (
                f"You are a cinematic scene designer for a short film.\n{photo_line}\n"
                f"{char_context}"
                "Each shot's image prompt EDITS the reference photo to create a starting "
                "frame. Think like a cinematographer — vary camera angle, framing, or lighting "
                "when it serves the scene, but maintain continuity when that makes more sense.\n\n"
                "Focus your prompt on WHAT TO CHANGE — new camera angles, different framing, "
                "new settings, repositioned characters, lighting shifts. You do NOT need to "
                "re-describe things that stay the same as the reference photo.\n\n"
                "WHEN TO DESCRIBE CHARACTERS: When you change the setting to a new location, "
                "re-describe characters by clothing/appearance so the editor knows who to place. "
                "When the setting stays the same, you only need to describe characters whose "
                "position changes.\n\n"
                "GOOD EXAMPLES:\n"
                "- 'change to a dramatic close-up of the woman in the white coat, soft backlight from the window'\n"
                "- 'change the setting to a dark alley with rain. The woman in the white coat stands against a wall.'\n"
                "- 'change to an over-the-shoulder shot from behind the man, looking across the room'\n\n"
                "BAD EXAMPLES (never write prompts like this):\n"
                "- 'change to a bright living room. Woman in pink sits on couch. Man sits next to her.' — re-describes what's already in the reference photo\n"
                "- 'Edit the provided image. Show Dr Ava standing.' — uses name, meta-instruction\n\n"
                "RULES:\n"
                "- Use a mix of shot types (close-ups, wide shots, over-shoulder, etc.) where "
                "appropriate — but continuity between consecutive scenes is fine when it fits.\n"
                "- Focus on WHAT TO CHANGE from the reference. Do not re-describe things that stay the same.\n"
                "- When setting changes to a new location, describe the new setting AND re-describe characters by appearance.\n"
                "- Stay faithful to each scene's scripted location — do NOT relocate a scene or invent new places for visual variety.\n"
                "- Match the mood and tone of the dialogue for that scene.\n"
                f"{char_rule}\n"
                "- Do NOT start with 'Edit the provided image'.\n"
                "- Do NOT use preservation meta-language ('preserve', 'maintain', 'keep unchanged').\n"
                "- Do NOT describe actions or motion — that belongs in video prompts.\n"
                "- Keep each prompt under 40 words.\n\n"
                "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
            )
        else:
            system_prompt = (
                f"You are a cinematic scene designer for a short film.\n\n"
                f"{char_context}"
                "For each shot, write a SCENE DESCRIPTION — where the characters are, "
                "what they are doing, their expressions, the lighting and mood.\n\n"
                "RULES:\n"
                "- Match the mood and tone of the dialogue.\n"
                "- Use cinematic composition — think about framing, depth, lighting.\n"
                f"{char_rule}\n"
                "- Keep each prompt under 25 words.\n\n"
                "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
            )
    elif prompt_type == "video":
        system_prompt = (
            f"You write short video motion prompts for a short film.\n"
            f"{char_context}"
            "Each shot is a SINGLE CONTINUOUS TAKE — one unbroken camera move "
            "with no cuts or edits. Describe WHO is on screen (by clothing and "
            "appearance, NOT by name), WHAT they are doing "
            "(gestures, expressions, body language), the SETTING, and how the "
            "CAMERA moves during this one take.\n\n"
            "RULES:\n"
            "- NEVER say 'montage', 'quick cuts', 'cut to', 'series of shots', "
            "or 'multiple angles' — these are impossible in a single take.\n"
            "- NEVER use character names — describe people by clothing/appearance only. "
            "Character names ARE only allowed in spoken dialogue.\n"
            "- CRITICAL: Each video prompt is generated INDEPENDENTLY — the video "
            "model has NO memory of previous scenes and does NOT know who any character "
            "is. You MUST re-describe every character's clothing, hair, and appearance "
            "in EVERY video prompt.\n"
            "- Focus on acting, body language, and emotional expression.\n"
            "- Match camera complexity to the content. For someone talking to camera "
            "or a simple conversation, use steady framing with minimal movement. "
            "For dramatic or action scenes, use expressive camera work "
            "(push in, dolly, tracking, pan, orbit).\n"
            "- Match the camera style to the emotional tone of the dialogue.\n"
            "- Keep each prompt under 25 words.\n\n"
            "Examples: 'The man in the dark suit slowly stands, camera pushes in on his face as he slams the table.' / "
            "'The blonde woman in the red blouse whispers across the table, camera drifts to a close-up of her trembling hands.' / "
            "'Medium shot, the woman in the gray sweater speaks calmly to camera, soft natural light, steady framing.'\n\n"
            "Format: numbered list. '1. prompt' then '2. prompt' etc. Output ONLY numbered prompts."
        )
    else:
        # Combined "both" mode
        image_instruction = (
            "- I (image edit): EDITS the reference photo to create the FIRST FRAME BEFORE action begins. "
            "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
            "show the room without them. Focus on WHAT TO CHANGE from reference. "
            "Describe POSES as static states (standing, seated, leaning). "
            "No motion verbs (walking, running, reaching, heaving, turning). "
            "No motion blur, speed lines, or long-exposure effects — the frame is sharp. "
            "Anchor the performer as 'the [descriptor] from the reference image'. "
            "NEVER use names. Actions belong ONLY in V, never in I.\n"
        ) if has_image else (
            "- I (image): the FIRST FRAME BEFORE action begins — a frozen still photograph. "
            "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
            "the room is empty. Describe static poses only (standing, seated, leaning). "
            "No motion verbs (walking, running, reaching, heaving, turning). "
            "Characters (by appearance, not name), positions, setting, lighting, mood.\n"
        )
        system_prompt = (
            f"You are a short film director.\n{photo_line}\n"
            f"{char_context}"
            "The user gives you a STORY CONCEPT describing the setting, characters, "
            "and narrative. Your job is to bring this story to life shot by shot.\n\n"
            "IMPORTANT — understand what each prompt controls:\n"
            "- V (video): a SINGLE CONTINUOUS SHOT with no cuts or edits. This is where "
            "ALL motion, action, gestures, dialogue, and camera movement happen. "
            "Describe who is on screen (by clothing/appearance — re-describe in EVERY "
            "prompt), what they DO during this take, and how the camera moves. "
            "NEVER say 'montage', 'quick cuts', 'cut to', 'series of shots', or "
            "'multiple angles' — these are impossible in a single take. "
            "Character names ARE only allowed in spoken dialogue.\n"
            f"{image_instruction}"
            "- I prompts describe ONE SINGLE STATIC IMAGE — the starting frame that "
            "the video animates FROM. No motion, no actions, no gestures. Just a frozen "
            "establishing shot: where everyone is, what the setting looks like.\n\n"
            "CINEMATIC DIRECTION:\n"
            "- Match camera complexity to the content. If someone is speaking directly "
            "to camera or having a simple conversation, use steady framing — a medium "
            "or close-up shot with minimal movement. If the story involves action, "
            "multiple locations, or dramatic reveals, use varied cinematic shots "
            "(over-shoulder, tracking, wide establishing shots, close-ups).\n"
            "- Use the DIALOGUE to inform character emotions and body language.\n"
            "- Use the Story Concept for setting, mood, and visual style.\n"
            "- Think like a cinematographer — lighting, depth of field, composition.\n"
            "- Match camera movement to emotional intensity.\n\n"
            "RULES:\n"
            f"{char_rule}\n"
            "- CRITICAL: Each V and I prompt is generated INDEPENDENTLY — the video/image "
            "model has NO memory of previous scenes and does NOT know who any character "
            "is. You MUST re-describe every character's clothing, hair, and appearance "
            "in EVERY V and I prompt, even if you described them in the previous prompt.\n"
            "- Do NOT put dialogue text or spoken words in I prompts.\n"
            "- Do NOT start I prompts with 'Edit the provided image'.\n"
            "- Do NOT use meta-language ('preserve', 'maintain', 'keep unchanged') in I prompts.\n"
            "- DO use action verbs in I prompts: 'change', 'make', 'move to', 'add'.\n"
            "- Keep each prompt under 40 words.\n\n"
            "Format: '1V. prompt' then '1I. prompt'. Output ONLY numbered prompts."
        )

    batch_images = [reference_image_path] if has_image else None

    print(f"[LLM] Short film prompts: prompt_type={prompt_type}, {len(clips)} clips")

    # Process all clips in one batch for narrative coherence
    clip_descriptions = []
    for j, clip in enumerate(clips):
        desc = _build_short_film_clip_description(
            clip, j, lyrics, speaker_names, characters,
        )
        clip_descriptions.append(desc)

    clips_text = "\n".join(clip_descriptions)
    batch_size = len(clips)

    if prompt_type == "image":
        user_prompt = (
            f"Story Concept: {scene_description}\n\n"
            f"Shots:\n{clips_text}\n\n"
            f"Design each shot's starting frame based on the Story Concept and dialogue context. "
            f"Output format: 1. description, 2. description, etc."
        )
        tokens_for_batch = max(max_new_tokens, batch_size * 80)
    elif prompt_type == "video":
        context_lines = ""
        if existing_image_prompts:
            img_context = [
                f"Shot {i + 1} starts as: {ip}"
                for i, ip in enumerate(existing_image_prompts) if ip
            ]
            if img_context:
                context_lines = "\nStarting frames:\n" + "\n".join(img_context) + "\n"

        user_prompt = (
            f"Story Concept: {scene_description}\n{context_lines}\n"
            f"Shots:\n{clips_text}\n\n"
            f"Write a video prompt for each shot. Focus on character acting, "
            f"body language, and cinematic camera movement. "
            f"Output format: 1. prompt, 2. prompt, etc."
        )
        tokens_for_batch = max(max_new_tokens, batch_size * 80 + 1024)
    else:
        user_prompt = (
            f"Story Concept: {scene_description}\n\n"
            f"Shots:\n{clips_text}\n\n"
            f"Direct this short film scene by scene. "
            f"Write V and I prompts for each shot. "
            f"Output format: 1V. ... then 1I. ... for each shot."
        )
        tokens_for_batch = max(max_new_tokens, batch_size * 150 + 1024)

    thinking_budget = 8192
    print(f"[LLM] Token budget: {tokens_for_batch} content + {thinking_budget} thinking")
    print(f"[LLM] User prompt:\n{user_prompt}")

    raw = generate(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=tokens_for_batch,
        temperature=0.8,
        image_paths=batch_images,
        thinking_budget=thinking_budget,
    )

    print(f"[LLM] Output:\n{raw}")

    if prompt_type in ("image", "video"):
        prompts_list: list = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
            if cleaned and len(cleaned) > 5:
                prompts_list.append(cleaned)

        key = "image_prompt" if prompt_type == "image" else "video_prompt"
        for k in range(batch_size):
            if k < len(prompts_list):
                all_plans.append({key: prompts_list[k]})
            else:
                fallback = f"Cinematic shot {k + 1}, {clips[k].get('section_label', 'scene')}"
                all_plans.append({key: fallback})
    else:
        video_prompts: list = []
        image_prompts: list = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            v_match = re.match(r"^\d+\s*V[\.\)]\s*(.*)", line, re.IGNORECASE)
            i_match = re.match(r"^\d+\s*I[\.\)]\s*(.*)", line, re.IGNORECASE)
            if v_match and v_match.group(1).strip():
                video_prompts.append(v_match.group(1).strip())
            elif i_match and i_match.group(1).strip():
                image_prompts.append(i_match.group(1).strip())
            else:
                cleaned = re.sub(r"^\d+[\.\)]\s*", "", line)
                if cleaned and len(cleaned) > 10:
                    video_prompts.append(cleaned)

        for k in range(batch_size):
            vp = video_prompts[k] if k < len(video_prompts) else f"Cinematic shot {k + 1}"
            ip = image_prompts[k] if k < len(image_prompts) else f"Scene {k + 1} establishing frame"
            all_plans.append({"video_prompt": vp, "image_prompt": ip})

    return all_plans[:len(clips)]


# ---------------------------------------------------------------------------
# Short Film Path C — plan scenes from a story description (no audio)
# ---------------------------------------------------------------------------


def plan_short_film_from_story(
    story_description: str,
    characters: Optional[list] = None,
    reference_image_path: Optional[str] = None,
    target_duration: int = 30,
    target_scenes: Optional[int] = None,
    narrative_mode: bool = True,
    fps: int = 24,
    frames_steps: int = 4,
    frames_minimum: int = 5,
    max_new_tokens: int = 1024,
) -> dict:
    """Plan a short film scene structure from a story description.

    Unlike ``plan_dialogue_scenes`` which analyses uploaded audio, this uses
    the LLM to create scenes from scratch — deciding how many scenes there
    are, what dialogue occurs, and how to pace them within *target_duration*.

    Returns ``{"clips": [...], "clip_plans": [...]}``:
    - clips: same format as ``plan_dialogue_scenes`` output
    - clip_plans: same format as ``plan_short_film_prompts`` output
    """
    import json as _json
    import re

    from services.audio_analysis import _snap_to_valid_frames

    if not target_scenes:
        # ~15 seconds per scene, cap at 30 scenes
        target_scenes = max(2, min(30, target_duration // 15))

    has_image = reference_image_path and os.path.isfile(reference_image_path)

    # ── Character context ─────────────────────────────────────────
    char_context = ""
    if characters:
        char_lines = []
        for c in characters:
            name = c.get("name", "")
            desc = c.get("description", "")
            if name:
                char_lines.append(f"  - {name}" + (f": {desc}" if desc else ""))
        if char_lines:
            char_context = "Characters:\n" + "\n".join(char_lines) + "\n\n"

    photo_line = (
        "You are given a REFERENCE PHOTO showing the characters. "
        "Use it to identify the people, their appearance, clothing, and setting.\n"
    ) if has_image else ""

    char_rule = (
        "- NEVER use character names in image_prompt OR video_prompt — neither the "
        "image editor nor the video model can identify people by name. Instead, "
        "describe each person by their VISIBLE appearance: clothing, hair, position "
        "in frame, based on what you visually see in the attached reference image. "
        "For example, write 'the woman in the white lab coat' instead of a character name.\n"
        "- Character names ARE only allowed in spoken dialogue within the video_prompt.\n"
        "- Describe characters using what you SEE in the photo + the character descriptions."
    ) if has_image else (
        "- NEVER use character names in image_prompt OR video_prompt. Instead, describe "
        "each person by their visual appearance: clothing, hair color, position.\n"
        "- Character names ARE only allowed in spoken dialogue within the video_prompt.\n"
        "- Describe characters using only the character descriptions provided."
    )

    # ── Load prompting guides ────────────────────────────────────
    guides_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_guides")

    def _load_guide(filename):
        p = os.path.join(guides_dir, filename)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"[LLM] Loaded guide: {filename} ({len(content)} chars)")
            return content
        return ""

    video_guide = _load_guide("LTX-2_PROMPTING_GUIDE_Embedded_Audio.MD")
    story_guide = _load_guide("Expert short-form storyteller.md") if narrative_mode else ""

    # ── System prompt: scene planner + prompt writer ──────────────
    image_instruction = (
        "- image_prompt: EDITS the reference photo to create the FIRST FRAME BEFORE action begins. "
        "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
        "show the room without them. Focus on WHAT TO CHANGE from reference. "
        "Describe POSES as static states (standing, seated, leaning). "
        "No motion verbs (walking, running, reaching, heaving, turning). "
        "Actions belong ONLY in video_prompt, never in image_prompt. "
        "NEVER use character names or meta-instructions.\n"
    ) if has_image else (
        "- image_prompt: the FIRST FRAME BEFORE action begins — a frozen still photograph. "
        "Show the INITIAL STATE: if clothing will be removed, it's still on; if someone enters, "
        "the room is empty. Describe static poses only (standing, seated, leaning). "
        "No motion verbs (walking, running, reaching, heaving, turning). "
        "Characters (by appearance, NOT by name), positions, setting, lighting, mood.\n"
    )

    # Build guide sections
    guide_sections = ""
    if video_guide:
        guide_sections += (
            "\n\nVIDEO PROMPTING GUIDE — follow this when writing video_prompt values:\n"
            "---\n"
            f"{video_guide}\n"
            "---\n"
        )
    # NOTE: The Qwen image edit guide is NOT included — its examples conflict
    # with Director mode requirements (uses character names, meta-instructions).
    # The system prompt rules are sufficient for Director image prompts.

    # Narrative vs non-narrative role
    if narrative_mode and story_guide:
        role_section = (
            "You are a short film director, screenwriter, and expert storyteller.\n"
            f"{photo_line}\n{char_context}"
            "Follow this storytelling guide when structuring your scenes:\n"
            "---\n"
            f"{story_guide}\n"
            "---\n\n"
            "The user gives you a STORY CONCEPT. Plan the short film with a clear "
            "narrative arc: setup, rising conflict, climax, and resolution.\n"
        )
    else:
        role_section = (
            f"You are a short film director and screenwriter.\n{photo_line}\n"
            f"{char_context}"
            "The user gives you a CONCEPT. Plan the short film as a sequence of "
            "visually compelling scenes that cover the concept thoroughly. "
            "Focus on variety, pacing, and visual impact rather than narrative arc.\n"
        )

    system_prompt = (
        f"{role_section}"
        f"Break the concept into scenes within {target_duration} seconds total. "
        f"YOU decide how many scenes based on the story — let pacing dictate the cuts.\n"
        "For each scene, write a video_prompt and image_prompt.\n\n"
        "PROMPT TYPES:\n"
        "- video_prompt: a SINGLE CONTINUOUS SHOT — one flowing paragraph following the "
        "video prompting guide below. Each clip can be up to 20 seconds long. "
        "NEVER say 'montage', 'quick cuts', 'cut to'. "
        "Each video_prompt is rendered independently with no memory of other scenes "
        "and the video model does NOT know who any character is. "
        "NEVER use character names in video_prompt — describe people by clothing and "
        "appearance. Character names ARE only allowed in spoken dialogue. "
        "Re-describe every character's clothing, hair, and appearance in EVERY "
        "video_prompt, plus re-state all relevant visual context (weather, time of day, "
        "environment state) within each prompt.\n"
        "CAMERA STYLE — match to the content:\n"
        "- If the concept is someone talking to camera, giving a speech, or having a "
        "simple conversation in one place: use steady, consistent framing (medium or "
        "close-up shots). Minimal camera movement. Keep image_prompts similar across "
        "scenes — same angle, same setting, subtle lighting or expression changes only.\n"
        "- If the concept is a narrative with action, multiple locations, or dramatic "
        "reveals: use varied cinematic shots (over-shoulder, tracking, wide establishing, "
        "close-ups). Vary image_prompts to show different angles, settings, and compositions.\n"
        "- Let the story concept guide you — don't force cinematic complexity onto simple content.\n\n"
        f"{image_instruction}\n"
        "IMAGE PROMPT GUIDE (how to write image_prompt values):\n"
        "The image_prompt EDITS the reference photo to create a starting frame for each "
        "scene. Match your approach to the camera style above — vary angle and framing "
        "when the story calls for it, maintain continuity when it doesn't.\n\n"
        "The image editor starts with the reference photo and applies your changes. "
        "Focus your prompt on WHAT TO CHANGE — new camera angles, different framing, "
        "new settings, repositioned characters, lighting shifts. You do NOT need to "
        "re-describe things that stay the same as the reference photo.\n\n"
        "IMPORTANT — what each prompt does:\n"
        "- image_prompt = the FIRST FRAME BEFORE action begins. Describe the shot setup: camera "
        "angle, framing, setting, lighting, character positions.\n"
        "- video_prompt = all motion, action, dialogue, gestures, camera movement\n\n"
        "WHEN TO DESCRIBE CHARACTERS: When you change the setting to a new location, "
        "re-describe characters by clothing/appearance so the editor knows who to place. "
        "When the setting stays the same, you only need to describe characters whose "
        "position changes.\n\n"
        "GOOD image_prompt examples (assuming reference = people sitting on couch in living room):\n"
        "- 'change to a dramatic close-up of the blonde woman in the pink shirt, soft "
        "backlight from the window, shallow depth of field'\n"
        "- 'change the setting to a restaurant at night. The blonde woman in the pink "
        "shirt and the man in the dark shirt sit across from each other at a candlelit table.'\n"
        "- 'change to an over-the-shoulder shot from behind the man in the dark shirt, "
        "looking at the girl in the light blue dress across the room'\n\n"
        "BAD image_prompt examples (NEVER write like this):\n"
        "- 'change to a bright modern living room. Blonde woman in pink shirt sits on "
        "beige couch. Man in dark shirt sits in middle. Girl in light blue dress sits "
        "on right.' — re-describes everything already in the reference with no visual change\n"
        "- 'The girl in blue raises her hand' — this is ACTION, belongs in video_prompt\n"
        "- 'Dr Ava stands in the hallway' — uses character name\n\n"
        "RULES for image_prompt:\n"
        "- Use a mix of shot types (close-ups, wide shots, over-shoulder, etc.) where "
        "appropriate — but continuity between consecutive scenes is fine when it fits.\n"
        "- Focus on WHAT TO CHANGE from the reference. Do not re-describe things that "
        "stay the same.\n"
        "- When setting changes to a new location, describe the new setting AND "
        "re-describe which characters are present by clothing/appearance.\n"
        "- If the user's concept pins the story to a specific location, every scene "
        "stays in that location — do NOT relocate scenes or invent new places for "
        "visual variety.\n"
        "- When setting stays the same, you can keep the same framing or adjust it — "
        "only mention characters whose positions change.\n"
        "- NEVER use character names — describe people by clothing/appearance only, "
        "based on what you visually see in the attached reference image\n"
        "- Character names ARE only allowed in spoken dialogue in the video_prompt\n"
        "- Do NOT start with 'Edit the provided image'\n"
        "- Do NOT use meta-language ('preserve', 'maintain', 'keep unchanged')\n"
        "- image_prompt = STILL PHOTOGRAPH. No motion verbs whatsoever: no walking, "
        "running, reaching, heaving, turning, gesturing, raising, dancing. "
        "Describe static poses only (standing, seated, leaning). All action belongs in video_prompt.\n"
        "- Carry forward cumulative visual state changes: if a character got wet, "
        "injured, or changed clothes earlier, mention that difference.\n"
        f"{char_rule}\n"
        f"{guide_sections}\n"
        "OUTPUT FORMAT — respond with ONLY a JSON array, no markdown fences, no thinking tags:\n"
        "[\n"
        '  {"title": "Scene title", "duration": 15, '
        '"dialogue": ["Character: \\"Full sentence of dialogue matching clip length\\""], '
        '"scene_type": "dialogue|action|opening|closing", '
        '"video_prompt": "Single flowing paragraph describing action, setting, and lighting. '
        'Character speaks with emotion, \\"Full dialogue woven into the scene with speaker cues '
        'and enough words to fill most of the clip duration.\\" '
        'Camera movement and reaction described.", '
        '"image_prompt": "Starting frame description"}\n'
        "]\n\n"
        "CRITICAL RULES:\n"
        f"- The durations must sum to approximately {target_duration} seconds.\n"
        "- Each scene should be 10-20 seconds long.\n"
        "- Every scene MUST be unique — never repeat the same video_prompt or image_prompt.\n"
        "- Do NOT repeat scenes to fill the duration. Use fewer, longer scenes instead.\n"
        '- DIALOGUE IN VIDEO_PROMPT: Any spoken dialogue MUST appear inside the video_prompt '
        'as quoted text with a speaker cue, woven into the scene description. '
        "The dialogue field is just a metadata summary — the video_prompt is what the "
        "video model actually reads and generates from.\n"
        "- DIALOGUE LENGTH: People speak at ~2 words per second. Aim for roughly "
        "duration × 2 words of dialogue per scene (e.g. ~20 words for a 10s scene, "
        "~30 for 15s). Don't write throwaway one-liners, but don't overpack either. "
        "The system will adjust if needed."
    )

    user_prompt = f"Story Concept: {story_description}"

    batch_images = [reference_image_path] if has_image else None

    print(f"[LLM] Planning short film from story: {target_scenes} scenes, {target_duration}s")
    print(f"[LLM] Story: {story_description}")
    print(f"[LLM] Token budget: {target_scenes * 400 + 256} content + 8192 thinking = {target_scenes * 400 + 256 + 8192} total")

    # Scale tokens to scene count — each scene needs ~200 tokens of JSON
    # Guide-quality prompts are longer (~400 tokens/scene with rich descriptions)
    tokens_needed = max(max_new_tokens, target_scenes * 400 + 256)

    # Reserve extra tokens for model thinking/reasoning so it doesn't eat
    # into the content budget. The 27B model can use 5000+ thinking tokens.
    thinking_budget = 8192

    raw = generate_streaming(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_new_tokens=tokens_needed,
        temperature=0.8,
        image_paths=batch_images,
        thinking_budget=thinking_budget,
    )

    print(f"[LLM] Story plan output:\n{raw}")

    # ── Parse JSON response ───────────────────────────────────────
    cleaned = raw.strip()

    # Strip thinking blocks (Qwen <think>...</think> and Gemma <|channel>thought\n...<channel|>)
    cleaned = _strip_thinking_tags(cleaned).strip()

    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    scenes = []
    try:
        scenes = _json.loads(cleaned)
    except _json.JSONDecodeError:
        # Try to find the LAST JSON array (skip any in think block remnants)
        matches = list(re.finditer(r"\[[\s\S]*?\](?=\s*$)", cleaned))
        if matches:
            try:
                scenes = _json.loads(matches[-1].group())
            except _json.JSONDecodeError:
                pass
        if not scenes:
            # Broader search
            match = re.search(r"\[[\s\S]*\]", cleaned)
            if match:
                try:
                    scenes = _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass

    # If full parse failed, try to salvage complete JSON objects from truncated output
    if not scenes:
        obj_pattern = re.finditer(
            r'\{\s*"title"\s*:.*?"image_prompt"\s*:\s*"[^"]*"\s*\}',
            cleaned, re.DOTALL,
        )
        salvaged = [m.group() for m in obj_pattern]
        for s in salvaged:
            try:
                scenes.append(_json.loads(s))
            except _json.JSONDecodeError:
                continue
        if scenes:
            print(f"[LLM] Salvaged {len(scenes)} complete scenes from truncated output")

    if not scenes:
        # Fallback: create evenly spaced scenes
        print("[LLM] WARNING: Could not parse scene plan, using fallback")
        scene_dur = target_duration / target_scenes
        for i in range(target_scenes):
            scenes.append({
                "title": f"Scene {i + 1}",
                "duration": scene_dur,
                "dialogue": [],
                "scene_type": "action",
                "video_prompt": f"Cinematic shot {i + 1}, characters in motion",
                "image_prompt": f"Scene {i + 1} establishing frame",
            })

    # ── Post-process: deduplicate and cap dialogue ────────────────
    # Remove scenes with identical video_prompt (LLM repetition loop)
    seen_prompts = set()
    unique_scenes = []
    for scene in scenes:
        vp = scene.get("video_prompt", "")
        if vp not in seen_prompts:
            seen_prompts.add(vp)
            unique_scenes.append(scene)
        else:
            print(f"[LLM] Removed duplicate scene: {scene.get('title', '?')}")
    if len(unique_scenes) < len(scenes):
        print(f"[LLM] Deduplicated {len(scenes)} → {len(unique_scenes)} scenes")
        scenes = unique_scenes

    # Cap dialogue lines per scene to prevent repetition loops
    MAX_DIALOGUE_LINES = 6
    for scene in scenes:
        dialogue = scene.get("dialogue", [])
        if len(dialogue) > MAX_DIALOGUE_LINES:
            print(f"[LLM] Capped dialogue in '{scene.get('title', '?')}': {len(dialogue)} → {MAX_DIALOGUE_LINES} lines")
            scene["dialogue"] = dialogue[:MAX_DIALOGUE_LINES]

    # ── Dialogue budget check: ask LLM to rewrite over-budget scenes ──
    # People speak at ~2-2.5 words per second. If dialogue exceeds that,
    # ask the LLM to condense just those scenes (it keeps narrative sense).
    over_budget = []
    for i, scene in enumerate(scenes):
        vp = scene.get("video_prompt", "")
        duration = float(scene.get("duration", 15))
        max_words = int(duration * 2.5)
        quotes = re.findall(r'"([^"]*)"', vp)
        if not quotes:
            continue
        total_words = sum(len(q.split()) for q in quotes)
        if total_words > max_words:
            over_budget.append((i, scene, total_words, max_words))

    if over_budget:
        print(f"[LLM] {len(over_budget)} scene(s) have dialogue over budget, requesting rewrite")
        rewrite_lines = []
        for idx, scene, actual, budget in over_budget:
            rewrite_lines.append(
                f"Scene {idx + 1} \"{scene.get('title', '')}\" ({scene.get('duration', 15)}s): "
                f"has {actual} words of dialogue, max {budget}. "
                f"Current video_prompt: {scene.get('video_prompt', '')}"
            )

        rewrite_prompt = (
            "The following scenes have too much spoken dialogue for their duration. "
            "People speak at about 2 words per second — if there are too many words, "
            "the actor will speak unnaturally fast or get cut off.\n\n"
            "Condense the dialogue in each video_prompt to fit the word budget. "
            "Keep the same meaning and narrative flow — just say it more concisely. "
            "Keep all non-dialogue parts (action, camera, setting) unchanged.\n\n"
            + "\n\n".join(rewrite_lines) + "\n\n"
            "Output ONLY the rewritten video_prompt for each scene, numbered to match:\n"
            f"Format: '{over_budget[0][0] + 1}. rewritten video_prompt'"
        )

        rewrite_raw = generate(
            prompt=rewrite_prompt,
            system_prompt="You condense dialogue to fit time constraints while preserving meaning and story continuity.",
            max_new_tokens=len(over_budget) * 200,
            temperature=0.5,
            thinking_budget=1024,
        )
        print(f"[LLM] Rewrite output:\n{rewrite_raw}")

        # Parse rewritten prompts and apply them
        for line in rewrite_raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)[\.\)]\s*(.*)", line)
            if m:
                scene_num = int(m.group(1))
                new_vp = m.group(2).strip()
                if len(new_vp) < 20:
                    continue
                # Find the matching over-budget scene
                for idx, scene, actual, budget in over_budget:
                    if idx + 1 == scene_num:
                        # Verify the rewrite actually reduced dialogue
                        new_quotes = re.findall(r'"([^"]*)"', new_vp)
                        new_word_count = sum(len(q.split()) for q in new_quotes)
                        print(f"[LLM] Scene {scene_num} dialogue: {actual} → {new_word_count} words (budget: {budget})")
                        scene["video_prompt"] = new_vp
                        # Update dialogue metadata from the new prompt
                        if new_quotes:
                            scene["dialogue"] = [f'"{q}"' for q in new_quotes]
                        break

    # ── Convert scenes to clip dicts ──────────────────────────────
    clips = []
    clip_plans = []
    current_time = 0.0

    for i, scene in enumerate(scenes):
        scene_dur = float(scene.get("duration", target_duration / len(scenes)))
        scene_dur = max(3.0, min(scene_dur, 20.0))  # clamp to reasonable range

        clip_start = round(current_time, 3)
        clip_end = round(current_time + scene_dur, 3)

        dialogue = scene.get("dialogue", [])
        scene_type = scene.get("scene_type", "dialogue" if dialogue else "action")

        clips.append({
            "start": clip_start,
            "end": clip_end,
            "beat_count": 0,
            "section_label": scene_type,
            "energy": 0.5,
            "suggested_prompt_hint": scene.get("title", f"Scene {i + 1}"),
            "duration_frames": _snap_to_valid_frames(scene_dur, fps, frames_steps, frames_minimum),
            "dominant_speaker": None,
            "dialogue_lines": dialogue,
        })

        clip_plans.append({
            "video_prompt": scene.get("video_prompt", f"Cinematic shot {i + 1}"),
            "image_prompt": scene.get("image_prompt", f"Scene {i + 1} establishing frame"),
        })

        current_time += scene_dur

    return {"clips": clips, "clip_plans": clip_plans}
