"""Live grammar-constraint test against the bundled llama-server.

Launches llama-server CPU-only (-ngl 0, never touches the GPU) on a
dedicated port, sends a chat completion with response_format =
{"type": "json_object", "schema": <the REAL Pass 2 shot schema>} and
thinking disabled — exactly the payload llm_service.generate_streaming
builds when json_schema is set — and asserts the output is schema-valid
JSON with every required field present.

Usage:
    python tests/manual_grammar_live_test.py <path-to-gguf> [max_tokens]

Manual/diagnostic — not part of an automated suite (needs a multi-GB
GGUF and several minutes of CPU inference).
"""
import json
import os
import subprocess
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from services.director.planners.short_film import _shot_list_schema  # noqa: E402

PORT = 18923
# llama-server binary — defaults to the standard install location relative to
# this repo; override with the LLAMA_SERVER_BIN env var for custom setups.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN = os.environ.get(
    "LLAMA_SERVER_BIN",
    os.path.join(_REPO_ROOT, "app", "ckpts", "llm", "bin",
                 "llama-server.exe" if sys.platform == "win32" else "llama-server"),
)

REQUIRED = [
    "title", "duration_sec", "scene_goal", "narrative_role",
    "scene_type", "subjects_on_screen", "environment",
    "visual_style", "lighting", "mood", "action_beats",
    "camera_plan", "ending_beat", "image_source", "image_prompt",
    "visual_changes", "video_prompt", "multishot", "window_prompts",
]


def main() -> int:
    gguf = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1600
    schema = _shot_list_schema(min_items=2, max_items=2, required=REQUIRED)

    cmd = [
        BIN, "-m", gguf,
        "--port", str(PORT), "--host", "127.0.0.1",
        "--threads", str(max(os.cpu_count() // 2, 2)),
        "--jinja", "--ctx-size", "8192", "--n-gpu-layers", "0",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        t0 = time.time()
        for _ in range(240):
            if proc.poll() is not None:
                print(f"FAIL: llama-server exited code {proc.returncode}")
                return 1
            try:
                if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            print("FAIL: server never became healthy")
            return 1
        print(f"server ready in {time.time() - t0:.0f}s")

        payload = {
            "messages": [
                {"role": "system", "content": "You are a film director breaking a story into shots. Output ONLY the JSON array."},
                {"role": "user", "content": "/no_think\n\nPlan exactly 2 shots (20s each) for a short film about a lighthouse keeper who finds a message in a bottle."},
            ],
            "max_tokens": max_tokens,
            "cache_prompt": False,
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "min_p": 0.05,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object", "schema": schema},
        }
        t1 = time.time()
        resp = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=payload, timeout=560)
        print(f"HTTP {resp.status_code} in {time.time() - t1:.0f}s")
        if resp.status_code != 200:
            print(f"FAIL: {resp.text[:1500]}")
            return 1
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        finish = data["choices"][0].get("finish_reason")
        print(f"finish={finish} content={len(content)}c reasoning={len(reasoning)}c")

        parsed = json.loads(content)  # raises on malformed = test failure
        assert isinstance(parsed, list), f"not a list: {type(parsed)}"
        assert len(parsed) == 2, f"expected exactly 2 items, got {len(parsed)}"
        for i, shot in enumerate(parsed):
            missing = [k for k in REQUIRED if k not in shot]
            assert not missing, f"shot {i} missing {missing}"
            extra = [k for k in shot if k not in _shot_list_schema(1, 1, [])["items"]["properties"]]
            assert not extra, f"shot {i} has non-schema keys {extra}"
        assert not reasoning, "thinking leaked despite grammar + enable_thinking=False"
        print("shot 0 preview:", json.dumps(parsed[0])[:300])
        print("PASS: grammar produced schema-valid JSON, exact count, no thinking")
        return 0
    finally:
        proc.kill()


if __name__ == "__main__":
    sys.exit(main())
