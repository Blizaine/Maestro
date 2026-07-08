"""
Base Planner — interface that all skill planners implement.

Planners are responsible for creative planning only. They output a ProductionPlan
with ShotPlan objects. They do NOT generate final model prompts — that's the
renderer's job.
"""

from __future__ import annotations
import json
import re
import os
from abc import ABC, abstractmethod
from typing import Optional, Any

# Optional dependency: json_repair handles common LLM JSON mistakes
# (missing commas between properties, trailing commas, single quotes,
# unclosed brackets, etc.). Used as a third fallback after strict
# json.loads + regex-extract-array. App still boots if missing — the
# fallback just degrades to "return None" for those cases like before.
try:
    import json_repair  # type: ignore
    _HAVE_JSON_REPAIR = True
except ImportError:
    json_repair = None  # type: ignore
    _HAVE_JSON_REPAIR = False

from ..schema import ProductionPlan, ShotPlan

# Grammar fallback for the JSON-fix retry when the caller didn't provide a
# shot schema: any JSON array of objects. llama-server compiles this to a
# GBNF grammar that masks every token which would break it, so the retry's
# output is structurally parseable by construction — no prose, no markdown
# fences, no half-finished repetition tail. Planners that know their exact
# shot shape pass a real schema instead (see short_film.py), which is
# stronger: closed objects (additionalProperties=false) emit each key at
# most once, making field-level repeat loops unrepresentable.
_GENERIC_ARRAY_SCHEMA = {"type": "array", "items": {"type": "object"}, "minItems": 1}


class BasePlanner(ABC):
    """Abstract base class for all skill planners."""

    # Subclasses set this
    skill_type: str = ""

    def __init__(self, llm_generate: Any = None, llm_generate_streaming: Any = None):
        """
        Args:
            llm_generate: callable matching llm_service.generate() signature
            llm_generate_streaming: callable matching llm_service.generate_streaming() signature
        """
        self._generate = llm_generate
        self._generate_streaming = llm_generate_streaming

    @abstractmethod
    def plan(self, **kwargs) -> ProductionPlan:
        """Create a ProductionPlan from skill-specific inputs.

        Each subclass defines its own keyword arguments.
        Returns a ProductionPlan with normalized ShotPlan objects.
        """
        ...

    # ── Shared LLM Helpers ───────────────────────────────────────────

    def _call_llm_json(
        self,
        user_prompt: str,
        system_prompt: str,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        temperature: float = 0.7,
        image_paths: Optional[list[str]] = None,
        streaming: bool = True,
        frequency_penalty: float = 0.3,
        presence_penalty: float = 0.1,
        json_schema: Optional[dict] = None,
    ) -> list[dict]:
        """Call LLM and parse JSON array response with retry on failure.

        Returns parsed list of dicts (shot plan objects).

        json_schema (optional) grammar-constrains the output on local
        llama-server so the model physically cannot emit malformed JSON
        or repeat-loop garbage. Applied on the FIRST attempt only when
        that model's thinking is off (grammar and thinking are mutually
        exclusive — see llm_service.generate — and the thinking-on Gemma
        path is a known-good flow we don't disturb). Applied ALWAYS on
        the parse-failure retry (with thinking forced off), falling back
        to a generic array-of-objects grammar when no schema was given,
        so every planner's retry is loop-proof. If the server rejects
        the grammar (old binary, remote provider), each constrained call
        degrades to the exact pre-grammar behavior.

        Repetition penalties (default frequency=0.3, presence=0.1) are
        critical here: planner output is structured JSON across many
        similar fields (subjects_on_screen, action_beats, window_prompts).
        Without penalties the model locks onto a "good" pattern and
        loops, producing window_prompts that repeat the same paragraph
        3-4 times within a single field — observed empirically on Qwen
        3.5/3.6 with the long Pass 2 system prompt. Restored after the
        c1a950c global-removal commit, which broke planning while
        helping creative streaming.

        thinking_budget is now MODEL-AWARE when caller leaves it as None:
          - Qwen3.5/3.6 ("qwen" thinking_style): budget=0, thinking forced
            off. These templates have known runaway behavior on long
            structured tasks — model burns the entire budget on internal
            reasoning before emitting any JSON (10-min hangs in test).
          - Gemma 3/4 ("gemma" thinking_style): budget=4096, thinking ON.
            Gemma's reasoning is well-behaved and helps it follow the
            structured-output rules (e.g. "21s shot needs window_prompts").
            Without thinking, smaller Gemma models like 4B miss these
            rules under cognitive load.
          - Other / unknown: budget=0, thinking off (conservative default).
        Callers can still pass an explicit thinking_budget to override.
        """
        gen_fn = self._generate_streaming if (streaming and self._generate_streaming) else self._generate
        if gen_fn is None:
            raise RuntimeError("No LLM generate function provided to planner")

        # Model-aware thinking budget when caller didn't specify
        if thinking_budget is None:
            try:
                from services import llm_service
                entry = llm_service._active_registry_entry()
                style = (entry or {}).get("thinking_style", "qwen") if isinstance(entry, dict) else "qwen"
                if style == "gemma":
                    thinking_budget = 4096
                else:
                    # Qwen and any unknown style: thinking off
                    thinking_budget = 0
            except Exception:
                thinking_budget = 0

        # When thinking_budget=0, explicitly disable thinking mode
        kwargs = {
            "prompt": user_prompt,
            "system_prompt": system_prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "thinking_budget": thinking_budget,
            "image_paths": image_paths or [],
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if thinking_budget == 0:
            kwargs["enable_thinking"] = False
            # Grammar on the first attempt only for thinking-off models:
            # the constraint applies from token 0, so it's a free win here
            # (Qwen 3.x and unknown styles), while thinking-on models keep
            # their unconstrained first attempt and rely on the retry net.
            if json_schema is not None:
                kwargs["json_schema"] = json_schema
        try:
            response = gen_fn(**kwargs)
        except Exception as e:
            if "json_schema" not in kwargs:
                raise
            # Never let the grammar make planning WORSE than before it
            # existed — a rejecting server (old llama-server binary, odd
            # provider) drops the constraint and runs the call as-is.
            print(f"[Planner] Grammar-constrained call failed ({e}); retrying unconstrained")
            kwargs.pop("json_schema", None)
            response = gen_fn(**kwargs)

        parsed = self._parse_json_response(response)
        if parsed is not None:
            return parsed

        # Retry once with explicit JSON fix instruction + a hard JSON
        # grammar. The grammar (caller's schema, or the generic array-of-
        # objects fallback) is what actually guarantees a parseable result
        # — the fix prompt alone has been defeated in the field (Gemma 4
        # 12B looped 96K chars of pseudo-JSON straight through it).
        # Thinking goes OFF here (grammar requires it; see llm_service).
        print("[Planner] JSON parse failed, retrying with fix prompt + JSON grammar...")
        fix_prompt = (
            "Your previous response was not valid JSON. "
            "Please output ONLY a JSON array of objects. "
            "No markdown fences, no explanation, no thinking tags. "
            "Just the raw JSON array starting with [ and ending with ].\n\n"
            f"Original request:\n{user_prompt}"
        )
        retry_kwargs = dict(
            prompt=fix_prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_tokens,
            temperature=0.3,
            thinking_budget=0,
            image_paths=image_paths or [],
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            enable_thinking=False,
            json_schema=json_schema or _GENERIC_ARRAY_SCHEMA,
        )
        try:
            response2 = gen_fn(**retry_kwargs)
        except Exception as e:
            # Same degradation contract as attempt 1: fall back to the
            # historical unconstrained retry (thinking budget restored).
            print(f"[Planner] Grammar-constrained retry failed ({e}); retrying unconstrained")
            retry_kwargs.pop("json_schema", None)
            retry_kwargs.pop("enable_thinking", None)
            retry_kwargs["thinking_budget"] = 2048
            response2 = gen_fn(**retry_kwargs)
        parsed2 = self._parse_json_response(response2)
        if parsed2 is not None:
            return parsed2

        print("[Planner] JSON parse failed on retry, returning empty list")
        return []

    def _parse_json_response(self, text: str) -> Optional[list[dict]]:
        """Extract and parse JSON array from LLM response text."""
        if not text:
            return None

        # Strip thinking tags (Qwen <think>...</think> and Gemma <|channel>thought\n...<channel|>)
        text = re.sub(r'<(think|thinking|seed:think|reasoning|reflection)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<(think|thinking|seed:think|reasoning|reflection)>.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<\|channel>thought\n.*?<channel\|>', '', text, flags=re.DOTALL)
        text = re.sub(r'<\|channel>thought\n.*$', '', text, flags=re.DOTALL)

        # Strip markdown fences
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                print(f"[Planner] JSON parse OK: {len(result)} items (direct)")
                return result
            if isinstance(result, dict) and "shots" in result:
                print(f"[Planner] JSON parse OK: {len(result['shots'])} items (shots key)")
                return result["shots"]
            return [result]
        except json.JSONDecodeError as e:
            print(f"[Planner] Direct JSON parse failed: {e}")
            print(f"[Planner] Text starts with: {text[:200]!r}")

        # Try to find JSON array in text
        match = re.search(r'\[[\s\S]*\]', text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, list):
                    print(f"[Planner] JSON parse OK: {len(result)} items (regex array)")
                    return result
            except json.JSONDecodeError as e:
                print(f"[Planner] Regex array parse failed: {e}")
                print(f"[Planner] Matched array starts with: {match.group()[:200]!r}")
        else:
            print(f"[Planner] No JSON array found in {len(text)} chars")

        # Try to find JSON object
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict) and "shots" in result:
                    return result["shots"]
                return [result]
            except json.JSONDecodeError:
                pass

        # Last-resort: json_repair handles common LLM mistakes that the
        # strict json.loads chokes on — missing commas between
        # properties, trailing commas, single quotes used for strings,
        # extra prose around the JSON, unclosed brackets, etc. We try
        # this last so the strict parser path is unchanged for clean
        # output (no perf regression) and json_repair only runs when
        # there's actually something to fix.
        if _HAVE_JSON_REPAIR:
            try:
                result = json_repair.loads(text)
                if isinstance(result, list):
                    print(f"[Planner] JSON parse OK via json_repair: {len(result)} items")
                    return result
                if isinstance(result, dict) and "shots" in result:
                    print(f"[Planner] JSON parse OK via json_repair (shots key): {len(result['shots'])} items")
                    return result["shots"]
                if isinstance(result, dict):
                    print("[Planner] JSON parse OK via json_repair: 1 item (single object)")
                    return [result]
            except Exception as e:
                print(f"[Planner] json_repair fallback failed: {e}")
        else:
            print("[Planner] json_repair not installed — install with `pip install json_repair` to recover from LLM JSON typos")

        print(f"[Planner] All JSON parse attempts failed. Text ends with: {text[-200:]!r}")
        return None

    # ── Guide Loading ────────────────────────────────────────────────

    @staticmethod
    def _load_guide(filename: str) -> str:
        """Load a guide file from llm_guides directory."""
        guides_dir = os.path.join(os.path.dirname(__file__), "..", "..", "llm_guides")
        filepath = os.path.join(guides_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"[Planner] Loaded guide: {filename} ({len(content)} chars)")
            return content
        print(f"[Planner] Guide not found: {filename}")
        return ""

    # ── Shot ID Generation ───────────────────────────────────────────

    @staticmethod
    def _make_shot_id(index: int, prefix: str = "shot") -> str:
        return f"{prefix}_{index:03d}"
