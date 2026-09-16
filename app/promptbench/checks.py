"""Evidence-based mechanical checks; semantic quality requires a separate review."""
import re

REF_FIELDS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
              "overall_soundscape", "non_diegetic_music")
BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")


def trace_calls(entries):
    """Read initial pilot traces as well as the corrected single-record traces."""
    wrappers = []
    for index, entry in enumerate(entries[:-1]):
        child = entries[index + 1]
        if (entry.get("function") == "generate" and not entry.get("payload")
                and entry.get("status") == "complete" and child.get("payload")
                and child.get("function") == "generate_streaming"
                and entry.get("output") == child.get("output")):
            wrappers.append(entry.get("index"))
    calls = [entry for entry in entries if entry.get("payload")]
    undispatched = [entry for entry in entries if not entry.get("payload") and entry.get("index") not in wrappers]
    return calls, wrappers, undispatched


def prompts_from(result):
    prepared = result.get("prepared") or {}
    params = prepared.get("params") or {}
    plan = prepared.get("h3_window_plan") or params.get("h3_window_plan") or {}
    prompts = plan.get("window_prompts") or params.get("h3_window_prompts")
    if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
        return prompts
    prompt = params.get("prompt")
    return [prompt] if isinstance(prompt, str) and prompt.strip() else []


def evaluate(case, result):
    checks = []
    def add(name, status, evidence):
        checks.append({"check": name, "status": status, "evidence": evidence})
    prompts = prompts_from(result)
    add("completed", "pass" if result.get("status") == "complete" and prompts else "fail",
        result.get("error") or f"{len(prompts)} native prompts")
    prepared = result.get("prepared") or {}
    plan = prepared.get("h3_window_plan") or {}
    warnings = list(dict.fromkeys([*(prepared.get("enhancement_warnings") or []),
                                   *(plan.get("planning_warnings") or [])]))
    add("draft_review", "review" if warnings or prepared.get("diagnostics_scope") or "fallback" in str(plan.get("planned_by", "")) else "pass",
        {"warnings": warnings, "planned_by": plan.get("planned_by"),
         "coverage": prepared.get("diagnostics_scope"),
         "diagnostics": plan.get("planning_diagnostics") or []})
    expected = case.get("checks") or {}
    if "expected_windows" in expected:
        add("window_count", "pass" if len(prompts) == expected["expected_windows"] else "fail",
            {"expected": expected["expected_windows"], "actual": len(prompts)})
    fields = REF_FIELDS if "ref2va" in str(case["params"].get("model_type", "")) else BASE_FIELDS
    for i, prompt in enumerate(prompts, 1):
        positions = [re.search(rf"(?m)^\s*{field}\s*:", prompt) for field in fields]
        valid = all(positions) and [p.start() for p in positions] == sorted(p.start() for p in positions)
        add(f"window_{i}_format", "pass" if valid else "fail", {"required_fields": fields})
    # H3's documented speech spans, not quotations around visual phrases.
    spoken = [(index + 1, match.group(1).strip()) for index, prompt in enumerate(prompts)
              for match in re.finditer(r"<d>(.*?)</d>", prompt, re.S)]
    if expected.get("no_dialogue"):
        add("silent_scene", "fail" if spoken else "pass", {"spoken_spans": spoken})
    for index, line in enumerate(expected.get("exact_dialogue", []), 1):
        count = sum(text.count(line["text"]) for _, text in spoken)
        add(f"exact_dialogue_{index}", "pass" if count == 1 else "fail",
            {"speaker_to_review": line["speaker"], "text": line["text"], "occurrences_in_speech": count})
        add(f"exact_dialogue_{index}_owner", "review", "Verify voice ID and visible speaking face in native prompt; exact text alone cannot prove ownership.")
    entries = result.get("calls") or []
    calls, wrappers, undispatched = trace_calls(entries)
    add("fresh_llm_calls", "pass" if calls else "fail", {"calls": len(calls)})
    missing = [call.get("index") for call in calls if not call.get("payload") or not call.get("metrics")]
    add("trace_coverage", "review" if missing else "pass", {"missing_payload_or_metrics": missing})
    if undispatched:
        add("undispatched_attempts", "review", [{"index": entry.get("index"), "error": entry.get("error")} for entry in undispatched])
    for call in calls:
        metrics = call.get("metrics") or {}
        if metrics.get("truncated") or call.get("status") != "complete":
            add(f"call_{call.get('index')}_completion", "review",
                {"finish": metrics.get("finish_reason"), "error": call.get("error")})
    add("creative_quality", "review", case.get("review") or [])
    add("rendered_quality", "not_tested", "Enhancement only; no video was generated or assessed.")
    # No composite 'quality score': structural success is not creative acceptance.
    return {"checks": checks, "failure_count": sum(c["status"] == "fail" for c in checks),
            "review_count": sum(c["status"] == "review" for c in checks), "warnings": warnings,
            "native_prompts": prompts, "spoken_spans": spoken,
            "calls": len(calls), "trace_entries": len(entries), "wrapper_entries": wrappers,
            "token_totals": token_totals(calls)}


def token_totals(calls):
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "answer_tokens"):
        values = [(call.get("metrics") or {}).get(key) for call in calls]
        available = [value for value in values if isinstance(value, int)]
        result[key] = {"known_total": sum(available), "covered_calls": len(available), "total_calls": len(calls)}
    return result
