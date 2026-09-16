# Prompt enhancement review rubric

Review the saved source request, every recorded writer call, the planner response, and the final native prompt delivered at the generation handoff. Cite concrete text for each judgment. Do not infer rendered motion, identity, voice quality, lip sync, or visual continuity from prompt text alone.

## Ordinal scale

Use the same four-point scale for every criterion:

- **0 — Failed:** absent, contradicted, assigned incorrectly, or unusable.
- **1 — Weak:** partly present but materially unclear, inconsistent, or difficult to execute.
- **2 — Sound:** preserves the requirement and gives a coherent, feasible treatment with only minor weaknesses.
- **3 — Strong:** precise, motivated, easy to follow, and developed beyond the minimum without violating the request.

Record `not_applicable` when a criterion does not apply. Do not convert it to a numeric score. Scores summarize evidence; they do not override a critical failure.

## Text criteria

1. **Source fidelity:** Preserves the requested premise, events, ending, restrictions, tone, and priorities.
2. **Causal action and geography:** Makes positions, movement, cause and effect, and the order of events readable and physically achievable.
3. **Character distinction and ownership:** Keeps people distinct and preserves ownership of dialogue, references, objects, actions, and knowledge.
4. **Continuity and state:** Carries locations, entrances, exits, doors, props, injuries, clothing, knowledge, and other persistent state across beats and windows.
5. **Dialogue integrity and feasibility:** Preserves exact supplied lines verbatim and in order when required; otherwise writes only appropriate dialogue with clear speakers, reactions, and enough delivery time.
6. **Direction and camera motivation:** Uses framing, coverage, and camera movement to clarify story and performance while honoring continuous-shot or no-cut constraints.
7. **Creative development:** Adds specific, relevant behavior, escalation, atmosphere, and resolution without replacing the user's intent or inventing prohibited mechanisms.
8. **Model-contract correctness:** Produces valid H3-native structure, reference bindings, window count, timing, dialogue tags, and final handoff prompts without unexplained fallback or truncation.

Prompt length is not a criterion. More detail earns no credit unless it improves fidelity, clarity, feasibility, or direction. Repetition, ornamental prose, and instructions that compete for the same moment should lower the relevant score.

## Critical flags

Report each flag as present or absent and quote the evidence:

- Missing, altered, reordered, duplicated, or wrongly owned exact dialogue
- Missing required event or requested ending
- Invented prohibited speech, cut, mechanism, character, object, or effect
- Reference, character, voice, action, object, or knowledge assigned to the wrong owner
- Continuity reset, impossible transition, duplicate subject or prop, or unexplained disappearance
- Invalid H3 format, missing expected window, malformed dialogue tag, or broken reference binding
- Truncation, timeout, empty output, or a fallback that loses a required meaning or event
- Final handoff prompt materially loses or changes a correct intermediate decision

A result with a critical flag cannot be described as fully successful even if its ordinal average is high.

Record repairs, deterministic fallbacks, and missing telemetry as separate operational flags even when the final text is sound. A successful focused repair is not itself a semantic failure. Missing evidence limits the claims that can be made; it is not a zero score for an otherwise observable writing criterion.

## Evidence and comparison

For each case, report criterion scores with a short rationale and exact evidence, then list critical flags, warnings, fallbacks, repairs, and missing evidence. Compare writers using all completed and failed attempts; do not select only the best output. When preferences are close, state the uncertainty and identify the evidence that would resolve it.

Keep these claim types separate:

- **Text finding:** supported by the recorded request, trace, planner output, or final native prompt.
- **Rendered observation:** supported only by the named video, audio, frames, or transcript actually inspected.
- **Rendered hypothesis:** a prediction from prompt text that still requires a matched render.

Rendered review, when explicitly run later, should examine identity, action execution, object and scene continuity, sound and voice ownership, speech timing, lip sync, transitions, and the requested ending across the full clip. Static frames or transcripts may support a limited observation but cannot establish continuous motion or lip sync.
