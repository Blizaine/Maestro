# H3 reference conversation enhancement

Validated locally on 17 September 2026; included in v2.2.3.

## Reproduction and diagnosis

A queued reference conversation returned the generic invalid-H3-draft warning.
Its source contained six exact lines (61 spoken words), while its saved settings
requested one 345-frame, 24 fps window (14.375 seconds). The maximum speech rate
allows 43 words in that window; the source needs additional duration.

Independent bugs obscured that conflict. A line following a named reaction was
missed or lost its chronological event anchor. Queued Enhance supplied a typed
reference inventory that the single-window writer's subject parser did not
recognize. The writer could also invent voice media for picture-only references.
Retrying immutable dialogue could not resolve the duration conflict.

The two-window test exposed separate scheduling problems: global production
directions consumed event slots, and a complete supplied conversation was handed
back to the writer for event-ID scheduling. Finally, decorative camera prose
could inflate the minimum time for a silent facial reaction, leaving exact speech
too little time despite a feasible overall duration.

## Changes and automated validation

- Reject infeasible supplied dialogue before model loading, with actual word and
  duration limits and an actionable duration suggestion. Preserve the source.
- Recognize unambiguous same-paragraph action-beat attribution, preserving its
  speaker and chronological speech event. Headings, written signs, thoughts,
  and ambiguous plural reactions remain non-dialogue.
- Use the actual ordered upload inventory for queued and interactive enhancement.
  Subject numbering and first-vocal-event speaker numbering remain independent.
  Keep voice, scene, style, and composition reference roles distinct.
- Allocate fully supplied, anchored exchanges before requesting cinematic
  treatment and camera direction. Open requests for additional dialogue still
  use story writing. Do not convert generic listener/pacing/prop-continuity rules
  into separate physical events.
- Preserve exact speech time and let stationary reactions use remaining time.
  Minimum action allowances use their source event, not metaphorical camera prose.
  Travel, contact, impact, and explicitly timed holds retain their protection.
- Remove outer Markdown code fences from native enhancement output.
- Keep spoken reports of an arrival out of visible entrance staging. Preserve
  physical descriptions such as motion that suggests activity, and muffled
  thumps, without mistaking them for speech or assigning prop motion to a person.

448 Python regressions passed, covering the new admission, ownership, timing,
and reference cases plus existing H3 planning, camera, dialogue, continuation,
and runtime tests. The actual rejected API request also returned the specific
61-word/43-word duration explanation without loading the LLM.

## Recorded local writer checks

The cases use the same two actual reference images and upload ordering: Sam is
Subject 1; Alex is Subject 2 and speaks first. The short case uses the first two
lines in one window. The complete case preserves all six lines across 672 frames
(28 seconds), using 345-frame windows with 18-frame continuation overlap.

Writers were the installed Qwen3.8 27B and Gemma 4 E4B GGUF models, run through
the production queued-enhancement preparation path. Each table entry is one
attempt; times include writer loading and preparation. These are development
iterations, not repeated performance benchmarks or independent holdouts.

| Candidate | Case | Qwen | Gemma |
| --- | --- | --- | --- |
| Ledger v90 | Short | 21.7 s, 1 call, no warning | 23.3 s, 2 calls, reference warning |
| Ledger v90 | Complete | 130.8 s, 4 calls, schedule and timing warnings | 49.3 s, 6 calls, schedule/camera/timing warnings |
| Ledger v91 | Short | 19.1 s, 1 call, no warning | 8.6 s, 1 call, no warning |
| Ledger v91 | Complete | 87.6 s, 3 calls, timing warning | 43.1 s, 4 calls, schedule warning |
| Ledger v92 | Complete | 66.9 s, 3 calls, timing warning | 23.6 s, 3 calls, no warning |
| Ledger v93 | Complete | 71.7 s, 3 calls, timing warning | 22.6 s, 3 calls, no warning |
| Ledger v94 | Complete | 66.9 s, 3 calls, no warning | Not rerun |

The final two small compiler corrections were validated by replaying both
complete v93 writer traces and the v94 Qwen trace through v96. All passed
without warnings or additional writer calls. These are CPU replays of saved
responses, not fresh v96 writer runs. All six lines remain exact and in order;
each retains its intended named reference owner. The closed-box rule and
final silent reaction remain present. The Qwen short draft had a camera weakness
(facing away despite close-up facial direction); structural acceptance alone does
not establish successful visual staging.

Final text review found usable speaker-oriented coverage and the required silent
ending. Some direction remains wordy or over-specified, including tight framing
that also asks to keep the box visible. This work fixes invalid enhancement and
compiler corruption; it does not establish that every camera suggestion is ideal.

Private inputs, complete traces, hashes, reports, and replays are retained under
the ignored `.codex-tmp/promptbench/reference-dialogue-20260917*` paths. No videos
were generated for this audit. Rendered identity, voice, lip sync, and motion
quality remain for human review; text validation does not establish them.
