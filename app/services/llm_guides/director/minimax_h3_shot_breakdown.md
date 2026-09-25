VIDEO PROMPT (video_prompt) — for MiniMax H3:

MiniMax H3 generates synchronized picture and stereo sound. A Director shot may
be text-only (FL2VA) or use soft image/audio references (Ref2VA); neither path
guarantees a fixed start frame. Every video prompt therefore stands on its own.

SELF-CONTAINED SHOT RULES:
- Describe the finished target shot, not instructions to copy, animate, replace,
  or edit a reference.
- Include the setting, composition, every visible subject, identity/appearance,
  wardrobe, action, camera behavior, lighting, dialogue, ambience, effects, and
  music that must exist in the result.
- Follow the model-aware CHARACTER NAMING block in the surrounding Director
  instructions. In prompt-only/direct-reference mode, preserve every recognizable
  proper identity and its series, film, franchise, or performer exactly as
  supplied and pair reference labels with useful visible traits. When generated
  shot images are enabled, follow the supplied name-to-description conversion
  instead so the image and video prompts stay aligned.
- Do not invent names, dialogue, franchises, or scene details absent from the
  screenplay or concept.
- References are guidance. Do not emit guessed <Picture N>, <Video N>, or
  <Audio N> tags; Maestro maps the per-shot references after planning.

CONTINUITY WITHOUT A FIXED START IMAGE:
- Treat wardrobe and blocking as explicit shot state. For every visible person,
  repeat the complete head-to-toe clothing and exact first-frame position:
  screen-left/center/right, depth, pose, facing direction, and nearby props.
- State the same opening composition in video_prompt; names alone do not carry
  appearance, clothing, or position into a new text-only generation.
- Give each uninterrupted place/time one continuity_group. Before an ordinary
  same-scene cut, visibly move each person into the next shot's opening position.
- Use continuity_strategy=extend_previous only for a literal same-composition
  continuation that should inherit the preceding generated final frame. Use
  continuity_strategy=continuous for normal cuts within the same scene.

CONTEXT-IR FORMAT:
- Structure video_prompt with exactly these labeled sections:
  integrated_multimodal_description, overall_soundscape, non_diegetic_music.
- Begin integrated_multimodal_description with [Shot 1] and no timestamp, then
  narrate visible action, camera, dialogue, and synchronized sound in
  chronological order. Later cuts begin [Shot N] At MM:SS.mmm.
- Give each speaking person a stable ID such as (S1) or (S2).
- When already numbered people speak or sing together, use a compound ID such
  as (S1,S2). Characters who never vocalize receive no speaker ID.
- For speech generated from the prompt, use <d>[Language] Exact words</d>,
  where Language is the actual language of that line. Speaker identity,
  action, delivery, and voice are outside the dialogue tag. Preserve scripted
  dialogue verbatim and never translate it.
- When no supplied driving audio owns the voices, every structured
  dialogue_beats entry must appear exactly once in video_prompt. Never leave
  actual generated spoken words only in the JSON field.
- For voiceover, use the exact phrase "says in an off-screen voiceover" and
  immediately state that the corresponding on-screen character's lips remain
  completely closed.
- Use <scenetrans> at both connecting points only when the same line genuinely
  crosses a cut, and <cutoff> only when speech is intentionally truncated by
  the video ending.
- Preserve visible signs, labels, banners, subtitles, and other on-screen text
  verbatim in English double quotation marks; never translate it.
- When supplied driving audio owns the voices, transcript beats are timing and
  acting metadata only: do not copy, quote, tag, or request their words in
  video_prompt. Keep actions, speaker framing, lip movement, and reactions
  synchronized to the supplied audio without duplicating or replacing it.
- When neither dialogue nor supplied vocals are requested, explicitly keep
  mouths closed and omit voices or speech-like sounds. Explicitly forbid
  muttering, murmuring, improvised words, and gibberish; never fill unused time
  with invented speech.
- After the last prompt-generated spoken line, use visible reactions or motion
  for remaining time and state that characters remain silent with mouths
  closed. For supplied driving audio, keep any ongoing vocal performance
  synchronized to that audio.
- overall_soundscape contains ambience, practical effects, and non-verbal human
  sounds. Do not repeat dialogue there.
- non_diegetic_music is audience-only music. Use N/A unless music is requested
  or the shot follows supplied driving music.

TIMING:
- Keep actions and prompt-generated dialogue realistic for the requested
  duration. Generated spoken text should target 2.8 words per second during
  speech and never exceed 3 words per second across all speakers. With
  supplied driving audio, follow its timing instead. Leave time for requested
  action and pauses.
- H3 renders bounded native shots. Do not put LTX sliding-window commands,
  references to a previous shot, or IC-LoRA ``Shot N (Camera, Xs)`` trigger
  syntax inside video_prompt. Use the required structured continuity fields
  only for Director's planning and handoff logic.
- For supplied driving audio, describe the visible performance, lip movement,
  rhythm, and action that synchronize to it; do not transcribe or replace its
  audible content.

Do not include negative prompts, model names, LoRA names, technical settings,
reference-index guesses, or explanatory prose in video_prompt.
