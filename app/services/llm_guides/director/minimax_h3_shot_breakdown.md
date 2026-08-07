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
- Begin integrated_multimodal_description with [Shot 1] and narrate visible
  action, camera, dialogue, and synchronized sound in chronological order.
- Give each speaking person a stable ID such as (S1) or (S2).
- Literal speech uses <d>[English] Exact words.</d> (change the language tag
  when requested). Speaker identity, action, delivery, and voice are outside
  the dialogue tag. Preserve scripted dialogue verbatim.
- Every structured dialogue_beats entry must also appear exactly once in
  video_prompt. Never leave the actual spoken words only in the JSON field.
- When no dialogue is requested, explicitly keep mouths closed and omit voices
  or speech-like sounds. Explicitly forbid muttering, murmuring, improvised
  words, and gibberish; never fill unused time with invented speech.
- After the last spoken line, use visible reactions or motion for remaining
  time and state that characters remain silent with mouths closed.
- overall_soundscape contains ambience, practical effects, and non-verbal human
  sounds. Do not repeat dialogue there.
- non_diegetic_music is audience-only music. Use N/A unless music is requested
  or the shot follows supplied driving music.

TIMING:
- Keep actions and dialogue realistic for the requested duration. Spoken text
  should generally stay at or below about two words per second.
- H3 renders bounded native shots. Do not put LTX sliding-window commands,
  references to a previous shot, or IC-LoRA ``Shot N (Camera, Xs)`` trigger
  syntax inside video_prompt. Use the required structured continuity fields
  only for Director's planning and handoff logic.
- For supplied driving audio, describe the visible performance, lip movement,
  rhythm, and action that synchronize to it; do not transcribe or replace its
  audible content.

Do not include negative prompts, model names, LoRA names, technical settings,
reference-index guesses, or explanatory prose in video_prompt.
