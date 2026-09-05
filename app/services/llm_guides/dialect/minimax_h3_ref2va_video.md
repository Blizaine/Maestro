MINIMAX H3 REF2VA RULES (apply to video_prompt):
- Describe the finished target clip in concise chronological prose using exactly these fields:
  subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape,
  and non_diegetic_music.
- Treat Maestro's ordered reference inventory as authoritative. Never invent or renumber
  <Subject N>, <Picture N>, <Video N>, or <Audio N> labels.
- <Subject N> is a stable visible identity. Identity references supply appearance rather than
  their source pose, background, framing, dialogue, action, or opening frame.
- Bind each voice to its visible subject: `<Audio N> is the voice-timbre reference for
  <Subject N> (Sx), guiding emotion and delivery.` Reuse the same local (Sx) beside that Subject's
  dialogue. Voice timbre is retained while the new speech uses the target scene's acoustics.
- Subject IDs and speaker IDs are independent. Keep Subject IDs fixed; assign (S1), (S2), etc.
  by first vocal-event order within the current generated clip.
- Preserve every supplied quoted line verbatim, once, in source order. Use direct dialogue syntax:
  `<Subject 2> (S1) says in the voice referenced from <Audio 1>,
  <d>[English] Exact words.</d>` Put only literal spoken words inside <d>.
- Keep scene description, camera direction, action, ambience, and delivery outside <d>. Do not
  turn those descriptions into narration or filler dialogue.
- `[Shot 1]` has no timestamp. Later cuts use `[Shot N] At MM:SS.mmm, ...`. Every shot must fit
  the requested clip duration and advance toward a concrete final state.
- For multi-clip work, include only the current clip's assigned events. Keep reference Subjects
  stable between clips and recalculate speaker IDs locally in each generated clip.
- Keep overall_soundscape to target-scene ambience and synchronized effects. Use
  non_diegetic_music: N/A unless music is requested.
- Prefer positive, performable visual direction and do not pad the prompt to a fixed word count.
