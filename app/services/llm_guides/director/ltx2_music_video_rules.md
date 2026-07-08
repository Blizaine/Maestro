VIDEO MODEL CAPABILITIES — the LTX-2 video generator is very capable:
- It generates up to 20 seconds of continuous video per clip.
- It handles multiple characters, performance, dancing, movement, all in ONE shot.
- It can show a performer singing, moving, gesturing, interacting — all in one clip.
- One image prompt + one long video prompt can cover an entire verse or chorus.
- Scale action/performance beats to clip duration — more beats for longer clips.
- Short clips (5-10s) are fine when they serve the story — a quick cutaway or reaction shot.
- Longer clips (15-20s) are great for sustained performance, conversations, or build-ups.
- Choose duration based on what the content needs, not a fixed rule.

EXTENDED SCENES (sliding window) — for scenes longer than 20 seconds:
- If a scene genuinely needs 30-60 seconds of continuous video (e.g. a long conversation,
  a building performance, an extended action sequence), use sliding window mode.
- Set duration_sec to the full length (up to 60s) and provide "window_prompts" — an array
  of 2-3 prompts, one per ~20s window. Each window prompt continues from where the last ended.
- Window prompts share the same scene/setting but describe what happens in THAT window's timeframe.
- Only use this for scenes that truly need extended duration — not every scene needs it.
- If window_prompts is provided, video_prompt should be the first window's prompt.

VIDEO PROMPT (video_prompt) — written for LTX-2 with AUDIO-DRIVEN generation:
These clips are generated WITH MUSIC AUDIO. The audio drives the energy, movement, and timing.
Your prompt sets the VIBE and SUBJECT — the music handles the rest.

CRITICAL — MUSIC VIDEO PROMPTS MUST BE SHORT AND ENERGETIC:
- Keep prompts SHORT: 15-40 words ideal. DO NOT over-describe.
- Use KEYWORDS and VIBES, not detailed stage directions.
- The more detail you add, the SLOWER and more STATIC the video becomes.
- Let the music drive the energy — don't try to choreograph every movement.
- Identify characters briefly by appearance: "the woman in red dress" not elaborate descriptions.
- NEVER use continuity words: "continuing", "still", "repeats", "again".

GOOD MUSIC VIDEO PROMPTS (short, energetic):
- "Dogs playing in a band. Singing. Dancing. Dynamic camera movement. Lights. Smoke. Atmospheric."
- "Woman in red dress singing on neon-lit stage. Crowd energy. Strobe lights. Handheld camera."
- "Man in leather jacket playing guitar. Close-up. Sweat. Stage lights. Raw energy."
- "Two dancers in a warehouse. Dramatic shadows. Spinning. Low angle. Dust in the air."
- "Singer in spotlight. Emotional performance. Tears. Slow push in. Dark background."

BAD MUSIC VIDEO PROMPTS (too detailed — makes video slow and boring):
- "Wide shot of a rock concert stage bathed in red and white lights. The grey shih tzu in the
  black leather jacket stands center stage holding a microphone. The light brown shih tzu with
  guitar stands on the left tuning an electric guitar. The brown shih tzu behind drums sits at
  a kit with a flame logo. Thick smoke swirls around their paws. The camera pans slowly from
  left to right capturing the tension." (TOO LONG — LTX tries to compose every detail, kills energy)

STYLE KEYWORDS TO USE:
- Energy: dynamic, energetic, intense, raw, explosive, dreamy, ethereal, moody
- Camera: handheld, tracking, spinning, low angle, close-up, wide shot, aerial
- Atmosphere: smoke, lights, strobes, neon, silhouette, shadows, dust, rain, sparks
- Performance: singing, dancing, playing, performing, headbanging, jumping, swaying

THE STILLNESS TRAP — words that freeze the whole video:
- LTX reads stillness words as "nothing in the scene moves" — INCLUDING
  the singer's lips. "Static hold", "static shot", "still", "frozen",
  "motionless", "holds perfectly still", "barely moves" can each produce
  a FREEZE FRAME with zero lip-sync. NEVER use them.
- For a calm, intimate shot: restrain the CAMERA only and give the
  performer explicit continuous motion in the same sentence.
  WRONG: "Extreme close-up. Static hold. His lips slowly articulate the
         lyrics, his head barely moves."
  RIGHT: "Extreme close-up, camera locked in place. He sings the lyrics,
         lips and jaw moving clearly with every word, chest rising as he
         breathes, eyes blinking softly."
- Never stack restraint cues ("slowly" + "barely" + "subtle" in one
  prompt compounds into no motion at all). Every vocal prompt names at
  least one thing that KEEPS MOVING for the full shot — the mouth first.

PERFORMER / VOCALIST — SHOW WHO IS SINGING (applies to EVERY clip):
- A music video is about the ARTIST performing. When a clip has lyrics (vocals),
  the image_prompt AND video_prompt MUST show a performer delivering them — a
  singer singing, a rapper rapping, or a vocalist speaking/lip-syncing to camera.
  Do NOT render empty scenery while a vocal plays; the performer is the subject.
- USE THE LITERAL WORDS. On vocal clips the video_prompt must contain an
  explicit vocal-performance verb attached to the performer: "singing",
  "rapping", "lip-syncing", or "singing the lyrics". The soundtrack alone
  sometimes animates the mouth, but the written word is what makes it happen
  EVERY time. "She performs on stage" is NOT enough — write "she sings
  passionately into the microphone".
- DELIBERATE NON-SINGING SHOTS ARE ALLOWED — SPARINGLY. At most about 1 in 4
  vocal clips may intentionally show the performer NOT singing while the
  vocal continues (cinematic b-roll: walking away from camera, staring out a
  window, a slow orbit around them standing still). When you choose this,
  SAY IT EXPLICITLY in the video_prompt — "she does not sing; her mouth stays
  closed as the song continues over the shot" — otherwise the audio will
  half-animate the mouth and the shot reads as a glitch. Never make two
  non-singing vocal clips in a row.
- If NO reference photo or named performer is given, INVENT one that fits the
  song's vibe and keep it CONSISTENT across every clip (the same described
  artist) — e.g. "a stylish female singer in a sequined jacket", "a young male
  rapper in a hoodie". Describe by appearance, never by name.
- MULTIPLE VOICES: if a clip's context names a dominant speaker/voice (e.g.
  "speaker: SPEAKER_01" or a duet), attribute the singing to the matching
  performer and keep each voice's performer consistent across the whole video.
  With no speaker info, the main performer sings everything.
- Match delivery to the vocal: singing → microphone, expressive face; rapping →
  rhythmic hand movement, head bob, attitude; spoken → direct-to-camera address.
- The clip context tells you each clip's vocal content (lyrics: "..." or
  instrumental). Show the performer performing THAT line on vocal clips.
- Instrumental clips (no lyrics): musicians playing, dancers, or atmosphere — the
  performer may recede, but keep the energy.
