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
- "Woman in red dress singing on neon-lit stage. Crowd energy. Strobe lights. Handheld camera."
- "Close-up of the guitarist's hands moving over the strings. Handheld camera. Stage lights. Raw energy."
- "Two dancers in a warehouse. Dramatic shadows. Spinning. Low angle. Dust in the air."
- "Singer in spotlight. Emotional performance. Tears. Slow push in. Dark background."
- "Empty road at night, headlights cutting through rain. Reflections ripple across the asphalt. Aerial camera."

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

PERFORMER ROLES — FOLLOW THE CURRENT SHOT:
- The Scene Concept and each shot's assigned subjects decide who appears. The
  soundtrack does not require a singer or any other performer on screen.
- When a visible subject is explicitly assigned as a vocalist, use a clear
  singing/rapping/lip-sync action only while their own vocal part is audible.
  Keep their lips relaxed and closed during instrumental gaps. Do not guess a
  vocalist from the soundtrack, camera focus, or a generic music-video concept.
- An instrumentalist assigned to the shot plays with relaxed closed lips and
  active hands/body. A wind or brass player uses the instrument's normal
  embouchure. Do not mention an off-screen singer in an instrument cutaway.
- In a group shot, only explicitly assigned vocalists sing. Backing vocals,
  duets, and an instrumentalist who also sings are valid when the user assigns
  them. Do not invent extra singers to fill the frame.
- Preserve requested cheers, shouts, dance, and other non-song actions without
  transferring the soundtrack's vocal part to that person.
- Keep each visible person's identity and assigned role consistent across
  cuts. If a voice is mapped to a person, match only that person's visible
  mouth movement to their own audible part.
- The clip context describes the audio. It does not require adding a person to
  represent that audio. A narrative, dance, or environment shot can remain so.
- For instrumental intervals, follow the shot concept: a visible assigned
  musician may play, dancers may move, or the scene may show atmosphere alone.
