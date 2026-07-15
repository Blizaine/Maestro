VIDEO PROMPT (video_prompt) — for the LTX-2 video model:
The video generates from a START IMAGE + text. The start image has the environment and characters.
The video prompt describes ONLY what HAPPENS — actions, dialogue, camera movement.

DO NOT describe environment/setting — the start image handles that.
- BAD: "Close-up shot, industrial spaceship interior. The blonde woman looks at the console."
- GOOD: "The blonde woman looks at the console and presses the button."

CRITICAL RULES:
- Identify characters by appearance EVERY time: "the blonde woman", "the man in blue shirt"
  NEVER use "she", "he" alone — LTX does not know who "she" is.
- NEVER use: "continuing", "still", "repeats", "again", "as before" — LTX has no memory.
- Simple direct actions: "The blonde woman removes her coat" not elaborate descriptions.
- Dialogue MUST include a VOICE TAG. Create a fixed voice tag for each character and
  COPY-PASTE it identically in EVERY shot and EVERY window_prompt. The audio model
  generates each clip with ZERO memory — if the tag is missing or worded differently,
  the voice WILL change. Do NOT paraphrase, shorten, or vary the tag.

  VOICE TAG FORMAT (gender is REQUIRED — do not omit):
    "in a [pitch] [texture] [male|female] voice with a [specific regional accent]"

  GENDER IS NON-NEGOTIABLE: the audio model picks gender randomly when omitted.
  A character described as a "soft breathy Japanese-accented voice" will be
  generated as a woman in one clip and a man in the next, completely breaking
  voice identity. Always include "male" or "female" explicitly. For non-human
  characters (animals, creatures, robots), pick a gender that fits the role
  and use it consistently — the audio model still produces a male or female
  human voice and you need to lock one in.

  Example tags (pick one per character, use VERBATIM in every shot):
  - "in a deep raspy male voice with a New Jersey accent"
  - "in a low slow male voice with a Southern Texas drawl"
  - "in a high-pitched female voice with an upper-class London accent"
  - "in a soft breathy female Japanese-accented English voice"
  - "in a bright energetic male voice with a clipped British accent"
  - "in a warm rich female voice with a smooth mid-Atlantic accent"

  BAD — tag varies OR omits gender (voice changes / flips gender every clip):
    Shot 1: 'says in a confident Asian-accented voice, "..."'        ← no gender
    Shot 2: 'says in a rich throaty voice, "..."'                    ← gender varies
    Shot 3: 'says, "..."'                                            ← no tag at all
    Shot 4: 'says in a soft breathy Japanese-accented voice, "..."'  ← no gender; LTX picks random
  GOOD — identical tag in every shot, gender locked (voice stays consistent):
    Shot 1: 'says in a soft breathy female Japanese-accented English voice, "..."'
    Shot 5: 'says in a soft breathy female Japanese-accented English voice, "..."'
    Shot 9: 'says in a soft breathy female Japanese-accented English voice, "..."'
- One flowing paragraph, present tense. 80-150 words for 20s.
- Show emotion through PHYSICAL CUES: jaw tightens, tears forming. NOT "feeling sad".
- NEVER say montage, quick cuts, cut to.

BUILD ORDER: character state → action beats → camera → dialogue → ending beat.

DURATION AND WINDOWS:
- 2-4s = 1 beat. 5-9s = 2-3 beats. 10-20s = 3-5 beats.
- Over 20s = MUST use window_prompts, leave video_prompt as "".
  Each window is ~20s. A 30s scene needs 2 windows. A 45s scene needs 3.
  The video model generates each window separately with only visual overlap.
  If you put 30s of content in a single video_prompt, the model will loop
  the same actions because it can only generate ~20s per window.
  WRONG: 30s scene with video_prompt and empty window_prompts.
  RIGHT: 30s scene with video_prompt="" and 2 window_prompts.

TIMED DIRECTION — LTX responds very well to timestamp-based prompts:
For scenes with specific pacing needs (multiple distinct actions, location changes within
a shot, or precise choreography), use timed segments. Each segment tells LTX exactly
when to perform each action. Format: "(Xs-Ys): description"

TIMED DIRECTION EXAMPLE (20s shot):
(0-2s): The blonde woman stands up from the couch.

(2-7s): tracking shot. The blonde woman walks across the room into the kitchen.

(7-9s): The blonde woman opens the refrigerator door.

(9-13s): The blonde woman reaches in and grabs a gallon of milk.

(13-20s): The blonde woman opens the milk and takes a long drink. Camera holds on her.

WHEN TO USE TIMED DIRECTION:
- Scenes with 4+ distinct action beats that need precise pacing
- Scenes where characters move between areas (walk somewhere, then do something)
- Scenes with a mix of fast and slow actions
- When you need specific timing for dialogue delivery

WHEN NOT TO USE TIMED DIRECTION:
- Simple scenes with 1-2 actions (just write a paragraph)
- Atmospheric/mood shots with no specific choreography
- Very short clips (under 5s) — just one beat, no timing needed

Each timed segment should have its own line with a blank line between segments.
The timestamps must cover the full duration of the shot with no gaps.

MODE AWARENESS:
- With start image: don't re-describe the source, focus on what CHANGES.
- With audio: keep prompts leaner, let audio drive timing. Reduce visual complexity during speech.

CAMERA: use concrete verbs (pushes in, tracks beside, circles slowly).
Avoid vague: "dynamic cinematic camera", "dramatic camera work".

THE STILLNESS TRAP — words that freeze the whole video:
LTX reads stillness words as "nothing in the scene moves" — including lips
and breathing. "Static hold", "static shot", "still", "frozen",
"motionless", "holds perfectly still", "barely moves" can each produce a
FREEZE FRAME with no animation at all. NEVER use them.
- For a calm, locked-down shot: restrain the CAMERA only ("camera locked
  in place", "camera holds on her face") and in the SAME sentence give the
  subject explicit continuous motion (speaking, breathing, blinking,
  shifting weight).
- Never stack restraint cues — "slowly" + "barely" + "subtle" in one
  prompt compounds into zero motion. Every prompt names at least one
  thing that KEEPS MOVING for the full shot.

GOOD PARAGRAPH STYLE (simple scene, 2-3 beats):
"The blonde woman whispers 'I want to show you something.' The blonde woman stands up,
the man in blue shirt watches. The blonde woman removes coat. The man in blue shirt stares.
The blonde woman sits back down and says 'Come here.' Camera slowly pushes in."

GOOD TIMED STYLE (complex scene, 4+ beats):
(0-3s): The blonde woman looks at the man in blue shirt and whispers 'I want to show you something.'

(3-8s): The blonde woman stands up slowly. The man in blue shirt watches, leaning forward.

(8-14s): The blonde woman removes her coat, letting it fall. The man in blue shirt stares.

(14-20s): The blonde woman sits back down on the couch and says 'Come here.' Camera slowly pushes in.

BAD: "Continuing the scene, she looks flushed. 'I want to show you something,' she repeats."
(Problems: "she" without ID, "continuing", "repeats")

WINDOW PROMPTS — for scenes over 20s:
- Each window is STANDALONE — no "continuing" or references to other windows.
- Re-identify all characters in every window.
- Each window = full paragraph (80-150 words).

BAD WINDOWS: "They keep talking" or "She stands up" (no character ID, too short).
