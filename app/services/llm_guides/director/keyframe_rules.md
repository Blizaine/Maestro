KEYFRAMES — visual reference images the video model animates TOWARD.

THE DECISION RULE — ask yourself: "Can the video model figure this out from just the start image and prompt?"
- YES → no keyframe needed. The video model handles all character animation, expressions,
  gestures, leaning, turning, walking, sitting down, and dialogue delivery.
- NO → add a keyframe. The video model CANNOT invent things it has never seen.

WHEN KEYFRAMES ARE NEEDED (video model lacks visual information):
- A NEW CHARACTER appears who isn't in the start image (model doesn't know what they look like)
  Example: close-up of a woman → camera pulls back to reveal a man. Keyframe shows the man.
- A SPECIFIC VISUAL RESULT that the model can't infer from the prompt alone
  Example: object transformation, specific destruction, novel pose the model hasn't seen.
- RAPID COMPOSITION CHANGES within a single shot that need precise visual control
  Example: 4-5 keyframes in a 20s scene to guide dramatic movement or quick angle cuts.
- A SCENE-ENDING STATE that differs dramatically from the start
  Example: start=tidy room, end=room destroyed. Keyframe shows the destruction.

WHEN KEYFRAMES ARE NOT NEEDED (video model handles these fine):
- Character animation: leaning in, turning, gesturing, standing up, sitting down
- Facial expressions: smiling, frowning, crying, looking surprised
- Dialogue delivery: talking, whispering, shouting
- Camera movement: push-in, dolly, tracking, orbit (described in video_prompt)
- Long dialogue scenes: even 60s with 3 windows needs ZERO keyframes if it's just people talking
- Simple physical interactions: hugging, handshake, picking up objects

KEYFRAME PROMPT FORMAT — READ CAREFULLY, KEYFRAMES ARE NOT IMAGE PROMPTS:

A keyframe EDITS the start image (or the previous keyframe). It is NOT a new
scene. The format and verbosity are COMPLETELY DIFFERENT from image_prompt.

HARD RULES FOR EVERY KEYFRAME (break any of these and the result is wrong):

- START WITH "same scene," as a prefix. This is the single most important
  signal that tells the image model "don't regenerate anything — just edit".
- MAX 2 short sentences. One sentence is better. Target ~20 words, not 80.
- Describe ONLY the visual DIFFERENCE from the previous frame.
- Use PLAIN DESCRIPTORS, not narrative verbs. Keyframes are FROZEN STATES,
  not actions in progress.
    BAD (narrative): "the man is reaching toward the door, hand outstretched"
    GOOD (state):   "same scene, the man's hand is now on the doorknob"
  Narrative verbs ("is reaching", "leans toward", "begins to") tell the model
  to animate motion, which makes it regenerate the whole scene.
- Use PLAIN NOUNS, not flowery sensory language. Keyframes are visual facts,
  not literary descriptions.
    BAD (flowery): "warm light cascades over her features as she gazes longingly"
    GOOD (fact):   "same scene, close-up of her face. soft expression."
  Words like "cascades", "gleams", "flows", "drapes", "captivated" push the
  model to rewrite the image. State the visible thing in plain language and stop.
- Do NOT use "create new scene" — the scene already exists.
- Do NOT use "close-up shot of the reference image" or similar re-framing
  preambles. Use "same scene, close-up of..." instead.
- Do NOT end with "Preserve character identity, attire, and body attributes
  from the reference image." — that ending is for image_prompts ONLY. A
  keyframe is not an image_prompt and must NEVER include it.
- Do NOT re-describe the setting, lighting, clothing, or character appearance
  the previous frame already shows.
- Do NOT add atmospheric details ("dimly lit", "warm golden glow",
  "cinematic intensity") — these belong in video_prompt, not keyframes.
- Do NOT narrate the action that led to the state — keyframes are FROZEN
  visual differences, not mini-paragraphs about what happened.

STRUCTURE EVERY KEYFRAME LIKE ONE OF THESE TEMPLATES:
- "same scene, [framing]. [subject] [new visible state]."
- "same scene, [subject] is now [new state/position]."
- "same scene, close-up of [specific body part / object]. [visible change]."

BAD vs GOOD — INTERNALIZE THESE PATTERNS:

BAD (too verbose, rewrites the scene, uses image_prompt ending):
"close up shot of the reference image, the man entering through the doorway,
his expression determined, the afternoon light catching his face dramatically.
Preserve character identity, attire, body attributes, and the art style of the reference image."

GOOD (concise, just the essential visual difference, no image_prompt ending):
"same scene, the man is now standing just inside the doorway."

MORE BAD vs GOOD PAIRS:

BAD (narrative verb + flowery sensory language, rewrites the scene):
"close-up of the woman's eyes widening as she leans forward in surprise, the
warm glow of the desk lamp catching her features dramatically."
GOOD (plain state + plain noun):
"same scene, close-up of the woman's face. eyes wide, mouth slightly open."

BAD: "wide angle shot, the man stands at the window, arms folded, gazing
out at the city lights with a contemplative expression."
GOOD: "same scene, the man is now at the window, arms folded."

BAD: "close-up of the cup of coffee, steam rising gently as the woman
reaches her hand toward it through soft morning light."
GOOD: "same scene, close-up of the woman's hand on the coffee cup."

EXAMPLES IN SCENE CONTEXT:

Scene: camera reveals a new character
- Start image: "create new scene, coffee shop. Close-up of the woman with red hair seated at a small table, warm window light."
- Keyframe: "same scene, wide shot pulled back. the man in a grey blazer is now visible across the table."
- Video prompt: "The camera slowly pulls back from a close-up of the woman with red hair, revealing the man in a grey blazer across the table..."

Scene: transformation / destruction
- Start image: "create new scene, dining room. The vase sits on the table, fresh flowers inside."
- Keyframe: "same scene, the vase is now shattered on the floor, water and petals spreading."
- Video prompt: "The vase wobbles as the cat brushes against it, then tips and falls. It hits the floor and shatters..."

Scene: long dialogue (NO keyframes needed)
- Start image: "create new scene, living room. The woman and man sit on opposite ends of the couch, tense expressions."
- Keyframes: [] (EMPTY — the video model handles all the talking, gestures, and expressions)
- Window prompts: ["Window 1: The woman turns to the man and says...", "Window 2: ...", "Window 3: ..."]

REMEMBER: when you write a keyframe, you are editing ONE small detail in an
existing frame. You are NOT producing a new image prompt. Keep it short,
keep it visual, and never append the image_prompt identity-preservation ending.
