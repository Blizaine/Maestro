LTX-2 / LTX-2.3 RENDERER GUIDE

PURPOSE
This guide defines how a renderer converts a structured ShotPlan into an LTX-ready prompt. It is not a storytelling guide. Story structure belongs in the planner. This guide is for writing model-facing prompts that are chronological, filmable, mode-aware, and efficient.

PRIMARY RULE
Write prompts as one flowing cinematic paragraph in present tense. Keep the language literal, visual, and chronological. Think like a cinematographer and performance director describing exactly what happens in the shot.

SHARED CORE RULES
- One single paragraph
- Present tense
- Clear chronological order
- Explicit camera language
- Visible physical emotion cues, not abstract emotional labels
- Dialogue in quotation marks
- Strong final beat or image
- Avoid montage language for single-shot prompts
- Keep scene complexity manageable
- Match detail density to runtime

SHOT BUILD ORDER
Build prompts in this order:
1. Shot type and visual style
2. Spatial setup and environment
3. Character appearance and visible emotional state
4. Action beats in order
5. Camera behavior tied to action
6. Audio or dialogue
7. Ending beat

DURATION RULES
0 to 4 seconds:
- one clear action beat
- one camera idea
- one final image

5 to 9 seconds:
- two to three beats
- simple progression or reveal
- short ending beat

10 to 20 seconds:
- three to five beats
- opening state -> first motion -> reveal/change -> reaction -> ending beat

20 seconds or extension workflows:
- continuation-focused
- preserve continuity
- do not reset the scene unless intentionally required

MODE RULES

TEXT-TO-VIDEO
Use the fullest description here because the model must invent the whole shot.
Include:
- framing
- setting
- lighting
- character appearance
- action beats
- camera movement
- audio design
Good for fresh scene creation and high-invention shots.

IMAGE-TO-VIDEO
Do not re-describe the entire source image.
Focus on:
- what changes after the starting frame
- who moves first
- what expression changes
- what object enters play
- how the camera reframes or follows
- what sound begins
Favor motion and behavior over static description.

AUDIO-TO-VIDEO
Keep prompts leaner so the audio can drive timing and lip sync.
Focus on:
- staging
- physical performance
- camera framing
- emotional behavior
- environment
Avoid overloading important speech moments with too many extra visual events.

RETAKE
Preserve the established shot identity.
Focus on:
- the corrected motion, expression, timing, or reveal
- continuity with the existing scene
Do not rewrite the entire shot unless needed.

EXTEND
Continue from the previous state.
Focus on:
- the next beat
- continuity of pose, emotion, camera logic, and atmosphere
Avoid resetting the setup or reintroducing the shot from scratch.

DIALOGUE RULES
- Put spoken lines in quotation marks
- Keep most lines short
- Break long speech into smaller phrases
- Pair each line with visible acting cues
- Add delivery cues only when useful
- During lip-sync-critical moments, reduce competing visual complexity

Good pattern:
[action or expression] [delivery cue] "spoken line" [reaction or camera response]

SPATIAL RULES
When relevant, specify:
- left/right
- foreground/background
- seated beside / facing / turning toward
- object position relative to body
- reveal path into frame

AUDIO RULES
Treat sound as part of the shot design.
Include when relevant:
- ambience
- room tone
- magical hum
- distant traffic
- wind
- crowd noise
- vocal quality
- silence or pause

CAMERA RULES
Use concrete verbs:
- pushes in
- pulls back
- tilts down
- pans left
- reframes tighter
- holds still
- tracks beside
- circles slowly

Avoid vague phrases like:
- dynamic cinematic camera
- dramatic camera work
- cool angles

ANTI-PATTERNS
Avoid:
- abstract emotion without visible behavior
- i2v prompts that restate the full source image
- a2v prompts overloaded with simultaneous visual events
- too-short prompts for long clips
- hard cuts inside a single continuous-shot prompt
- long uninterrupted monologues with no acting beats
- conflicting lighting or blocking logic
- too many characters doing too much at once

COMPRESSION RULES
Trim:
- repeated adjectives
- redundant wardrobe restatement
- duplicate environment details
- extra camera adjectives that do not change the move
Preserve:
- action order
- camera intent
- dialogue wording
- ending beat
- continuity-critical details

QUALITY CHECK
Before finalizing, verify:
- one paragraph
- present tense
- chronological
- mode-aware
- enough beats for duration
- explicit camera language
- physical emotional cues
- dialogue in quotes if present
- strong ending beat
- no major anti-patterns