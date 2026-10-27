# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json

HY_NOMAS_INFOS = """### What it makes
Long, consistent 24 fps videos of a world described as a sequence of timed segments: each segment is generated after the previous ones and
remembers them, so places, characters and objects stay the same while the scene evolves. The model is distilled: 8 steps, no guidance.

### Prompt
- **World description (recommended)**: a JSON object with the style, the perspective, one main subject and the timed segments, each with
  what changes in the scene and what the subject does. The World History Duration follows the segments' timestamps (up to 60 seconds).
- **Plain text**: one segment of the selected World History Duration. The prompt enhancer can turn a plain idea into a world description.
- **Exploration instructions**: lines such as `[Forward]`, `[Look_Left]` or `[Event] a deer crosses the trail` after the description
  walk through the world as you would in the explorer, each line one move (see the prompt help).

### What gets saved
The **Generate** choice of the General tab decides what a generation keeps:
- **World Only** (the default): the explorable world, in the **Worlds** gallery. Its video is not made, which saves the decoding time;
  you see the world when you explore it.
- **World and its Video**: the world, plus its video in the Video gallery.
- **Video Only**: just the video, as a normal video model would; there is nothing to explore afterwards.

### Worlds gallery and exploration
Every world you generate lands in the **Worlds** gallery. Select it and press **Explore**: the world opens full screen and continues
from its last frame as you play. Hold **W A S D** to walk (the keys at those places on other layouts, e.g. **Z Q S D** on AZERTY: the
viewer shows your keyboard's letters) and the **arrows** to look around. Each move adds a short stretch of video (0.8 s by default,
longer moves in the viewer's Move menu) and takes a few seconds to generate; while you hold a key, the next move starts as soon as a
clip arrives. The viewer's **Quality** menu trades detail for speed: **Fast** (the default) uses 4 steps per move, about a third faster
than **Best**, the model's 8 steps; **Balanced** uses 6. Press Esc to leave: the next exploration starts again from the stored world.

**Memory**: how much of what you have already seen each move takes into account. The viewer's **Memory** menu counts it in moves
of the length chosen in **Move**, from 2 to 29 moves of 0.8 s (1.5 to 24 s of video); open the menu to see the seconds and the memory
each choice needs. The world's own video counts as explored: a 10 s world
already fills 10 s of a 24 s memory before your first move, and a 3 s memory is full from the start. Once full, it is not just the last
moves: the world always keeps the very beginning and the latest moments in view, plus a sample of everything in between, sparser for
older parts, and it stores at least twice as much as it shows, so older views can come back into the sample. With more memory, places, objects and characters you walked
away from are more likely to look the same when you come back; with less, the world can slowly drift. The memory is kept in computer
memory (RAM) and sent to the graphics card piece by piece while each move is computed, so it takes little graphics card memory (VRAM);
the open menu shows how much of each every choice needs for this world, for the memory profile used for videos. More memory makes each
move slower, as there is more to send: with memory profile 4 on a 24 GB graphics card at Fast and 832x480, about 3 seconds per move with
7 moves (6 s). With a profile that keeps part of the model in VRAM (profiles 1 and 3, or a VRAM Preload setting), the part of the memory
that fits beside the whole model stays in VRAM too, which saves the sending (the menu then shows the VRAM range). If moves become very
slow or fail, choose less memory.

**Events**: type something that happens, then Enter. An event makes a longer move (at least 2.5 s, all of it remembered) so that it
has time to happen. Name
the world's subject to make it act ("the fox sits down in the snow"); describe what appears or changes otherwise ("a bear walks out
of the trees on the left"). Changes of weather or light did not happen in our tests.

**While exploring**: **H** hides or shows the controls drawn over the video, **G** shows statistics: graphics card and computer
memory in use, how much you have explored (the world's video plus your moves), how much of it the memory holds against its size, in
seconds, video frames and moves, and the last move's time. The buttons at the top right save the current view as an image or add
it to the Image gallery, without the controls; below them, **Record** saves the world's own video followed by all your moves as a new
video of the Video gallery (it takes a few seconds per second of video), **Replay** plays it all here (Space pauses, Esc stops
the replay), and **Save World** saves the world as it is now, with your moves, as a new world of the Worlds gallery (shown there when
you leave): explore it later to resume from where you are, or continue it with a generation. The event box appears once the world is ready, and the viewer keeps your Move, Quality and Memory choices for the next
exploration.

**Continue World** (in the media information) instead makes the next generation extend the world with the prompt's segments or
exploration instructions (a prompt of instructions alone is enough) and saves the result as a new world.

### Camera control
A segment may set a `camera`: the camera then follows that motion instead of a camera description in the text. Use a named
trajectory (`"forward"`, `"pan_left"`, `"tilt_up"`, `"arc"`...) or held keys (`{"keys": "w+left"}`): W / S walk forward / back, A / D
step sideways, the arrows turn. Segments without a camera are steered by their text alone.

### Tips and limits
- Give each segment enough time for its changes: 3 to 10 seconds; a big change of scene in 2 seconds may not happen.
- Each generated segment sees the first frame and up to 18 latent frames (about 3 seconds) chosen across the history, plus the text
  of the segments they come from: very old details can fade on long worlds.
- Very fast turns (a named turn squeezed into a very short segment) are refused.
- The PixelUMM decoder weights are licensed for non-commercial research and evaluation only (NVIDIA One-Way Noncommercial License).
"""

HY_NOMAS_PROMPT_INFOS = """### World description
Write one JSON object (timestamps in MM:SS.mmm, segments back to back from 00:00.000; `<|Subject1|>` is the one main subject):
```
{"perspective_view": "third-person follow", "Style": "real-world footage",
 "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "red fox",
                   "subject_attribute": "A slender red fox with a bushy white-tipped tail, seen from behind a few meters ahead."}],
 "segments": [
  {"start_time": "00:00.000", "end_time": "00:04.000",
   "visual_description_long": "A snowy pine forest at dawn, a narrow trail winding between tall pines in soft pink light.",
   "contained_actions": [{"start_time": "00:00.000", "end_time": "00:04.000",
                          "action_description": "<Action1><|Subject1|> trots along the trail away from the camera</Action1>."}]},
  {"start_time": "00:04.000", "end_time": "00:08.000",
   "description_change": "Compared to the previous segment, the trail opens onto a frozen lake under a brightening sky.",
   "contained_actions": [{"start_time": "00:04.000", "end_time": "00:08.000",
                          "action_description": "<Action1><|Subject1|> stops at the shore and sniffs the air</Action1>."}]}]}
```
- The scene texts never mention the subject: the first segment describes the whole place in `visual_description_long` (80 to 200
  words); later segments describe only what changes, starting with "Compared to the previous segment,".
- Everything the subject does goes in `contained_actions`, wrapped in `<ActionN>...</ActionN>`, numbered from `<Action1>` in each
  segment, at least 2 seconds each, without overlaps. Its looks go in `subject_attribute`.
- Optional camera moves in the scene texts: `<Camera1>The camera slowly pans to the right</Camera1>.`
- Perspectives: first-person, second-person, third-person follow, third-person observer. Styles: real-world footage, CG, video game,
  animation/anime.

### Exploration instructions
After the description, each line in brackets is one move through the world, made as the explorer makes it, one after the other:
```
A narrow cobblestone street in an old town on a sunny afternoon, a fountain in a small square ahead.
[Forward]
[Forward 2s]
[Forward+Look_Left]
[Look_Up]
[Wait]
[Event] a cat jumps onto the fountain
[Forward] [Event] the street lamps light up
```
- `[Forward]`, `[Backward]`, `[Left]` and `[Right]` walk and step sideways, relative to where the camera looks; `[Look_Left]`,
  `[Look_Right]`, `[Look_Up]` and `[Look_Down]` turn the camera; `[Wait]` keeps it still. Join commands with `+` or write them on the
  same line to do them together.
- A move lasts about 1 second; give another length in seconds: `[Forward 2s]`, `[Look_Left 1.5]`.
- `[Event]` makes the rest of its line happen during the move: naming the world's subject makes it act ("the fox sits down"), anything
  else changes the scene. An event move lasts at least 2.5 seconds so that it has time to happen. Put `[Event]` last on its line.
- The description sets the opening video (its timestamps, or the World History Duration); each instruction adds its move after it. When you
  continue a world, the prompt can be instructions alone, and the video shows only the new part (the world keeps all of it: Record in
  the explorer makes a video of the whole world).

### Camera
Add `"camera"` to a segment to drive its camera; any `<Camera1>...</Camera1>` description in that segment is then ignored.
- Named trajectories, spread over the segment: `forward`, `dolly`, `pan_left`, `pan_right`, `tilt_up`, `tilt_down`, `truck_left`,
  `truck_right`, `crane_up`, `crane_down`, `arc`, `circular`, `spiral`, `zigzag`, `random_walk`.
- Held keys, at walking speed and 25 degrees per second: `{"keys": "w"}` for the whole segment, or a sequence such as
  `[{"keys": "w", "duration": 1.0}, {"keys": "w+right", "duration": 1.5}]` (seconds; an entry without a duration lasts until the end of
  the segment, the camera stops after the last one). Keys: `w`, `s`, `a`, `d`, `up`, `down`, `left`, `right`; optional `mouse_dx` /
  `mouse_dy` turn a little more per frame.
"""

HY_NOMAS_DEEPY_INFOS = """Hy-Nomas world model (text-to-video, 24 fps, up to 60 s): segments generated in order, each attending to the first frame and up to 18
latent frames of history plus their segments' text (spatial and identity memory). Distilled: num_inference_steps 8, guidance_scale 1, no
negative prompt. resolution: multiples of 32 (832x480 default, 1280x704 native high). The prompt is a JSON world description whose segment
timestamps set video_length; a plain text prompt is one segment of video_length frames (prompt_enhancer "T" writes a world from it). No
image/video inputs. Camera control per segment through the JSON "camera" field (named trajectory or held keys); other segments follow their
text. Segments of 3-10 s. Exploration instruction lines after the description ([Forward], [Look_Left], [Event] ...) add one move each.
world_output: "world" (default; a .wgpworld world in the Worlds gallery, no video decoded), "world_video" (the world and its video),
"video" (a video only, not explorable). custom_guide = a world (Worlds gallery media id or .wgpworld path) continues it with the prompt's
segments or instructions (instructions alone suffice; its video holds only the new part, from the continued world's last frame, and its
world file the whole world); interactive keyboard exploration happens in the Worlds gallery viewer, which agents
cannot drive. Instructions: Look_* turn 25 degrees per second (90 degrees = [Look_Left 3.6s]). Decoder weights non-commercial."""

HY_NOMAS_DEEPY_PROMPT_INFOS = """Prompt = one JSON object: perspective_view (first-person | second-person | third-person follow | third-person observer), Style
(real-world footage | CG | video game | animation/anime), subject_list [{subject_id "<|Subject1|>", display_name, subject_attribute (looks,
position w.r.t. the camera)}], segments [{start_time, end_time (MM:SS.mmm, back to back from 00:00.000), visual_description_long (first
segment only, 80-200 words, scene only), description_change (later segments, scene changes only, starts "Compared to the previous segment,"),
contained_actions [{start_time, end_time inside the segment, >= 2 s, no overlap, action_description "<ActionN><|Subject1|> ...</ActionN>."}]}].
Scene texts never mention the subject; subject behaviour only in actions; optional camera sentences "<CameraN>...</CameraN>." in scene texts.
Video length = last end_time. Optional per segment "camera": a trajectory name (forward, dolly, pan_left, pan_right, tilt_up, tilt_down,
truck_left, truck_right, crane_up, crane_down, arc, circular, spiral, zigzag, random_walk) spread over the segment, or held keys
{"keys": "w+left"} / [{"keys": "w", "duration": 1.0}, {"keys": "right"}] (keys w s a d up down left right, durations in seconds, optional
mouse_dx / mouse_dy); <CameraN> text blocks of such a segment are dropped.
Exploration instructions: after the description (or alone when custom_guide continues a world), one line per move, in order:
[Forward] [Backward] [Left] [Right] (walk / step, relative to the view), [Look_Left] [Look_Right] [Look_Up] [Look_Down] (turn), [Wait]
(still); combine with + or on one line ([Forward+Look_Left]); length in seconds [Forward 2s] (default ~1 s); [Event] text (last on its
line, >= 2.5 s): the world's subject acts when named, otherwise the scene changes. Each instruction is one segment after the description's
video; the description's timestamps / video_length do not include them."""

# condensed from the release's online prompt rewriter (apps/prompts/online_user_video_rewriter_system_prompt.md, Apache License 2.0)
_EXAMPLE = {
    "perspective_view": "third-person follow",
    "Style": "real-world footage",
    "subject_list": [
        {
            "subject_id": "<|Subject1|>",
            "display_name": "red fox",
            "subject_attribute": "A slender red fox with a thick russet coat, black legs and a bushy white-tipped tail, seen from behind a few meters ahead of the camera."
        }
    ],
    "segments": [
        {
            "start_time": "00:00.000",
            "end_time": "00:04.000",
            "visual_description_long": "A snowy pine forest at dawn. A narrow trail of trampled snow winds between tall dark pines whose branches bend under thick snow. Soft pink light filters through the trunks and long blue shadows stretch across the untouched snow on both sides of the trail. Small clumps of snow drop from the branches now and then, and a thin mist hangs low between the trees in the distance. <Camera1>The camera tracks forward along the trail at a low height</Camera1>.",
            "description_change": None,
            "contained_actions": [
                {
                    "start_time": "00:00.000",
                    "end_time": "00:04.000",
                    "action_description": "<Action1><|Subject1|> trots along the trail away from the camera, its tail swaying and its breath steaming in the cold air</Action1>."
                }
            ]
        },
        {
            "start_time": "00:04.000",
            "end_time": "00:10.000",
            "visual_description_long": None,
            "description_change": "Compared to the previous segment, the pines thin out and the trail opens onto the white expanse of a frozen lake, and the first rays of sun rise above the far shore and turn the ice golden.",
            "contained_actions": [
                {
                    "start_time": "00:04.000",
                    "end_time": "00:07.000",
                    "action_description": "<Action1><|Subject1|> slows down and stops at the edge of the frozen lake</Action1>."
                },
                {
                    "start_time": "00:07.000",
                    "end_time": "00:10.000",
                    "action_description": "<Action2><|Subject1|> lifts its head to sniff the air, then turns toward the camera</Action2>."
                }
            ]
        }
    ]
}

HY_NOMAS_ENHANCER_INSTRUCTIONS = """You write world descriptions for Hy-Nomas, a world model that generates a video segment after segment, each segment continuing and
remembering the previous ones. Turn the user's request into ONE JSON object describing a world that lasts {{duration_seconds}} seconds.
If the request is already such a JSON object, keep its structure, subject, timing and camera fields and only improve its wording.
Write in English whatever the language of the request. Stay faithful to the request and add the concrete visual details it lacks.

Schema (every key required, in this order):
{"perspective_view": "first-person | second-person | third-person follow | third-person observer",
 "Style": "real-world footage | animation/anime | CG | video game",
 "subject_list": [{"subject_id": "<|Subject1|>", "display_name": "short name of the subject",
                   "subject_attribute": "its looks, clothing or materials, and its position and orientation relative to the camera"}],
 "segments": [{"start_time": "MM:SS.mmm", "end_time": "MM:SS.mmm", "visual_description_long": "string or null",
               "description_change": "string or null", "contained_actions": [{"start_time": "MM:SS.mmm", "end_time": "MM:SS.mmm",
               "action_description": "<Action1><|Subject1|> ...</Action1>."}]}]}

Rules:
- Exactly one controlled subject, always written <|Subject1|>: a person, an animal, a vehicle or a group. Other people, animals and
  moving things are part of the scene.
- Three kinds of sentences, never mixed. Scene sentences describe the place, light, weather, objects and other beings, and never mention
  the subject (no token, name or pronoun for it). Camera sentences describe a deliberate camera move, wrapped as
  <Camera1>The camera slowly pans to the right</Camera1>. (numbered from 1 in each text, only for real moves, never for a still shot).
  Action sentences describe what the subject does or how it changes, wrapped as <Action1><|Subject1|> ...</Action1>.
- The first segment has "visual_description_long": 80 to 200 words of scene sentences (composition from foreground to background,
  materials, colors, lighting, atmosphere) plus optional camera sentences, and "description_change": null.
- Every later segment has "visual_description_long": null and a "description_change" that starts with "Compared to the previous
  segment, " and states only what changes in the scene (new places reached, objects moving, light or weather changing), with optional
  camera sentences; it never names or describes the subject, whose moves go in "contained_actions". If nothing changes, write: "Compared to the previous segment, the visible scene, background, and other-object
  layout remain unchanged."
- "contained_actions" lists the subject's actions in order: action numbers restart at <Action1> in each segment, each action lasts at
  least 2 seconds, actions never overlap and stay inside their segment. Break complex actions into clear steps. Use [] when the subject
  does nothing visible.
- Timing: the first segment starts at 00:00.000, each segment starts at the previous end_time, the last segment ends at the total
  duration. Use 3 to 10 seconds per segment, one segment per clear stage of the story; give actions and scene changes enough time.
- No on-screen text, logos, subtitles or game interface unless the user asks for them.
- Add a "camera" key to a segment only if the user asks for a camera path in that segment: a name among forward, dolly, pan_left,
  pan_right, tilt_up, tilt_down, truck_left, truck_right, crane_up, crane_down, arc, circular, spiral, zigzag, random_walk, or held keys
  such as {"keys": "w+left"}. Such a segment has no camera sentences.

Example for a 10 second request "a fox in a snowy forest reaches a frozen lake at sunrise":
""" + json.dumps(_EXAMPLE, indent=2, ensure_ascii=False) + """

Before answering, check: every action_description is exactly "<ActionN><|Subject1|> ...</ActionN>." with its closing tag; no scene
text names, describes or refers to the subject (not even "the knight" or "the walker"); action numbers restart at <Action1> in each
segment; the last end_time equals the total duration; a segment where the user asks for a key or a camera path has a "camera" key.
Output only the JSON object, complete and valid, indented with 2 spaces like the example: no Markdown, no comments, no text before or
after it."""
