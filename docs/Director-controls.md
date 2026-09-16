# Director controls

Director keeps its project setup and planning history in one vertical message
area, with the story composer and Start / Queue actions below it. Character
and speaker fields fit the available panel width. Long descriptions, filenames,
errors and planning logs wrap inside their cards. Progress updates scroll only
the Director message area, leaving the surrounding page in place.

Director already shares Studio's themes, mobile drawer viewport tracking,
keyboard focus handling and gallery scroll lock. H3 references use the same
reference component and saved-character support.

LoRAs and Advanced appear directly below the model selectors, before the music
and image inputs, so generation settings can be chosen before audio analysis.

## Target duration

Target duration uses Studio's compact duration control. Auto and the current
duration stay above the Time / Window tabs. Auto dims the Time slider and
presets, but grabbing the slider, using its arrow keys, selecting a preset or
editing the timecode immediately switches to manual timing. Auto is available
from either tab.

Time follows model-specific duration steps through five minutes, with 10m,
15m, 30m, 60m and Custom choices for longer projects. Window keeps its exact
window-count controls. Director retains its ten-second minimum and chooses
shot capacity from its selected video model, resolution and GPU recommendation.

For H3 First / Last models that support **Seamless**, Director renders the
full timeline through native continuation windows. The saved maximum shot
length limits each window, while the full movie keeps its planned duration,
including the final trimmed window. The total movie length does not have to
match the frame increments of a single H3 shot. Independent shots still need
to fit the model's frame increments and the saved maximum.

## Music-video clip length

**Advanced → GPU clip limit**, beside inference steps, sets the maximum
generated shot length for supported bounded models. Auto uses Maestro's GPU
recommendation. A manual choice is remembered per model across refreshes and
restarts. For example, H3 fused Ref2VA can be set to **14.4s** (345 frames at
24fps) on a setup where the automatic recommendation is 10.1s. Choosing a
longer limit uses more GPU memory; it does not force every shot to that length.

Music-video setup includes **Clip length**, with Auto and a maximum-length slider.
Auto follows the chosen video model, resolution and memory recommendation. Moving
the dimmed slider selects a custom maximum immediately. The displayed time follows
the model's frame increments, and the minimum clip count updates with it.

Cuts follow section changes, lyric phrases, performer changes, and strong beats.
**Cut Speed** controls how tightly these cues divide the song: **0** keeps the
normal musical layout; **+1 / +2** favor shorter clips, including within sections
that already fit the maximum. **−1** favors longer clips. **−2** uses the fewest
clips that fit the selected maximum, balancing their lengths and placing cuts
near musical cues. For a two-minute song with H3's 14.4s limit, −2 produces nine
clips averaging about 13.3s. Section and percussion changes inside a longer clip
still guide its performance and camera plan without requiring another clip.

New music plans keep cut positions on the output frame
grid; Maestro generates a valid model-native duration and trims the extra tail.
That preserves the soundtrack timing through generation, reruns, and rejoining.
Shorter clips can lower peak memory, though
they add preparation and transitions. After planning, choose **Update clip layout**
to apply a changed maximum. The setting is saved with the project. Separate-clip
length is disabled for Seamless, which uses its native continuation windows.

New audio analyses also estimate sustained percussion passages and entrances
using transient energy on the CPU. Director receives the entrance's song time
and time within the shot, favoring the referenced drummer at those moments
instead of inventing a new drum entrance on an arbitrary beat. These are
percussion estimates, not separated drum stems or exact instrument recognition;
other transient sounds can resemble drums. Unknown timing stays unknown.
Re-analyze the soundtrack and create a new plan to add these cues to an older project.

## Shared prompt writing

Director uses the adaptive writing rules from Studio inside its existing planning
stages. Short concepts receive development; detailed scripts retain their events,
exact lines and constraints. Shot planning includes character/prop ownership,
physical transitions, first-frame authority and the final state needed for continuity.
Music videos retain the supplied song's lyrics, voices and vocal timing.

Music planning assigns visible people their vocal or instrumental roles. During
guitar and drum cutaways, prompts keep the lead singer's voice off screen and
the instrumentalist's lips closed while their playing stays active. Wide band
shots reserve lip-sync for the assigned vocalist. Explicit backing singers,
singing guitarists, wind instruments and crowd cheers retain their intended
performance. These directions apply to newly planned shots; create a new plan
to update older prompts.

H3 compilation keeps complete actions, camera destinations, dialogue delivery and
ending states. It may remove repeated boilerplate, but a cosmetic prompt-length
target cannot cut off an action or discard the end of a shot. Planning caches include
the writing contract and optional-content setting. Reviewed per-shot edits remain
authoritative. Prompt review is still useful: successful preparation does not
guarantee perfect movement, lip synchronization or visual continuity.

## LoRA strengths

Activated Video and Image LoRAs each show a strength slider and numeric input.
H3 has one editable strength even though it does not use a guidance-phase
schedule. Models with multiple phases retain independent controls for each
phase. Values, including zero, are saved and carried into Director projects.
Older empty weight arrays recover serialized strengths where available, or
use 1.0 when no strength was saved.

## Recommended next UI pass

These are proposed changes, not yet implemented:

- Replace the permanently expanded resolution, aspect and target-duration
  controls with compact indicators that show their current values and open
  menus sized to the Director panel. Preserve Director's own model capabilities
  and settings rather than changing the active Studio generation settings.
- Group H3 optimizations, LoRAs and finishing under one Advanced area with
  collapsible sections. Show only the sections applicable to the chosen models.
- Use a compact Characters entry and consistent reference tiles across H3 and
  other Director models, retaining character, location, voice and soundtrack
  roles. Keep both video and optional image model choices available.

Retain Director's skill selection, staged planning and prompt review, plus its
existing Start / Queue actions. Studio's explicit Enhance action should not
replace the Director planning workflow.

## Local validation

After building `ui`, set `MAESTRO_UI_DIRECTOR_ONLY=1` and run
`node tests/ui/sidebar_redesign.cjs http://127.0.0.1:<Maestro port>`.
The existing isolated browser harness uses test projects and mocked writes;
it does not submit a real plan or generation. Checks cover horizontal bounds,
vertical scrolling, long descriptions and logs, character and speaker inputs,
and simulated mobile keyboard geometry at desktop, 390px and 320px widths.
The same command checks duration takeover from Auto, native steps, long
presets, H3 and multi-phase LoRA weights, saved strength recovery, and the
duration/weights in an intercepted queue request.
Real iOS keyboard animation still needs device validation.
