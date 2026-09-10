# Director controls

Director keeps its project setup and planning history in one vertical message
area, with the story composer and Start / Queue actions below it. Character
and speaker fields fit the available panel width. Long descriptions, filenames,
errors and planning logs wrap inside their cards. Progress updates scroll only
the Director message area, leaving the surrounding page in place.

Director already shares Studio's themes, mobile drawer viewport tracking,
keyboard focus handling and gallery scroll lock. H3 references use the same
reference component and saved-character support.

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
