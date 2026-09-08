# Director controls

Director keeps its project setup and planning history in one vertical message
area, with the story composer and Start / Queue actions below it. Character
and speaker fields fit the available panel width. Long descriptions, filenames,
errors and planning logs wrap inside their cards. Progress updates scroll only
the Director message area, leaving the surrounding page in place.

Director already shares Studio's themes, mobile drawer viewport tracking,
keyboard focus handling and gallery scroll lock. H3 references use the same
reference component and saved-character support. Target duration uses the
shared duration planner with model-specific steps. Its Auto option shows the
current recommended duration, including before switching from Time or Window.

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
Real iOS keyboard animation still needs device validation.
