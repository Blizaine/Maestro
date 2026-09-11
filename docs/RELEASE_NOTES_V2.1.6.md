# Maestro v2.1.6

Released 11 September 2026.

This patch improves H3/Viggle transformer residency, fixes more false dialogue
errors in detailed prompts, simplifies reference editing, and adds remembered
step counts with a higher limit for H3 Fused.

## Better H3 and Viggle memory use

Maestro already calculated a per-job transformer allowance after reserving
workspace for H3's packed video/audio sequence. Streaming profiles now receive
that allowance as an actual MMGP transformer budget. Previously, the safety
coefficient only set a ceiling, leaving a small default residency budget and
unnecessary repeated weight transfers on cards with available VRAM.

- Applies to eligible H3/Viggle jobs using profiles **2, 4, 4.5 and 5**.
- Retains activation headroom and the existing VAE, encoder and catch-all budgets.
- Respects explicit preload settings and tighter safety coefficients.
- Reloads a cached H3 model when its requested residency budget changes, and
  restores the base budget after completion, failure or cancellation.

In a contributed RTX A4500 / 28 GB RAM test, a 480p, three-step Viggle workload
improved from approximately **75 to 60 seconds per denoising step** relative to
v2.1.5. The earlier INT8 kernel improvement shipped in v2.1.5; it is separate
from this release's residency change. These are reported measurements for that
workload, not a universal speedup or a guarantee against out-of-memory errors.
Thanks to Cocktail Peanut for the investigation and hardware testing.

See [Performance Auto-Tune](Performance-auto-tune.md) and the
[validation record](VALIDATION_V2.1.6.md).

## Detailed prompts without false dialogue errors

H3 enhancement now recognizes compound production headings such as **Scene
description**, **Visual requirements**, and **Final state**, including quoted
visual descriptions. Production sections and their subheadings no longer turn
camera, effects or action directions into speaking characters.

Imported briefs can also use **Role A/B** character descriptions and titled
time ranges such as `【0.00—4.00｜Opening exchange】`. Their complete action phases,
cast identities and explicit prohibitions such as no slow motion remain part
of the adaptation. This includes pasted text whose line breaks were lost.

Actual screenplay dialogue and explicitly attributed speech still receive
exact-line and duration checks. The selected Maestro duration remains
authoritative. Run **Enhance** again from the original prompt to replace an
older draft, then review **Exact H3 prompts** before generating.

## Compact reference editing

Click a reference thumbnail to expand its name and type fields directly below
that row. The editor flows with the sidebar instead of covering the prompt.

- Drag the top-level thumbnails to reorder references in either direction.
- Saved character appearance and voice move together.
- Preview remains available through the thumbnail's eye button.
- Replacement, image/audio role, background isolation and video soundtrack
  controls remain available where supported.
- Escape closes the inline editor and returns focus to its thumbnail;
  Alt+Left/Right provides keyboard reordering.

The shared reference editor is used in Studio and Director. See
[Studio controls](Studio-controls.md).

## More H3 Fused steps, remembered per model

**H3 Fused 4-Step — Frames** and **H3 Fused 4-Step — References** now allow
**4–12 total steps**. Four remains the default speed preset; additional steps
take longer and are an optional refinement setting.

Studio remembers each model's last selected or used step count across browser
refreshes and application restarts, including a changed Pinokio port. Restored
values respect the selected model's limits and fixed-step acceleration modes.
Prompts, seeds and reference uploads are not restored by this preference.

## Updating

Use **Update** in Pinokio, restart Maestro, then refresh the browser. These
changes require no new dependencies or model downloads. Models, characters,
outputs, workspaces and saved projects remain in place.

See the [validation record](VALIDATION_V2.1.6.md) for automated checks,
reproduction evidence and the limits of the performance measurements.
