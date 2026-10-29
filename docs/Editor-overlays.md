# Animated title overlays and chroma key

Chroma key removes a chosen solid background from an image or video clip. Put the
keyed clip on a higher video track to reveal another scene underneath. This is a
non-destructive clip setting: the original media file stays unchanged.

## Create an animated title

### From Studio

Choose **Video → Create → Titles**, below **Blend**, to generate a title without
opening an Editor project. Describe the title and animation naturally, optionally
choose a style preset, and set the background, native duration, shape and resolution.
The title has its own draft; unrelated Studio prompts, references and LoRAs are
not reused. **Load Settings** on a title output restores its title controls.

The output is saved in the workspace selected when you submit. Queue progress
and completed media use the normal Studio gallery. Changing workspaces while it
runs does not move the output or modify an open Editor project. Lost receipts
and browser reloads recover the original submission rather than generating again.

**Black background** produces an opaque title card. **Chroma key (green)** saves
a video with a green background for keying in the Editor or another editing app;
the Studio video itself has no alpha channel. Generated audio remains in the
source clip and can be muted or removed during editing.

### From the Editor

Choose **Create animated title** in the Editor media panel. Use the same composer:
describe the exact words, their lettering, order and motion in one request. H3 Fused
4-Step — Frames must already be installed. The description starts blank. The optional
style list offers dust, smoke, neon, typewriter, glitch, liquid metal and the other
animation presets; leaving it unset follows your own description.

### AI planning and review

For example: `Show “CREATE”, then “EXPLORE”, then “INSPIRE”, one at a time. White
lettering on black. Each word assembles from dust and clears before the next appears.`
Put exact on-screen text in quotes. Choose the actual background with its control.

**Enhance** asks the configured prompt-enhancement writer to prepare a plan without
rendering a video. **Generate title** prepares a plan first when the description,
style, duration or background has changed. **Review plan** shows the inferred words,
order, motion and timing. Potential wording changes require reviewing the proposal
and pressing Generate again. The application controls native frame counts and
backgrounds; the LLM supplies the creative interpretation.

Planning makes one bounded writer call. It shares Maestro's existing GPU lease, so
it cannot compete with an active generation. Writer drafts and reasons are saved
locally and exposed under **AI draft and diagnostics**. A malformed reply preserves
the description and available draft; it does not submit a video or silently replace
the text. There is no automatic retry loop.

**Advanced** retains exact text, lettering, animation and phrase controls. Use it
to adjust the AI plan or prepare a title without an LLM, then choose **Use manual
title plan**. Existing saved title settings and pending render receipts remain usable.

**Chroma key (green)** is selected by default. Maestro prepares matching, flat green
start/end images at the requested dimensions and automatically keys the generated
green background, so the title shows over lower footage. Choose **Black background**
for an opaque black field instead. That clip stays opaque and covers footage below
across its full frame. Maestro does not reuse Studio reference images, LoRAs,
enhancement drafts or other creative settings. Global runtime/performance preferences
still apply as they do to other generation jobs.

The result is inserted at the captured playhead position on a new **Title FX** video
track above existing visual tracks. Green keying and starting edge/spill settings are
enabled automatically for the chroma option; the black option remains opaque without
automatic keying. Generated clip audio is muted unless **Keep generated clip audio**
was selected. Review the exact wording, timing and edge quality: generated results
can vary, and changing baked-in lettering requires another render.

### Phrase sequences

The composer infers a single composition or a sequence from the description.
Under **Advanced**, choose **Single title** for one composition, or **Phrase sequence**
to adjust different wording in order. Each sequence card specifies the exact text, its time
slot and its animation. A slot includes the entrance, a readable hold and the exit.
Move cards up or down, add or remove cards, or paste one phrase per line. Describe
custom motion directly, or use an optional preset as an editable starting point.

For example, create cards for `CREATE`, `EXPLORE` and `INSPIRE`. Give each its own
motion, such as scaling in, rotating in and fading in. Every phrase is instructed
to clear before the next appears. The final phrase clears to the selected background;
the complete list is never requested as a final composition. Appearance belongs in
**Lettering and visual style**. Shared sequence motion is optional and separate from
the single-title animation, so a previous full-title preset does not carry over.

A sequence contains 2–6 phrases, up to 80 characters per phrase and 160 overall.
Choose **Sequence duration** from H3's native clip lengths: 124–345 frames in
17-frame steps at 24 fps (5.167–14.375 seconds). Changing the clip length fits the
phrase slots proportionally into that exact frame count. Editing one phrase time
snaps it to a video frame and redistributes the other slots to preserve the chosen
total. Adding, removing or pasting phrases also keeps the slots aligned. Each slot
allows 1–10 seconds; six phrases therefore require at least the 158-frame option.

H3's native-duration rule applies to the whole generated clip. Individual phrases
can be shorter. The timing preview uses contiguous, whole-frame slots whose sum
matches the chosen clip length. This creates one video with one generation job.
H3 still interprets the requested timing and typography: review the actual result
for spelling, word isolation and transitions. For frame-accurate cuts, generate
separate single-title clips and place them on the Editor timeline.

### Writing a readable kinetic title

Use the exact wording you want, including punctuation, and describe the lettering,
materials, colors and visual style in your own terms and language. Give the animation
a clear sequence with concrete motion verbs—for example, reveal, trace, assemble,
sweep or dissolve. Say when the complete title should hold still long enough to read,
and how it should leave the frame. A simple reveal → readable hold → exit gives the
model a clear timing arc while leaving room for your chosen style and effects.

The [WeShop guide to AI text animation with MiniMax H3](https://www.weshop.ai/solutions/models/ai-text-animation-with-mini-max-h3-create-a-black-and-white-kinetic-typography-video)
is one source of inspiration for typography-focused motion, exact copy, specific
movement and readable holds. It is not a guarantee of the same results in Maestro;
generations can vary.

You can keep editing while generation runs. The title is tracked by its exact
submission/job identity. A lost receipt is checked read-only, never automatically
resubmitted. If you switch projects, the result waits for the original project;
it is not inserted into the new one. **Stop tracking** does not cancel the GPU job.
The original generation remains in its workspace if you dismiss tracking.

Frame counts are snapped to H3's native 24 fps lattice, so the displayed effective
duration can differ slightly from the requested seconds. The wizard creates both
boundary images with the selected shape. In ordinary Studio Frames workflows,
an explicit aspect ratio uses a centered cover crop when a start/end image has a
different shape; proportions are preserved and outer edges can be trimmed.
**Auto** follows the source image's aspect instead.

## Key an existing clip

1. Add the main scene to the timeline.
2. Add the title/effect clip to a higher video track and select it.
3. Open **Basic → Chroma key** in the Inspector and enable it.
4. Choose **Green**, **Blue**, or a custom **Key color**.
5. Increase **Color tolerance** until the background disappears. Keep it as low
   as practical to avoid removing similarly colored foreground details.
6. Adjust **Edge softness** to blend dust, smoke and fine edges. **Spill reduction**
   reduces leftover green or blue tint; it is inactive for other key colors.
7. Use the clip's **Audio** controls to mute unwanted generated sound. Muting the
   entire video track also hides its visual content.

Keying happens before clip color adjustments, transforms, fades and opacity.
Existing source alpha is multiplied by the new matte, so already transparent
pixels remain transparent. Settings are saved with the clip, included in portable
projects, and can be undone or disabled. Locked tracks cannot be changed.

## Generate footage that keys cleanly

Ask for a perfectly flat, evenly lit green or blue background with no texture,
shadows, lighting changes or matching-color reflections on the subject. Choose a
background color absent from the lettering and particles. A fixed camera is
helpful for stationary titles. Start/end references can keep the background
consistent, but generated footage can still introduce haze or color changes.

For example: “Animated title lettering turns to dust over a perfectly uniform
bright green background. The background remains flat and unchanged. No green
reflections or green particles. Keep the lettering fully inside the frame.”

Keying cannot distinguish a green particle from an identical green background.
Green-tinted generated smoke can leave fringes or lose detail as tolerance is
increased. Softer edges and spill reduction help; cleaner source footage is the
best improvement when colors are inseparable.

## Preview and export

Keyed preview reads the original media rather than a compressed preview proxy,
which could alter the background colors. WebGL renders the matte in the browser.
A reduced-resolution CPU preview is available when WebGL fails; it is labeled
and runs at a reduced frame rate. Export always keys the original source at its
native resolution before scaling. Browser decoding, preview scaling and encoded
output can differ slightly at edges; inspect a short export for critical work.

The normal MP4 export is a finished composition over the lower tracks/background.

## Export a transparent MOV

Select a visual clip or text title and choose **Export transparent MOV** in its
Basic Inspector. This exports only that overlay, moved to time zero, at the chosen
delivery resolution. Background footage and unrelated audio tracks are excluded.
Selected keying, framing, color, opacity, fades and automation are retained. The
saved project and original media are not removed or replaced. Its source clip's
mute setting and the export's audio toggle determine whether embedded audio is
included.

For a whole-project transparent render, open **Export** and choose **MOV · ProRes
4444 with alpha**. All visible tracks are composited over transparency; an opaque
video still covers the background under it. Use the clip-specific action when
you want to isolate a title from footage underneath.

ProRes 4444 MOV preserves partial transparency for dust and soft edges. It is a
large, software-encoded editing master, not a small web-delivery file. Optional
audio is PCM. The export requires a local FFmpeg build with `prores_ks`; its
availability is shown in the export dialog. AI upscaling/finishing passes are
unavailable because they do not preserve alpha. An alpha-capable editing program
can composite the MOV over its own footage. Browser/phone players may be unable
to play the MOV or show black where alpha is transparent; that alone does not
mean transparency was lost.

H3 itself produces RGB video. This workflow calculates an alpha matte after
generation and saves it; changing the file format cannot repair contamination
or missing particles in that matte. Genuine source alpha is preserved when
supported by the source codec and decoder.

The CPU export uses FFmpeg's [RGB color key](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_colorkey.c)
and [despill](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_despill.c)
filters, with an additional source-alpha multiplication step. No AI model or new
weights are required for keying.
