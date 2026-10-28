# Animated title overlays and chroma key

Chroma key removes a chosen solid background from an image or video clip. Put the
keyed clip on a higher video track to reveal another scene underneath. This is a
non-destructive clip setting: the original media file stays unchanged.

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
This feature does **not** add an alpha-channel video export preset. Real alpha
assets avoid color separation when supplied, but video alpha also depends on the
codec and browser decoder. Portable transparent overlay exports are separate work.

The CPU export uses FFmpeg's [RGB color key](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_colorkey.c)
and [despill](https://github.com/FFmpeg/FFmpeg/blob/master/libavfilter/vf_despill.c)
filters, with an additional source-alpha multiplication step. No AI model or new
weights are required for keying.
