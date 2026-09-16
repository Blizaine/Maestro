# Viggle Animate

Use **Studio → Video → Animate** to carry an edited person or object through a
control video. Maestro selects the dedicated Viggle model and its required
distillation adapter. It does not use ordinary H3 text prompting or load Qwen.

1. Upload a control video. Drag the preview's start and end handles to choose
   the range to animate. Move the **white playhead** to the frame you want to
   edit; the preview follows it. The **Start**, **End** and **Frame to edit**
   fields also accept precise times in seconds, measured from the original video.
2. Choose **Use a character**, then **Saved character** or **Upload image**.
   Select the clearest character view and optionally describe clothing or other
   changes under **Appearance**. The selected playhead frame is sent to Klein.
   **Preview replacement frame** is optional;
   **Generate** runs preparation and animation together.
   For a manually prepared image, choose **Use an edited frame**. Upload it or
   select **Edit current video frame in Maestro**, generate the edit in Image
   mode, then **Apply & return**.
3. Start with **480p**. The aspect ratio follows the source. Auto duration follows
   the selected range; Time or Window lets you choose a shorter runtime.
   Generation never extends past the trim's end. Viggle still builds its fixed
   windows internally and trims the final output to the selected duration.
4. Choose no input audio, the control video's soundtrack, or custom audio.
   The default fixed prompt requests silence, although the resulting track may
   contain faint sound. Audio conditioning is experimental; use a synchronized
   track when retaining source or custom audio. Custom audio should align with
   the original, untrimmed video: Maestro applies the same start/end trim to it.
5. Generate. Results appear in the normal gallery and support Load Settings,
   the queue, postprocessing, and optional face refinement.

The manual Image-mode handoff starts with this editable prompt. Add the
replacement character as the second image:

> Replace the character from source image with character from the second image. Preserve the exact pose, body orientation, props, background, camera framing, lighting and image dimensions from the source image.

The recipe uses **three Euler evaluations**, video/audio flow shift 3, and
**124-frame windows at 24 fps (5.17 seconds)**. Longer videos continue with
18-frame overlap, carrying the previous generated motion. The final window is
trimmed to the requested timeline. The edited image conditions every window;
it is not forcibly inserted as the opening frame.

## Automatic character preparation

Flux 2 Klein receives the source frame as image 1 and the selected character
image as image 2. **Preparation settings** defaults to Klein 9B with four steps;
4B is also available. Required model assets download on first use. The default
replacement instructions are:

> Replace the main character in source frame with character in second image. Preserve the exact pose, body orientation, hands, props, background, camera framing, lighting and image dimensions.

Maestro appends the optional appearance description to that image prompt.
These instructions can be edited or reset. They affect Klein's replacement
image; Viggle itself still uses its required fixed prompt. Saved voices are
not used for image preparation. Viggle's separate Audio control determines
soundtrack guidance.

RefMods supply recovered PNG views through the character picker. Their tensors
remain intact. Ordinary characters use original photos or selected video frames.
Choose one clear view for Viggle; Image mode also accepts multiple views within
the selected model's reference limit.

Preparation pads the source to Klein's image grid, then crops the result to the
exact original frame dimensions. It never stretches the pose to fit another
aspect ratio. Output is a lossless PNG. Generative editing can still change
details; preview the result when exact prop, hand or background preservation
matters. Very large source frames need more VRAM during Klein preparation.

Klein and Viggle run serially in the same cancellable job. A preview is reused
only when source, character image, frame time, prompts, model and seed still
match. Changing these inputs prepares a new frame. Prepared images and the
appearance settings are retained in the completed video's Load Settings data.
The original control video and character library assets are preserved.
Trim bounds and the selected frame are also preserved by **Load Settings** and
the Image-mode edit/return workflow. Moving a trim handle keeps the selected
frame unless it falls outside the new range, in which case it moves inside.
Changing the selected frame clears an automatic replacement preview so it can
be prepared again. Uploaded edited images remain available.

The INT8 transformer is approximately 21.1 GB on disk, plus a 1.33 GB adapter,
small fixed prompt tensors, and the shared H3 video/audio VAEs. Assets download
on first use and can reuse Maestro's linked model folders. Ordinary H3 LoRAs,
Turbo/PDD acceleration, Sol and First Block Cache are disabled for this recipe.

Clear source motion and a well-aligned edit matter. Hard cuts, fast occlusion,
interacting people and lip sync can be difficult. Increasing the timeline does
not increase Viggle's fixed generation-window size.

## Studio duration and canvas controls

**Time** now includes a slider through the last model-aligned duration within
five minutes. For ordinary H3, the first steps are 124, 141 and 158 frames:
approximately **5.2s, 5.9s and 6.6s**. Extend measures *new footage*, subtracting
the source-tail context. Presets through **60 minutes** and custom timecodes
remain available.

The automatic window grows with the requested duration up to the model/GPU
recommendation. Open **Duration → Window Length** to set a temporary cap or save
an override for that model and resolution. Longer timelines use multiple windows.
Viggle always uses its fixed 124-frame recipe.

Resolution and aspect are now expandable buttons in the main Studio controls.
Tap a button to see the model's supported choices. Source-dependent transforms
retain the source aspect ratio.

## API

Use absolute paths to media on the Maestro host, or paths returned by uploads.
The model enforces the recipe server-side, including for restored queue jobs.
`video_length` is the total number of output frames at 24 fps: 124 for one full
window, or 230 for two full overlapping windows. Poll `/api/v1/status/{job_id}`.
Replace the port below with the URL shown by Pinokio.

```javascript
const response = await fetch('http://127.0.0.1:42015/api/v1/generate', {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({model_type: 'viggle_animate', generation_mode: 'video',
    video_guide: '/absolute/control.mp4', image_refs: ['/absolute/edited.png'],
    resolution: 'auto_480p', video_length: 230, audio_prompt_type: 'K', seed: 42})
});
console.log(await response.json());
```

```python
import requests
job = requests.post('http://127.0.0.1:42015/api/v1/generate', json={
    'model_type': 'viggle_animate', 'generation_mode': 'video',
    'video_guide': '/absolute/control.mp4', 'image_refs': ['/absolute/edited.png'],
    'resolution': 'auto_480p', 'video_length': 230,
    'audio_prompt_type': 'K', 'seed': 42,
}).json()
print(job)
```

```sh
curl http://127.0.0.1:42015/api/v1/generate -H 'Content-Type: application/json' \
  -d '{"model_type":"viggle_animate","generation_mode":"video","video_guide":"/absolute/control.mp4","image_refs":["/absolute/edited.png"],"resolution":"auto_480p","video_length":230,"audio_prompt_type":"K","seed":42}'
```

For automatic replacement, omit `image_refs` and provide `viggle_character`:

```json
{
  "model_type": "viggle_animate",
  "generation_mode": "video",
  "video_guide": "/absolute/control.mp4",
  "resolution": "auto_480p",
  "video_length": 124,
  "seed": 42,
  "viggle_character": {
    "reference_path": "/absolute/selected-character-view.png",
    "image_model": "flux2_klein_9b",
    "frame_seconds": 0,
    "appearance_prompt": "A dark leather jacket and blue jeans."
  }
}
```

Optional `swap_prompt` replaces the default instructions. The character picker
downloads the chosen PNG from `/api/v1/characters/{id}/images/{view_id}` and
uploads a copy through `/api/v1/upload`, freezing the chosen input for the job.
The same contract works with a normal uploaded character image. Do not pass a
safetensors file as `reference_path`.

Add `_viggle_prepare_only: true` to prepare a preview without animation.
Poll the usual status endpoint; `viggle_preparation` contains `image_path`,
`image_url`, dimensions, prompt and signature. Passing this object back as
`_viggle_prepared` allows verified reuse in the full animation request.
Preview jobs use the ordinary queue and cancellation endpoint but do not add
unfinished video entries to the gallery.

To animate a selection, include `_viggle_trim_start` and `_viggle_trim_end`
(seconds in the original source, with an exclusive end). For example, `2` and
`8` select six seconds. `viggle_character.frame_seconds` stays source-relative:
`5` selects the original video's fifth second, not five seconds after the trim.
The frame must fall within the selected range. `_viggle_frame_seconds` stores
the manual editor's playhead. Output `video_length` is capped to the range at
24 fps. Omitting both bounds preserves the existing full-source API behavior.

Sources: [Viggle model and guidance](https://huggingface.co/Viggle/Viggle-Animate),
[pinned Wan2GP implementation](https://github.com/deepbeepmeep/Wan2GP/tree/057f9ecab9ad57dfbec9768b2daf7a4426ce986c/models/minimax_h3).
See the [implementation and validation record](development/Viggle-and-Studio-Controls.md).
