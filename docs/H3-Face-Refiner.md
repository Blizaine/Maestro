# H3 Face Refiner

Maestro can detect, track and refine up to five faces in a video with H3
Ref2VA. It restores stabilized face crops and blends them into the original
frames. The output keeps the source resolution, frame count and frame rate.
The original soundtrack is retained (copied without re-encoding when its
codec fits MP4). The original video is never overwritten.

## Automatic refinement

In Studio Video, open **Advanced → Finishing** and enable **Refine faces after
generation**. The default **Auto · up to 5** finds relevant identities. Choose
1–5 to request a specific maximum, including smaller or briefly visible faces.
Refinement runs after the complete clip is assembled and before temporal
interpolation, DLSS/FlashVSR finishing, film grain and voice replacement.
A new `_faces_refined.mp4` copy appears alongside the original.

Saved characters in Reference mode are matched by facial similarity, not by
their order in the prompt. Unmatched faces retain their own identity. Under
**Character matching & advanced settings**, optionally choose a different set
of up to five saved characters. Ordinary characters and imported RefMods both
work through their visual reference. Face refinement uses visual identity;
saved character audio does not replace the video's original soundtrack.

## Existing generations and uploaded videos

Open a gallery video's **More clip actions (⋯) → Refine faces**. The same
action is available in Tools, where a video can also be uploaded.

Click **Refine faces** for automatic processing, or first choose **Detect & map
faces**. Each detected identity gets a thumbnail and a selector:

- **Keep original identity** uses a clear source crop as the reference.
- A **saved character** explicitly guides that tracked face's appearance.
- **Skip this face** leaves its track out of the refinement pass.

Changing the source, face count or automatic matching candidates clears the
previous analysis. Maestro also checks the source file before applying a saved
mapping. Face numbers refer to those thumbnails, not prompt speaker order.

## Settings and runtime

- Default strength: **75%**. Lower it for a gentler pass; stronger refinement
  can alter facial details or expression.
- Default: **4 steps**, adjustable through 8.
- Default window: **10.1 seconds**. Use 5.2 seconds for less VRAM or 14.4 seconds
  for longer context. Longer videos use overlapping windows and separate
  presence segments, with resets across discontinuities.
- **Auto model** uses the installed H3 Fused checkpoint when available;
  otherwise it uses H3 Ref2VA Pruned with the LightX2V four-step adapter.
  Either can also be selected explicitly. PDD is not used for this pass.
- NVIDIA CUDA is required for H3. Detection models and any missing H3 assets
  download on first use. `ultralytics==8.3.228` is included in Maestro's
  requirements; existing installs obtain it through the normal dependency update.
- Detection, refinement and generation share Maestro's GPU queue. Jobs can
  be cancelled, and continue if the dialog is closed.
- Source and output frames are backed by temporary files to avoid retaining
  entire long, high-resolution videos in RAM. This uses temporary disk space.

## API

`POST /api/v1/tools/face-refiner` accepts `video_path`, `workspace`, `options`,
and optionally `analyze_only`, `analysis_id`, and `assignments`.

```json
{
  "video_path": "my-video.mp4",
  "workspace": "default",
  "options": {"face_count": 0, "strength": 0.75},
  "analyze_only": true
}
```

Poll the returned `job_id` through `/api/v1/status/{job_id}`. Once detection
completes, read `/api/v1/face-refiner/analyses/{analysis_id}` for face thumbnails
and mappings. Submit again without `analyze_only`, using the returned
`analysis_id` and an `assignments` list:

```json
[{"track_id": 1, "character_id": "saved-character-id", "skip": false},
 {"track_id": 2, "character_id": null, "skip": true}]
```

Automatic generation accepts the same settings as `face_refiner`, with
`enabled: true`, on the ordinary `/api/v1/generate` request.

The tracking/crop/compositing port is pinned to
[Wan2GP 1e1dd275](https://github.com/deepbeepmeep/Wan2GP/tree/1e1dd2757f24923f008593d9d4ec09062234be20/postprocessing/h3_face_refiner),
including its MIT-attributed Carasibana face-tracking work. Maestro adapts the
noise schedule, model lifecycle, queue, character mapping and interface to its
own backend; see `THIRD_PARTY_NOTICES.md`.
