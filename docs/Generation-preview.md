# Generation previews

In **Settings → Performance → Generation Preview**, choose:

| Option | What appears in Studio |
| --- | --- |
| Off (default) | The usual generating card; no preview decoder or capture work. |
| Fast Frames | A small set of approximate RGB frames sampled during denoising. |
| Clearer Frames (Tiny VAE) | A small set of decoded frames from a lightweight preview VAE. |
| Live Video (Tiny VAE) | A muted, looping preview of the current window. Images use clearer frames. |

The setting takes effect when the next generation begins, including queued jobs. It remains available with Performance Auto enabled. The preview stays inside the Studio generating card, alongside progress and ETA, and labels its clip and window. Click the video or its pause/play button to control playback without pausing generation.

On phones, live previews play inline. If the browser blocks automatic playback, the preview remains available with a **Play live preview** button. Tap that button or the video to start playback; generation continues while the preview is paused.

Below the setting, Maestro lists native preview support for the selected Studio model. The labels follow the exact model variant and update when the model changes. The preference remains selectable across models; unsupported Tiny VAE modes use Fast Frames where available. Image generations show still previews, including when Live Video is selected. Audio generations have no visual previews.

Intermediate output is approximate and can start as noise. Video previews have reduced resolution and frame rate and show only the current window. They do not establish final detail, sound quality, identity retention or lip sync. Previews add decoding work and can increase GPU memory use and generation time. Leave them Off for the lowest overhead.

H3 previews show its predicted clean result during denoising, so the scene can become visible before the sampling schedule removes all remaining noise. Early composition and motion can still change. The final preview uses the fully denoised window. This changes only preview presentation; the generation's seed, sampling steps and final video/audio stay the same.

H3, supported Wan variants and LTX-2/2.5 use registered Tiny VAE decoders. H3 requires Clearer Frames or Live Video; it has no legacy RGB preview. Models with other latent contracts use Fast Frames where supported; LongCat currently uses that fallback. Audio-only jobs do not produce video previews. The first Tiny VAE use downloads a small decoder checkpoint into Maestro's model storage and verifies its checksum. Decoder setup, capture or encoding errors fall back to Fast Frames or display a preview notice while the render continues. A cancelled download still respects cancellation of the job.

Preview media is temporary, isolated by job and window, and bounded in memory. It is cleared on completion, failure, cancellation and restart. It never becomes a gallery asset or an archived job payload. Refreshing the page while a job runs reconnects to its latest preview.

## API

Read or update the option through `GET` / `PUT /api/v1/system-config`. Valid values are `off`, `rgb`, `tiny_vae_frames` and `tiny_vae_video`.

`GET /api/v1/models` includes `preview_support` on each model entry, with boolean `rgb`, `tiny_vae_frames` and `tiny_vae_video` fields. These describe registered native support without downloading or loading a preview decoder. For image jobs, `tiny_vae_video` means the setting produces still previews. Older backends omit the metadata, which the UI shows as unknown support.

```javascript
await fetch(`${baseUrl}/api/v1/system-config`, {
  method: 'PUT',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({generation_preview: 'tiny_vae_video'}),
});
```

```python
import requests
requests.put(f"{base_url}/api/v1/system-config",
             json={"generation_preview": "tiny_vae_video"}).raise_for_status()
```

```sh
curl -X PUT "$MAESTRO_URL/api/v1/system-config" \
  -H 'Content-Type: application/json' \
  -d '{"generation_preview":"tiny_vae_video"}'
```

`GET /api/v1/status/{job_id}` and `GET /api/v1/jobs` include `preview` and `preview_notice`. `preview` is null until media is ready and after a job ends. When present, it contains `url`, `kind` (`image` or `video`), `revision`, `mode`, `window`, `total_windows`, `clip` and `total_clips`. Fetch its job-specific `url` for JPEG or MP4 bytes; video range requests are supported. Expired or completed-job URLs return 404. Clients should use the returned URL and treat media errors as presentation failures.

The decoder implementation is adapted from [Wan2GP's live previews](https://github.com/deepbeepmeep/Wan2GP/blob/b8b18f8114e432eea8f3d7e853a51dd91fa99571/docs/PREVIEWS.md), contributed by GOvEy1nw. Decoder sources, weight hashes and accompanying licenses are included under `app/shared/tinyvae`.
