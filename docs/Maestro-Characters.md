# Portable characters and H3 RefMods

Maestro characters can be shared as a single file named after the character:

```text
blaine.maestro.safetensors
```

The file contains an H3 visual RefMod, the character's saved image or video,
its name and description, and its saved voice recording when present. Recovered
characters can also include selected lossless PNG views and their cover choice. Importing
it recreates a local Character Library entry with working media paths. It does
not depend on the sender's folders or an online service.

## Use the browser

Open **Model Browser → Characters / RefMods**. Search your library or filter by
**With voice**, **Without voice**, or **H3 RefMods**.

- **Import character / RefMod** accepts `.maestro.safetensors` and standard H3
  RefMod `.safetensors` files. For an older RefMod with a separate JSON sidecar,
  select both files together.
- **Import from Hugging Face** accepts a repository or an exact file URL. Select
  **Find files**, search the returned filenames when necessary, and import the
  selected file. A collection's first file is never substituted for an explicit
  file URL. Downloads from other sites can be imported with the file picker.
- **Saved collection → malcolmrey / MiniMax H3** opens the built-in
  [H3 RefMod collection](https://huggingface.co/malcolmrey/minimaxh3/tree/main)
  and lists its available files with one click.
- **Add voice / Replace voice** attaches an uploaded recording. Choose at least
  two seconds of clear speech. Imported visual-only RefMods do not acquire a
  cloned voice automatically.
- **Export** downloads the portable character, including its current saved voice.
  Export and file import are also available in the Reference and Speech character
  libraries. Imports appear in all three places automatically.
- **Recover images** opens a RefMod's image gallery. Ordinary saved characters
  show **Choose images**; recovered characters show their image count.

The browser scrolls on phones and desktops. Video characters have cached still
thumbnails, so their appearance is visible without starting playback. Character
cards and file lists hide the `minimaxh3_` prefix and RefMod / safetensors suffixes
and show filename underscores as spaces. Original files and character bindings
keep their existing names.

The first export of an ordinary saved character needs an installed H3 video VAE
and an NVIDIA GPU. It waits for the shared generation lock, releases resident
generation/prompt models, and encodes the visual reference. Later exports reuse
that visual encode. Image exports retain a full encode, with a maximum 1024-pixel
long edge. Video characters use up to eight independently encoded appearance
views sampled from the first 15 seconds; the original video is also embedded.
These are identity references, not a cache of the video's complete motion.

Complete Maestro character imports with embedded media need no GPU encoding.
A bare third-party RefMod needs the H3 VAE to recover its images and make a
representative image or short gallery preview for visual-language prompting.

## Recover better images

Open **Model Browser → Characters / RefMods → Recover images** on a character.
Recovery waits for the GPU when needed and shows progress. You can leave the
gallery while it runs and return later.

- RefMods decode directly at their stored spatial resolution into separate
  lossless PNGs. Recovery does not resize the latent grid or pass the images
  through an MP4 or JPEG. Up to 128 stored latent views can be recovered.
- Saved original photos are preferred over VAE reconstruction, preserving
  resolution and transparency. Saved original videos provide up to 32 frames
  sampled across their entire duration at their original resolution.
- Tap any view for a larger preview. **Use as cover** sets the library thumbnail;
  checkboxes select images for downloads and portable character exports. At
  least one image must be selected, including the cover.
- **PNG** downloads the current full-resolution view. **Download selected** saves
  the selection and downloads a ZIP of its PNGs. **Save selection** updates the
  character without downloading. The gallery scrolls on mobile and desktop.
- **Recover again** rebuilds the images from their source. A failed attempt
  preserves the previous images and choices. Completed results are cached, so
  opening the gallery again does not reload the VAE.

Recovery preserves the character's original RefMod tensor, saved visual media,
and voice. The selected PNGs do not replace H3's native conditioning. New portable
exports include the chosen PNGs and cover; importing them restores those views
without decoding them again. Earlier Maestro exports and ordinary RefMods remain
supported. Large selections count toward the existing 256 MiB character limit.

This removes avoidable preview compression; it does not invent missing detail.
Pooled or optimized RefMods may already have lost fine features. Independent
image stacks are recovered as separate images. Video or mixed-reference latents
with uncertain frame boundaries are labeled as representative views, not exact
reconstructions of the source video. Each PNG is limited to 32 megapixels.

## Generation

In Reference mode, expand **Characters** to browse portrait cards or search by
name. Tap a card to add its appearance and saved voice to the scene; an accent
border and check mark identify characters already added. Each card's **…** menu
contains export and delete. **Import file** and **New character** are below the
scrollable library.

Add the character from the existing Reference-mode Character Library. Its visual
reference and optional voice remain bound to the same character. Native H3
Ref2VA uses the **original RefMod tensor**, including its stored temporal and
spatial layout, instead of re-encoding the preview. Standard reference noise
augmentation still applies. Background isolation and reference-detail resizing
do not alter an already encoded RefMod.

Multiple RefMods stay separate characters: each retains its own visual tensor
and any saved voice. The current Reference limits are nine image references,
three video references, three audio references, and twelve media references in
total. A video RefMod counts as one video; its saved voice uses an audio slot.
Adding more RefMods also increases conditioning memory and processing time.

Name the speaker beside each line, for example:

```text
Eliza Dushku says, <d>[English] Welcome back.</d>
Sydney Sweeney replies, <d>[English] Good to see you.</d>
```

Manual prompts and AI Enhance share the same speaker matching. Imported names
such as `minimaxh3_elizadushku_v1_refmod` also recognize `Eliza Dushku` and the
displayed `elizadushku v1`. Full filenames remain valid. No stored files or
character IDs are renamed. If two references have the same name, include their
version or use their explicit `<Subject N>` identity. Subject order follows the
character references; `(S1)`, `(S2)`, etc. are assigned by first speaking order,
so the second character can speak first. A bare “she says” with multiple possible
speakers needs a name or Subject tag; Maestro does not assign lines by list order.
Plain RefMods have no saved voice unless audio has been added separately.

The saved voice is also available to Maestro's Speech models that accept voice
references; it is stored as ordinary audio, so it is not tied to the H3 audio VAE.
Compressed MP3/M4A/AAC imports get a local WAV copy for TTS compatibility; the
original recording remains embedded when the character is exported again.
Qwen preset/design variants retain their existing switch to voice-cloning mode.
Visual RefMods require an H3 Reference model, including Maestro's fused Reference
variants. They are not trained LoRAs and do not belong in the LoRA picker.
Recognized RefMods downloaded through the existing model URL importer are routed
to characters automatically.

Character files are limited to 256 MiB and 32,768 visual reference tokens per
file. These are Maestro import limits, not claims about the upstream format.
Compressed or heavily trained third-party RefMods may decode imperfect previews;
their original conditioning tensor is retained. Generation quality depends on
the source RefMod, model and prompt.

## Use characters in images and Viggle

In **Studio → Image**, use **Add character** beside the image references. Choose
a saved character, then one or more views. The saved cover is selected by default.
Missing RefMod views are recovered once and cached. Reopening the picker uses
the existing PNGs. You can import a character directly from this picker.

Chosen images append to the existing references in selection order. For a
character replacement, place the source image first and character image second.
Describe the replacement and appearance changes in the main image prompt. The
picker respects the model's reference limit, including the source image's slot
in reference-capable Inpaint/Outpaint modes. Models without image-reference
support cannot use a character image; Image Generate can switch to a compatible
model when references are attached.

These workflows use ordinary image files, including original photos and
recovered native PNGs. They do not load H3 RefMod tensors into another model,
modify the saved character, or apply its voice recording to an image.

In **Studio → Video → Animate → Use a character**, choose a single view or
upload a character image. Maestro can automatically prepare the source frame
with Flux 2 Klein before Viggle animates it. See the
[Viggle character preparation guide](Viggle-Animate.md#automatic-character-preparation).

## Backward-compatible file contract

Maestro preserves the [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod)
visual contract, reviewed at commit
`d0aea3429393cb8517ce9720a54c539371f0f363`:

- `latent`: normalized H3 visual tensor `[1, 24, T, H, W]`.
- `__metadata__.refmod_meta`: the original JSON metadata, format version 2.
- `__metadata__.maestro_character_meta`: optional JSON extension, versioned
  independently with `schema_version: 1`.
- `maestro.visual.bytes`: the original image/video file bytes in a one-dimensional
  `uint8` tensor.
- `maestro.audio.0.bytes`: optional original saved audio file bytes in a
  one-dimensional `uint8` tensor.
- `maestro.image_views.view-NNNN.bytes`: optional selected PNG file bytes in
  one-dimensional `uint8` tensors. The extension's `image_views` object has
  `version: 1`, `source`, `items`, `selected_ids`, `cover_id` and optional `notes`.
  Each item contains an `id` and the same `tensor`, `encoding`, `extension` and
  `type` media fields used below. Import verifies the PNGs and derives their size.
- `visual_is_preview`: optional extension flag identifying a reconstructed
  compatibility preview, so it is not mistaken for original source footage.

Example extension (stored as a JSON string in safetensors metadata):

```json
{
  "schema_version": 1,
  "name": "Blaine",
  "description": "Tutorial presenter",
  "visual": {
    "tensor": "maestro.visual.bytes",
    "encoding": "file",
    "extension": ".png",
    "type": "image"
  },
  "audio": [{
    "tensor": "maestro.audio.0.bytes",
    "encoding": "file",
    "extension": ".wav",
    "type": "audio",
    "role": "voice"
  }]
}
```

Audio is not in `refmod_meta` and is not disguised as an H3 visual latent or
LoRA weight. No executable objects, filesystem paths or model weights are needed
to deserialize the embedded recording.

The reviewed upstream loader reads `latent` and ignores extra tensors, so it can
use the visual portion of a `.maestro.safetensors` file. It does **not** use the
embedded voice. Its original save operation writes only visual data and will
drop Maestro's extension if the file is loaded and saved again there. Keep the
original portable file when sharing audio or selected images. Earlier Maestro
versions can read the visual and voice portions and ignore the optional image
gallery. Maestro preserves unknown metadata
and extra tensors when importing and re-exporting accepted files.

## API

All endpoints are relative to the running Maestro server:

| Request | Purpose |
| --- | --- |
| `GET /api/v1/characters` | List saved characters, including local visual/voice paths and optional RefMod metadata |
| `POST /api/v1/characters/import` | Multipart `file`, optional legacy `metadata_file`; returns transfer ID |
| `GET /api/v1/characters/remote-files?url=…` | List candidate safetensors filenames from Hugging Face |
| `POST /api/v1/characters/import-url` | JSON `url`, `filename`, optional `name`; returns transfer ID |
| `PUT /api/v1/characters/{id}/voice` | JSON `voice_path`, obtained from Maestro's audio upload API |
| `POST /api/v1/characters/{id}/images/recover?force=false` | Recover/cache images; returns transfer ID; use `force=true` to rebuild |
| `PUT /api/v1/characters/{id}/images` | JSON `selected_ids` and `cover_id`; returns the updated character |
| `GET /api/v1/characters/{id}/images/{view_id}` | Serve a full-resolution PNG; `?download=true` downloads it |
| `GET /api/v1/characters/{id}/images.zip` | Download the saved selection as a PNG ZIP |
| `POST /api/v1/characters/{id}/export` | Start export; returns transfer ID |
| `GET /api/v1/character-transfers/{id}` | Poll `running`, `completed`, or `failed`; includes progress or error |
| `GET /api/v1/characters/{id}/export-file` | Download the completed export with the character filename |

Export completion includes `result.character`, `result.filename` and
`result.url`. Imports include `result.character`. Transfers continue on the
server while the user is on another page; after an export completes, its file
can be downloaded from the export-file endpoint. Recovery completion includes
`result.character.image_views`, with image URLs, dimensions, selection and cover.
Use the returned IDs when updating the selection, for example:

```json
{"selected_ids": ["view-0002", "view-0005"], "cover_id": "view-0002"}
```
