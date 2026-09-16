# YuE2 music and personal styles

YuE2 3B is the default in **Studio → Audio → Music** and Director's song
generator. It produces 48 kHz stereo audio from lyrics and a music style. Installing
or updating to v2.2.0 enables and selects it once. If you choose another music
model afterward, Maestro remembers that choice across refreshes and restarts.
MiniMax Music3 and ACE-Step remain available with their own settings and models.

## Make a song

1. Open Music with **YuE2 3B** selected. Its weights download when first needed;
   selecting it as the default does not start a download by itself.
2. Enter the words to sing, with section labels such as `[Verse]` and `[Chorus]`.
   Describe language, genre, instruments, mood, voice and tempo in **Music Style**.
3. Start with **Direct generation** (the default) and 32 synthesis steps. It skips
   the written score. Optionally choose **Melody and chords** to plan a tune and
   harmony first, or **Melody only** to leave more freedom for accompaniment.
4. Set the maximum duration and Generate or add the job to the queue.

Duration is a ceiling. YuE2 can finish before it, and a short ceiling can truncate
the song. Longer lyrics need enough time to sing. Changing the seed tries a new
performance. Prompt enhancement can write or adapt lyrics and style for YuE2;
it does not rewrite a manually supplied ABC score.

Outputs retain their source/enhanced inputs, seed, planning mode, generated score,
native sample rate/channel count and model revision in their sidecar metadata.
The gallery information panel exposes the score and any saved style used. WAV
outputs remain lossless and can be imported into the Editor or selected as a
Director soundtrack. Export settings can intentionally change the delivery format.

## Score or source song

Expand **Score and source song** to enter compatible ABC notation or choose a
recording. Both require a planning mode; they are unavailable in Direct generation.
Choosing a source song downloads the optional SheetSage2/MERT2 transcription model
on first use. It extracts musical notes and harmony, not lyrics. Supply the matching
lyrics separately. An uploaded source takes precedence over a typed score.

The first cover request also installs the small optional notation packages
(`mir_eval`, `pretty_midi`, `mido`, and `importlib_resources`) if missing. It uses
Maestro's existing environment without upgrading Torch or its other AI packages.

This creates a new performance. It does not preserve a source waveform or guarantee
the same singing voice. A source song is separate from training a reusable style.

## My music (Experimental)

**My music** contains a saved-style library, preparation/training projects and
adapter import. Training is optional and experimental. Start with clean, original
recordings, accurate lyrics and a modest training run, then listen to held-out
auditions before committing to a longer run.

Training can influence musical style and vocal characteristics, but matching a
specific singer's voice consistently across new songs is not guaranteed. A close
reconstruction of a training recording does not establish the same result with new
lyrics. Treat saved styles as creative experiments and compare actual generated
songs, including lyrics and recordings held out of training.

1. In **Train**, add at least two different recordings. Enter their actual lyrics
   and section labels and a style caption for each. Mark at least one recording
   as held out; it is excluded from gradient updates.
2. Create a named project and **Prepare audio**. This downloads the optional
   real-audio tokenizer, MERT features, matched NAR adapter, full-precision training
   model and regularization data as needed. It keeps source copies unchanged and
   caches their semantic tokens.
3. Choose the bounded step count and adapter rank, then train. Preparation and
   training use the normal GPU queue, so they wait for generation instead of
   competing with it. Training currently requires at least 20 GB of detected VRAM;
   the exercised development configuration is a 24 GB RTX 4090. Larger contexts
   and ranks need more memory; support on smaller cards is not established.
4. Cancel when needed. The worker completes its current optimizer step and writes
   a resumable checkpoint. Preparation caches survive cancellation. After restart,
   use **Resume** with the same dataset, rank, seed and learning rate; increasing
   the total step count is supported.
5. Audition a saved checkpoint. It appears in **Music style**, paired with the v4
   tokenizer's matching NAR adapter. Generate new lyrics, then compare with **Base
   YuE2** using the same seed, lyrics, duration and synthesis settings. Check
   intelligibility, style retention, repetition and whether the style works beyond
   the training songs. Validation loss alone does not establish listening quality.

Saved styles use Direct generation because this real-audio semantic dialect is
not the score-planning dialect. Selecting one clears incompatible score/source-song
conditioning. The strength slider adjusts the trained AR adapter; the matching NAR
adapter and tokenizer remain paired for decoding compatibility.

The standard training button trains an AR style adapter using the supplied v4
tokenizer and NAR adapter. The optional audio-adaptation workflow below also
trains a personal NAR adapter. Neither changes the foundation model, MERT encoder
or tokenizer head. Files, cached tokens, optimizer state and checkpoints stay locally in
`app/settings/music_training`; reusable styles live in `app/settings/music_styles`.
These paths, uploaded recordings and downloaded model weights are Git-ignored.

### Lyric timing and source sound (experimental)

Use **New experiment with these recordings** to try another training objective.
It copies preparation into a separate project and starts with no trained weights;
the original checkpoints and resume state remain available.

**Align lyrics** separates vocals and force-aligns the supplied words. This first
implementation supports Latin-script lyrics and displays the proportion of
confident words. Review the lyrics when confidence is low. **Use lyric timing
when training** adds the v4 recipe's auxiliary cursor objective (weight 0.08).
Only confident word spans contribute; tags, instrumental tracks, long intros and
silent gaps do not. Existing AR runs retain their original objective; enabling
timing requires a new experiment. The cursor head is saved for resume but is not
needed in exported styles or ordinary playback.

Under **Adapt source sound**, prepare 48 kHz stereo audio targets and start a
short audio training run. The frozen official VAE encoder supplies deterministic
25 Hz targets. Training begins with the paired v4 NAR adapter, retains the fixed
tokenizer, and learns NAR rank-32 adapters plus input/output projections. A pinned
41 MB subset of 32 minted training recordings and four held-out controls provides
regularization (25% of updates). Source holdouts never receive gradient updates.
Evaluation uses fixed windows, noise and flow times so checkpoints are comparable.

Audio checkpoints save separately every 100 steps and when stopped. Choose base
YuE2 or a music checkpoint from this project as conditioning; the exported audio
style automatically includes that same AR configuration. Changing conditioning
requires a new experiment. Lower flow loss is a diagnostic, not evidence of vocal
similarity or intelligible new-song generation. Listen to held-out reconstruction
and new lyrics before extending training. Tokenizer-head adaptation is not included.

## Import and export

Export a saved style as a ZIP containing its manifest, AR weights and NAR weights.
Importing it creates a new library entry. Maestro verifies the declared model and
tokenizer revisions, checksums, adapter shapes and filenames before using it.

Import also accepts the supported upstream AR `.pt` or `.safetensors` layout and
an optional matching NAR adapter. Without a NAR file, Maestro obtains the pinned
Mothersuperior v4 NAR adapter. `.pt` imports use restricted weights-only loading;
arbitrary training programs are not executed. An incompatible dialect, rank or
tensor layout produces an import error rather than silently skipping weights.

## Source reconstruction diagnostic

Before extending an unsuccessful training run, reconstruct the prepared source
tokens to check which musical characteristics survive tokenization and decoding.
This developer diagnostic is available through the API and normal GPU queue.

First import the desired saved checkpoint with `audition-style`. Then POST to
`/api/v1/music-training/projects/{project_id}/reconstruct`:

```json
{"style_id": "SAVED_STYLE_ID", "seconds": 60, "seed": 22005, "steps": 32}
```

Optionally supply `track_ids` to select 1–4 recordings from the project, or
`artist_strength` (0–1.5, default 1) for a controlled comparison at lower strength. Each
comparison uses the first 10–60 seconds (or the available source length), fixed
prepared semantic tokens and identical noise/decoder settings. It saves three
48 kHz stereo WAVs per recording in the workspace selected when queued:

- `original`: an excerpt of the source recording.
- `adapter-off`: source tokens decoded with the base AR and matching v4 NAR adapter.
- `adapter-on`: the same source tokens with the personal AR adapter at the selected strength,
  retaining the same v4 NAR adapter.

The AR model still supplies acoustic conditioning in both cases; it does not
sample new semantic tokens. Both paths share ordinary generation's acoustic/VAE
code and current model precision/profile. The job returns the media filenames
and a reconstruction report with source/token/adapter hashes and settings.
Source files, prepared data, checkpoint weights and training/resume state are
unchanged. Use the ordinary cancel endpoint to stop a diagnostic.

Listen for rhythm, melodic contour, phrasing, instrumentation and vocal character.
This is lossy resynthesis, so waveform identity is not a quality criterion. If
both reconstructions lose the desired characteristics, investigate preparation
and decoding. If only `adapter-on` deteriorates, investigate the adapter and its
conditioning. Good reconstructions with weak new-song results point toward
training generalization or generation conditioning rather than lost source data.

## Review recordings before training

In **My music → Train a style**, open **Recordings, captions & lyrics**. Play each
recording beside its caption and lyrics. Describe audible genre/pace, voice and
delivery, instruments/rhythm, and mood/production for that particular recording.
Match lyrics to the actual version or excerpt, including repeated sections and
spoken intros. Use `[Instrumental]` when there are no vocals.

**Draft lyrics from recordings** queues the existing local Whisper model behind
other GPU work. Empty transcriptions are retried once without the speech filter;
an empty retry is explicitly marked as unrecognized. Timestamped drafts highlight uncertain passages and never
replace supplied lyrics automatically. Whisper can mishear singing or miss
sections; it does not generate instrumentation or voice captions. Listen, then
use **Review draft in an edited copy** or **Edit captions and lyrics in a new
project**. Original labels, recordings and checkpoints remain intact.

The review checkbox is tied to the exact audio identity, caption, lyrics and
split. Editing a draft invalidates its review. An optional **Original song** name
groups related excerpts: the same song cannot appear in both training and the
held-out set. After preparation, each recording shows how much audio fits the
12,288-token AR training sequence, accounting for its text. Shorten any truncated
recording and supply matching excerpt lyrics if its ending should be learned.

## Automatic checkpoint auditions

Enable **Automatically audition checkpoints** in the training project and enter
a short, fixed test caption, new lyrics, seed, maximum length (8–60 seconds) and
style strength. This is optional and adds generation time. The saved request is
independent of the Studio prompt and remains fixed during that training job.

At each saved checkpoint (every 200 style steps, every 100 sound-adaptation steps,
and the final step), training saves its optimizer and random state, releases its
models, renders the test through the normal YuE2 generator, then resumes that
exact training state. The whole job uses Maestro's existing GPU queue; training
and inference do not overlap. Stopping during a preview keeps the checkpoint.
A failed preview is displayed beside its checkpoint and can be retried while
training continues to its requested total.

**Checkpoint comparisons** contains playable local FLAC samples and their request
IDs, settings and generation engine. Compare the same request ID within the same
training stage/engine; changing lyrics, caption, seed, length or strength creates
a different comparison. The time is a maximum, so a sample may finish early or
end at its limit. Style checkpoints use the released v4 acoustic companion;
sound-adaptation checkpoints use the exact AR checkpoint used during adaptation.
These previews do not add duplicate styles to the library or media to the gallery.
Use **Render a saved checkpoint now** to compare existing checkpoints, or the
existing **Use for audition** button to select a style for ordinary generation.

Two-step CUDA regression coverage verifies identical final AR adapter tensors,
optimizer state and Python/PyTorch random state with and without preview
interleaving. This verifies training continuity, not musical or singer quality.
Choose checkpoints by listening as well as held-out loss.

## API

The following endpoints use the running Maestro base URL:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/music-styles` | List saved styles. |
| `POST /api/v1/music-styles/import` | Multipart `name`, `trigger`, `ar`, optional `nar`; `ar` can be an exported ZIP. |
| `GET /api/v1/music-styles/{id}/export` | Download a portable style bundle. |
| `GET /api/v1/music-training/projects` | List local projects and durable progress. |
| `POST /api/v1/music-training/projects` | Create `{name, trigger, tracks}`. Each track supplies `audio_path`, `lyrics`, `style`, `holdout`. |
| `GET /api/v1/music-training/projects/{id}` | Project, checkpoint and resume details. |
| `POST /api/v1/music-training/projects/{id}/prepare` | Queue tokenization with `{}`. |
| `POST /api/v1/music-training/projects/{id}/fork` | New experiment with the same sources and reusable preparation. |
| `POST /api/v1/music-training/projects/{id}/align-lyrics` | Queue vocal separation and lyric alignment with `{}`. |
| `POST /api/v1/music-training/projects/{id}/review-data` | Queue independent local Whisper lyric drafts with `{}`; preserves source labels. |
| `POST /api/v1/music-training/projects/{id}/review-track` | Mark `{track_id, reviewed}` against the current recording and labels. |
| `GET /api/v1/music-training/projects/{id}/recordings/{track_id}` | Play a recording belonging to this project. |
| `POST /api/v1/music-training/projects/{id}/prepare-audio` | Queue stereo source-audio targets with `{}`. |
| `POST /api/v1/music-training/projects/{id}/train` | Queue `{steps, rank, seed, learning_rate, resume, lyric_alignment?}`. |
| `POST /api/v1/music-training/projects/{id}/adapt-audio` | Queue `{steps, seed, learning_rate, resume, conditioning_checkpoint?}`; default 200 steps, base AR. |
| `POST /api/v1/music-training/projects/{id}/audition-style` | Save `{checkpoint}` or `{audio_checkpoint}` into the style library. Audio exports include matching conditioning. |
| `POST /api/v1/music-training/projects/{id}/render-audition` | Queue `{branch: "style" or "audio", checkpoint, audition}` using a saved checkpoint. |
| `GET /api/v1/music-training/projects/{id}/auditions/{audition_id}` | Play a completed checkpoint preview. |
| `POST /api/v1/music-training/projects/{id}/reconstruct` | Queue a fixed-token comparison with `{style_id, seconds, seed, steps, track_ids?}`. |
| `POST /api/v1/cancel/{job_id}` | Cancel a queued/running music job. |

Both `train` and `adapt-audio` accept an optional `audition` object:

```json
{"enabled": true, "style": "Solo voice, acoustic guitar, intimate folk",
 "lyrics": "[Verse]\nA new morning on an open road",
 "seconds": 30, "seed": 22005, "strength": 1.0}
```

Omitting it or supplying `{"enabled": false}` retains ordinary training without
previews. Sampling is fixed to direct mode, 32 decoder steps, temperature 0.8,
top-p 0.95, top-k 64, CFG 1 and 512-frame VAE tiles; returned sample metadata
records actual audio length, engine and model/adapter identities. Project track
inputs also accept `source_song` and `reviewed` for the review workflow.

Reconstruction defaults to an audio before/after comparison for a personal audio
checkpoint: `audio-before` uses the released v4 NAR, `audio-after` uses the personal
NAR, with identical source tokens, AR, seed and synthesis settings. Set
`comparison: "ar"` to retain the existing AR off/on diagnostic, or `"audio"` to
request the acoustic comparison explicitly. Reports include the actual adapter
hash for each variant.

Ordinary generation uses the existing generation endpoint with `model_type:
"yue2"`, `generation_mode: "audio"`, lyrics in `prompt`, style in `alt_prompt`,
`duration_seconds`, and `model_mode` (`0` full, `1` melody, `2` direct). A saved style
uses `custom_settings: {artist_id, artist_strength}` in mode 2. Score input uses
`custom_settings.abc`; source-song extraction uses `audio_prompt_type: "A"` and
the uploaded path in `audio_guide`.

## Provenance and usage terms

The integration is based on [Wan2GP's YuE2 implementation](https://github.com/deepbeepmeep/Wan2GP/tree/5c40db6500cc8a142a15ed93720174955587babe/models/TTS/yue2)
and the [official YuE project](https://github.com/multimodal-art-projection/YuE).
Exact runtime and asset revisions are recorded in
`app/models/TTS/yue2/MAESTRO_PORT.md`, `assets.json` and `music_assets.py`.

The [YuE2 model weights](https://huggingface.co/m-a-p/YuE2-3B) and
[Mothersuperior real-audio tokenizer v4 weights](https://huggingface.co/Mothersuperior/yue2-mothersuperior-realaudio-tokenizer-v4)
are labelled **CC BY-NC 4.0**. Their model terms are distinct from the source-code
licenses and from Maestro's other music models. Original code/license notices
remain with the port. Personal adapter bundles record their base/tokenizer revisions
and non-commercial model terms.

Source-audio preparation uses the pinned [official YuE2 VAE](https://huggingface.co/m-a-p/YuE2-Vae).
Lyric timing uses torchaudio's [MMS forced-alignment model](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html)
and [Hybrid Demucs bundle](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.html),
with SHA-256-verified optional weights. No Torch replacement or extra separator
package is installed for this workflow.
