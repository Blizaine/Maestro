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

## Instrumental

Select **Instrumental** to use Mothersuperior's dedicated instrumental AR LoRA
at strength 1. Maestro automatically selects **Melody and chords** (`cot=full`):
YuE2 writes its score before the audio tokens. The original BF16 adapter is
downloaded once on first use (about 140 MB), pinned to a specific revision and
verified by size and SHA-256. This path uses the stock acoustic decoder.

Describe genre, instruments, mood and production in **Music Style**. **Write
Song** uses an instrumental guide with no vocal directions. The normal input is
`[instrumental]`, which lets the model choose its sections. API clients may also
supply bare section tags, one per line: `intro`, `verse`, `pre-chorus`, `chorus`,
`bridge`, `outro`, enclosed in brackets. Timed tags such as
`[intro 0:00-0:15]` are supported. Lyrics or production prose are replaced with
the bare instrumental prompt for this mode; instrument details belong in Style.

Artist LoRAs, manual scores and source-song scoring are paused for instrumental
jobs. Their Studio selections are kept; turn Instrumental off to use them again.
Queued jobs and Director's music generation use the same recipe. Output metadata
records the adapter, strength and planning mode, and Load settings restores the
instrumental selection. Normal vocal generation removes the instrumental adapter.

For API generation, set `custom_settings.instrumental` to `true`. The legacy
`[Instrumental]` prompt also selects this path. Duration remains a maximum;
timed sections do not guarantee exact timing or a natural ending at the cap.
The adapter improves instrumental conditioning but does not guarantee that every
seed is free of vocal-like sounds.

Source and instructions: [Mothersuperior/YuE2-instrumental-cot-full-loras](https://huggingface.co/Mothersuperior/YuE2-instrumental-cot-full-loras).
Weights use the same CC BY-NC 4.0 terms as YuE2.

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

In **My music → Saved LoRAs**, enable **Show in LoRA selector** for the LoRAs
you want readily available. This shortlist is saved across restarts and does
not activate them. Then check a LoRA in **Advanced → LoRAs & presets** to use
it and adjust its strength, just like the image and video LoRA controls.
Unchecking deactivates it; turning off its library toggle also removes it from
the shortlist. Newly saved LoRAs stay in My music until you choose to list them.
A LoRA loaded from a saved job or training audition stays visible while active,
even when it has not been shortlisted.

**My music** remains the place to train, import, search and manage the complete
library. These are LoRA-based bundles with an AR adapter and NAR decoder.
You can select multiple bundles with separate strengths for an **Experimental
mix**. Both apply across the whole song; this does not route individual LoRAs
to particular verses or choruses. Describe the intended singers and delivery
in Music Style, but expect that voices may blend or one may dominate. Start
with a short excerpt to compare. Known v4 and v9 tokenizer versions cannot
be combined; separately adapted voices from the same version can be tested.

AR LoRA effects add at their chosen strengths. Complete sound companions
blend using their relative strengths, including the full input/output weights,
so the last selection does not overwrite the first. Joint AR/NAR LoRAs add
their deltas to both branches. A single complete companion retains its sound
decoder at strength zero, as before; with all such strengths zero, their
decoders blend equally. Uncheck a bundle to remove all of its contribution.
Mixing independently trained sound components is experimental, even when their
base tokenizer version matches. Active bundles, strengths and hashes are saved
with the song, shown in gallery details and restored by Load settings.

Each selected bundle's training trigger is added to the generation style prompt
automatically, and appears beside its strength control. Training auditions add
the same trigger. You can add delivery instructions such as rapping, whispering
or singing to the style prompt; the trigger does not replace those directions.

**Remove** hides a saved LoRA from the library and deselects it in the current
composer. **Undo removal** or **Show removed LoRAs → Restore** brings it back.
Removal retains adapter files for already queued jobs and restoration; training
projects, checkpoints and generated songs are unchanged. It does not free disk
space. Search matches names, triggers and checkpoint names.

### Auto training

Choose **My music → Train a style → New project → Auto** for a run that
prepares the dataset and completes both training stages without clip reviews
or approvals between stages. Guided remains available for reviewing each step.

1. Give the project a name and style trigger, then upload the songs. Keep at
   least one different song **Check only**; it measures learning and never
   becomes training material. The second uploaded song starts checked this way.
2. Choose **Voice & sound steps** and **Song-style steps** up front. The defaults
   are **100 / 200**, a short first listening test. Each target can be 1–1600.
3. Click **Start Auto training**. If another GPU job is running, Auto waits.
   Maestro analyzes the songs, selects suggested clips of the main detected
   voice, prepares the data, trains voice/sound, prepares fresh music tokens
   with that learned sound, trains song style, and saves the matched LoRA.
4. When complete, choose **Use LoRA & try a song** and supply new lyrics. The
   saved LoRA also appears in **Advanced → LoRAs & presets**.

An existing preparation or ordinary training project can **Continue automatically**.
Auto reuses saved preparation and resumable training where available. An already
adapted song-style project links back to its original voice project.

Progress lives in Maestro, so closing the panel, refreshing, or closing the
browser does not stop the run. Keep Maestro itself running. **Stop Auto training**
saves the current training step before releasing the GPU; **Resume Auto training**
continues the unfinished stages with the original targets. After an app restart,
resume explicitly from the project. Failed stages keep completed work for retry.

Afterward, **Open voice training** or **Open song-style training** returns to the
normal controls. Both offer additional steps and older checkpoints
under **Expert settings**; song style offers **Continue song-style training** there.
Training voice further creates a separate matched song-style project when you
choose that new voice version. Already saved LoRAs retain their original weights.

Auto uses the most prominent detected voice per song, not cross-song identity
recognition. Guest vocals, rap and effects can still confuse detection or lyrics.
The dataset summary records automatic selection, and **Open prepared clips**
lets you inspect it afterward. Auto does not mark clips as human-reviewed. If
analysis finds no usable voice clips, it stops with an explanation; adjust the
clips or recordings and resume. Voice training needs at least one 20.5-second
training excerpt and a 5.2-second check excerpt. Once a dataset is built, changing
its source clips creates a separate dataset rather than altering trained weights.

### Prepare full songs automatically (Guided)

In **Train a style → New project**, automatic song preparation is on by default.
Upload 2–50 distinct songs (up to 20 minutes and 1 GB each), name the project and
enter its style trigger. Lyrics are optional at this stage. You can add a music
description for each song; it becomes the initial description of its clips.
Keep at least one separate song held out for evaluation.
One training song and one different check-only song is supported. Only the
training song teaches the adapter; the check-only song measures how it handles
unseen material. Judge a small dataset with new lyrics and multiple seeds.

**Analyze songs** uses the shared GPU queue. It separates vocals for analysis,
transcribes words with timestamps, and groups voices. Completed stages are cached;
**Stop preparation** or an app restart does not discard them. Resume analysis from
the saved project. A running model call may need to finish before stopping.

For each song:

1. Listen to the voice previews. Maestro initially selects the detected voice with
   the most confidently transcribed words. Labels do not identify an artist and
   are independent between songs. A singer can occupy several clusters; select
   all relevant voices and choose **Suggest clips for selected voices**.
2. Open the suggested clips. Suggestions aim for complete 20–30 second phrases,
   avoid other voices and ambiguous overlaps, and leave very short excerpts
   excluded. Listen and adjust start/end times, lyrics, music descriptions and
   **Rap / Sung / Rap and singing** labels. You can add excerpts manually.
3. Check the review box for clips you have listened to, then **Save song review**.
   Changing a voice selection and suggesting again replaces that song's clip edits.
4. After reviewing included clips in all songs, **Create training dataset**, then
   **Open training dataset** to prepare tokens and train as usual.

All excerpts of an original song stay together on the training or held-out side.
The compiled dataset uses the original full mix, not the isolated vocal stem, and
supports up to 500 excerpts. Original audio, other projects and checkpoints stay
unchanged. Editing the preparation draft creates a new dataset when compiled
again. Vocal-only training augmentation is not included.

Transcription and diarization are review aids: backing vocals, fast rap, effects
and instrumental passages can produce errors. If the speaker model is unavailable,
the UI says so and leaves a reviewable unassigned voice instead of claiming to
identify the singer. **Retry voice analysis** reuses the saved stems and lyrics;
it replaces clip edits only for songs whose voice analysis failed. If no words are recognized, enter an excerpt manually or
leave that song's clips excluded. Keep both a training and a held-out song in the
final selection. Turn off automatic preparation to use existing clips and lyrics.

### AI-Toolkit imports and training comparisons

**My music → Import** also accepts a combined AI-Toolkit YuE2 `.safetensors`
LoRA. Supply the style name and the trigger used in training. Leave the separate
audio-adapter field empty for a combined file. Maestro splits fused attention
and MLP projections while preserving the stored delta and any alpha scaling.
It applies the imported AR and NAR adapters to base YuE2, without adding another
generic NAR adapter. The style-strength control scales both imported branches;
zero disables both. Existing separate styles retain their fixed decoder companion.
Files with incomplete or unsupported targets are rejected. Exported ZIPs preserve
these differences, and older Maestro bundles remain usable.

**Train music and sound together · experimental** learns AR song structure and
NAR rendering in one run. It uses full-song token loss, up to 60-second acoustic
crops, a detached AR conditioning cache, rank-32 adapters by default, and 25%
minted regularization. It starts with base YuE2, freezes tokenizer and I/O weights,
and saves a matched pair every 100 steps and when stopped. **Stop after current
step** saves both adapters and optimizer/random state. The ordinary style and
sound-adaptation trainers remain available with their existing settings.
This is a score-free comparison recipe, not a reproduction of an external
trainer's unspecified settings. Step counts differ in data exposure between recipes.

**Joint training starting point** can select a saved style with the same
tokenizer pair and exact trigger as the project. This refines both existing
adapters and saves a **Joint baseline** before the first update. The original
style stays unchanged. Saved adapter ranks, independent projection matrices,
decoder I/O and strength behavior are preserved; the initial learning rate is
0.00002. New base-model runs retain their original settings. Resume restores
the saved run, including its starting-style identity and decoder weights.

New projects default to **v9 · recommended for new projects** for the matched
Mothersuperior tokenizer head and audio decoder. v4 remains selectable in
**Expert setup**. Existing v4 projects offer, under **Expert settings**,
**Compare v9 in a new project**: original recordings and checkpoints are retained,
but preparation must run again for the new token dialect. The pair is fixed for
each project and follows its checkpoints into auditions, reconstruction and
exported styles. No existing project silently changes its tokenizer.

For comparisons, use the same recordings, held-out split, new lyrics, seed,
duration and strength. Keep settings fixed across checkpoints; listen for
pronunciation, rhythm, vocal character and distortion. A lower loss or a newer
tokenizer does not establish singer-identity accuracy.

### Standard workflow

**My music** contains a saved-style library, preparation/training projects and
adapter import. Training is optional and experimental. Start with clean, original
recordings, accurate lyrics and a modest training run, then listen to held-out
auditions before committing to a longer run.

Training can influence musical style and vocal characteristics, but matching a
specific singer's voice consistently across new songs is not guaranteed. A close
reconstruction of a training recording does not establish the same result with new
lyrics. Treat saved styles as creative experiments and compare actual generated
songs, including lyrics and recordings held out of training.

1. In **Train a style**, prepare full songs using the workflow above, or turn off
   automatic preparation and add existing clips with their lyrics, section labels
   and style captions. Keep at least one original song held out; it is excluded
   from gradient updates.
2. **Learn voice & sound:** choose **Prepare voice training**, then **Train voice
   & sound**. The first run defaults to 100 steps so you can test a short run
   before extending it. It adapts the tokenizer and sound decoder together.
   Preparation downloads the optional models and reference data as needed.
3. **Continue to song style** uses the latest saved sound version automatically.
   Maestro opens its existing song-style project if one exists, or creates one
   with the matching learned sound. Choose **Prepare song-style training**, then
   **Train song style**. New runs default to a 200-step first listening checkpoint.
   These short defaults are starting comparisons, not established optimal counts.
   Your supplied lyrics are always used. This stage does not ask you to retrain
   the voice or choose an adapter rank.
4. Cancel when needed. The worker completes its current optimizer step and writes
   a resumable checkpoint. Preparation caches survive cancellation. After restart,
   use **Resume** with the same dataset, rank, seed and learning rate; increasing
   the total step count is supported.
5. **Use LoRA & try a song** saves/selects the latest song-style checkpoint and
   returns to Studio. Enter new lyrics and Generate. It also appears in
   **My music → Saved LoRAs** and
   can be selected under **Advanced → LoRAs & presets**, paired with the
   project's matching NAR adapter. Generate new lyrics, then compare with base
   YuE2 by disabling the LoRA, using the same seed, lyrics, duration and synthesis settings. Check
   intelligibility, style retention, repetition and whether the style works beyond
   the training songs. Validation loss alone does not establish listening quality.

**Expert settings** keeps exact total step targets, older checkpoints and sound
comparisons, alternate training methods, and new experiments available. Resume
keeps a run's rank, objective, seed, learning rate and unfinished target. Once its
target is reached, the next suggested target adds 100 sound or 200 song-style
steps, capped at 1600. A song-style project with learned sound links back to the
original voice-training project; it does not offer a fresh voice stage in place.
Existing AR-only and experimental projects keep their original workflow.

Rank controls how many parameters a LoRA can learn, not a quality score. Maestro
keeps rank 64 for the song-writing (AR) adapter and rank 32 for the voice/sound
decoder (NAR). The upstream author also uses these different ranks for these
different components. Comparing a rank-32 LoRA from another training method to
an AR rank-64 run alone does not establish which will sound better.

Preparation and training use the normal GPU queue. Training currently requires
at least 20 GB of detected VRAM; the exercised development configuration is a
24 GB RTX 4090. Existing recordings and checkpoints remain unchanged.
**Test songs & comparisons** contains optional automatic checkpoint auditions.

Saved styles use Direct generation because this real-audio semantic dialect is
not the score-planning dialect. Selecting one clears incompatible score/source-song
conditioning. The strength slider adjusts the trained AR adapter; the matching NAR
adapter and tokenizer remain paired for decoding compatibility.

The song-style stage trains an AR adapter using the project's tokenizer and NAR
adapter. Guided voice/sound adaptation updates the tokenizer head and NAR
adapter first; alternate methods below remain separate experiments. Training
leaves the foundation model and MERT encoder unchanged. Files, cached tokens,
optimizer state and checkpoints stay locally in
`app/settings/music_training`; reusable styles live in `app/settings/music_styles`.
These paths, uploaded recordings and downloaded model weights are Git-ignored.

### Lyric timing and source sound (experimental)

Use **Expert settings → New experiment with these recordings** to try another training objective.
It copies preparation into a separate project and starts with no trained weights;
the original checkpoints and resume state remain available.

**Align lyrics** separates vocals and force-aligns the supplied words. This first
implementation supports Latin-script lyrics and displays the proportion of
confident words. Review the lyrics when confidence is low. In **Expert settings →
Extra word-timing guidance**, **Add extra word-timing guidance** adds the v4
recipe's auxiliary cursor objective (weight 0.08). It is off for new runs by
default. It is not an option to include or exclude lyrics: ordinary song-style
training already uses them.
Only confident word spans contribute; tags, instrumental tracks, long intros and
silent gaps do not. Existing AR runs retain their original objective; enabling
timing requires a new experiment. The cursor head is saved for resume but is not
needed in exported styles or ordinary playback.

Choose **Expert settings → Training method → Other training experiments** to
access joint training and the separate decoder-only experiment. Under **Adapt
source sound**, prepare 48 kHz stereo audio targets and start a
short audio training run. The frozen official VAE encoder supplies deterministic
25 Hz targets. Training begins with the project's paired NAR adapter, retains the fixed
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
- `adapter-off`: source tokens decoded with the base AR and matching NAR adapter.
- `adapter-on`: the same source tokens with the personal AR adapter at the selected strength,
  retaining the same NAR adapter.

Joint checkpoints instead compare both personal adapters off/on against base
YuE2, without layering in a generic NAR adapter. The report labels this as a joint
comparison. A style must match the project's tokenizer revision.

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

At each saved checkpoint (every 200 style steps, every 100 sound-adaptation or joint steps,
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
end at its limit. Style checkpoints use the project's matched acoustic companion;
sound-adaptation checkpoints use the exact AR checkpoint used during adaptation.
These previews do not add duplicate styles to the library or media to the gallery.
Use **Render a saved checkpoint now** to compare existing checkpoints, or the
existing **Use for audition** button to select a style for ordinary generation.

Two-step CUDA regression coverage verifies identical final AR adapter tensors,
optimizer state and Python/PyTorch random state with and without preview
interleaving. This verifies training continuity, not musical or singer quality.
Choose checkpoints by listening as well as held-out loss.

## Author voice and sound adaptation (experimental)

**My music → Train a style** defaults to the guided voice/sound → song-style
workflow. **Expert settings → Training method** exposes alternatives:

- **Voice & sound, then song style:** adapt the real-audio tokenizer head and
  audio decoder together, compare the sound, select a saved pair, and train the
  song-writing model (AR) in a new project using freshly prepared tokens.
- **Song style only (AR):** learn musical structure, lyric delivery and style
  with the project's existing sound pair. This does not train the decoder.
- **Other training experiments:** the earlier decoder-only and joint AR/NAR
  experiments. They are different objectives from the author's head/NAR method.

Start author adaptation at 100 steps. Matched head/decoder checkpoints are saved
every 25 steps and when stopping. **Compare original / before / after** queues
up to 30 seconds from one check-only excerpt and one training excerpt. The
gallery receives the source and two reconstructions, each using its own head
and matching decoder, the same seed, and no AR style adapter. This isolates
what the sound pair can preserve; it does not predict a new song's voice quality.
Select **Continue to song style** to use the latest sound version, or **Expert
settings → Saved sound versions & comparisons → Train song style with this
version** to select an earlier checkpoint. This reopens an existing matching
song-style project or creates one if needed. Existing
tokens, projects and saved styles remain intact. Its auditions and exported
styles use the adapted decoder, with both pair hashes recorded in the manifest.

To extend sound adaptation, use **Open original sound training** from the
song-style project's **Expert settings**. That opens the project containing the
sound-training optimizer and checkpoints. In its **Expert settings**, set **Sound adaptation steps** to the
desired total (400 after 100 completed adds 300 steps), then choose **Resume
voice & sound adaptation**. The song-style project contains a saved sound pair,
not the original sound-training resume state.

This ports Mothersuperior's published `scripts/joint_v6.py` objective at revision
`e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5`: gradients through frozen AR conditioning
into the tokenizer, NAR LoRA plus full input/output projections, real-audio flow
loss, and a frozen differentiable VAE for log-mel, multiresolution spectral and
stereo-width losses. Waveform loss runs on six-second crops at noise levels up
to 0.4, with weight 4. Minted token-neighbor and flow losses provide anchors.

This is a bounded personal adaptation, not a reproduction of the historical
author run: it starts with the chosen matched pair (v9 by default), uses 32
minted training tracks and 4 disjoint check tracks rather than 4,000/200, and
accumulates the 16-anchor head batch in microbatches of 2. The resumable learning
rate uses a fixed 3,000-step schedule. First preparation downloads about 1.1 GB
of reference audio plus any missing models. Training windows are 512 frames
(20.48 seconds); shorter training clips are skipped for this stage and remain
usable for AR training. Check excerpts need at least 128 frames (5.12 seconds);
short checks use a smaller aligned waveform crop. Listening matters more than
a small change in one numerical loss; specific singer matching remains unproven.

### What “check only / held out” means

A check-only song is never used to update model weights. It asks whether the
learning helps on a different recording instead of merely memorizing its
practice songs. It is not a voice reference used during generation. Maestro
requires a separate song to make that check possible; all excerpts from one
song stay on the same side of the split. A clean, representative passage of
20–30 seconds is more informative than a very short phrase. Training needs
separate examples of the vocal delivery you want to learn.

Song preparation's **Full song & transcription** section lets you choose a
language (for example `en`) and rescan that song. Long songs are transcribed in
overlapping one-minute sections so the opening does not choose the language for
the entire track. Rescanning preserves reviewed/manual clips and their lyrics;
new suggestions need review. Sparse recognition is flagged. Voice groups are
local clustering suggestions, not named people; one singer may occupy several
groups. Listen to the full song and examples before choosing groups.

## API

The following endpoints use the running Maestro base URL:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/music-styles` | List saved music LoRAs; `?include_archived=true` includes removed entries. |
| `PATCH /api/v1/music-styles/{id}` | `{in_selector: true}` shortlists a LoRA in Advanced without activating it; `false` unlists it. `{archived: true}` removes a library entry; `false` restores it. Weights stay available to existing jobs. |
| `POST /api/v1/music-styles/import` | Multipart `name`, `trigger`, `ar`, optional `nar`; `ar` can be an exported ZIP. |
| `GET /api/v1/music-styles/{id}/export` | Download a portable style bundle. |
| `GET /api/v1/music-training/projects` | List local projects and durable progress. |
| `POST /api/v1/music-training/projects` | Create `{name, trigger, tracks, tokenizer_pair?}`; pair is `v9` (default) or legacy `v4`. Each track supplies `audio_path`, `lyrics`, `style`, `holdout`. |
| `GET /api/v1/music-training/projects/{id}` | Project, checkpoint and resume details. |
| `POST /api/v1/music-training/projects/{id}/prepare` | Queue tokenization with `{}`. |
| `POST /api/v1/music-training/projects/{id}/prepare-pair` | Prepare real features, stereo targets and minted audio anchors. |
| `POST /api/v1/music-training/projects/{id}/adapt-pair` | Queue author adaptation with `{steps, seed, resume}`; defaults to 100 steps. |
| `POST /api/v1/music-training/projects/{id}/reconstruct-pair` | Queue original/before/after sound comparison with `{checkpoint}`. |
| `POST /api/v1/music-training/projects/{id}/select-pair` | Create a separate AR project using `{checkpoint}`; fresh tokenization is required. |
| `POST /api/v1/music-training/projects/{id}/analyze-songs` | Scan a preparation draft; optional `{track_id, revision, language}` rescans one song while preserving reviewed/manual clips. |
| `POST /api/v1/music-training/projects/{id}/fork` | New experiment with the same sources; optional `{tokenizer_pair: "v9"}` requires fresh preparation. |
| `POST /api/v1/music-training/projects/{id}/align-lyrics` | Queue vocal separation and lyric alignment with `{}`. |
| `POST /api/v1/music-training/projects/{id}/review-data` | Queue independent local Whisper lyric drafts with `{}`; preserves source labels. |
| `POST /api/v1/music-training/projects/{id}/review-track` | Mark `{track_id, reviewed}` against the current recording and labels. |
| `GET /api/v1/music-training/projects/{id}/recordings/{track_id}` | Play a recording belonging to this project. |
| `POST /api/v1/music-training/projects/{id}/prepare-audio` | Queue stereo source-audio targets with `{}`. |
| `POST /api/v1/music-training/projects/{id}/train` | Queue `{steps, rank, seed, learning_rate, resume, lyric_alignment?}`. |
| `POST /api/v1/music-training/projects/{id}/adapt-audio` | Queue `{steps, seed, learning_rate, resume, conditioning_checkpoint?}`; default 200 steps, base AR. |
| `POST /api/v1/music-training/projects/{id}/train-joint` | Queue `{steps, rank, seed, learning_rate, resume, window_frames?, initial_style_id?}`; defaults to 200 steps, rank 32, 1500 audio frames. A saved starting style retains its own AR/NAR ranks and defaults to LR 0.00002. Requires prepared tokens/audio and 20 GB VRAM. |
| `POST /api/v1/music-training/projects/{id}/audition-style` | Save `{checkpoint}`, `{audio_checkpoint}` or `{joint_checkpoint}` into the style library. Exports retain the matching pair. |
| `POST /api/v1/music-training/projects/{id}/render-audition` | Queue `{branch: "style", "audio" or "joint", checkpoint, audition}` using a saved checkpoint. |
| `GET /api/v1/music-training/projects/{id}/auditions/{audition_id}` | Play a completed checkpoint preview. |
| `POST /api/v1/music-training/projects/{id}/reconstruct` | Queue a fixed-token comparison with `{style_id, seconds, seed, steps, track_ids?}`. |
| `POST /api/v1/cancel/{job_id}` | Cancel a queued/running music job. |

`train`, `adapt-audio` and `train-joint` accept an optional `audition` object:

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
checkpoint: `audio-before` uses the project's matched NAR, `audio-after` uses the personal
NAR, with identical source tokens, AR, seed and synthesis settings. Set
`comparison: "ar"` to retain the existing AR off/on diagnostic, or `"audio"` to
request the acoustic comparison explicitly. Reports include the actual adapter
hash for each variant.

Ordinary generation uses the existing generation endpoint with `model_type:
"yue2"`, `generation_mode: "audio"`, lyrics in `prompt`, style in `alt_prompt`,
`duration_seconds`, and `model_mode` (`0` full, `1` melody, `2` direct). A saved style
uses `custom_settings: {artist_loras: [{id, strength}, ...]}` in mode 2. Each
strength is between 0 and 1.5. The legacy `{artist_id, artist_strength}` form
remains supported; an explicit `artist_loras` list takes precedence, and `[]`
clears the selection. Score input uses
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
