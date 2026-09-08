# H3 Voice Audio

In **Studio → Audio → Speech**, select **H3 Voice Audio — Pruned**.

The **Characters** picker includes Blaine and other characters from
Reference mode. Choose **Add voice** to fill a speaker slot, or save a new
character directly from Speech. See [saved characters in TTS](TTS-Characters.md)
for speaker labels, settings restoration, and support across speech models.

- **Maximum output duration:** 5–300 seconds, with 15 seconds selected by default.
- **Maximum individual generation:** 45 seconds. Long scripts split at sentence or word boundaries and assemble automatically into one audio file.
- **Output:** 32 kHz stereo audio, without video decoding.
- **Speech pacing:** planned around 2.8 words/second, allowing up to 3 when the selected budget is tight. If the complete script cannot fit, Maestro asks for a longer duration or shorter script before queueing; it does not silently drop the ending.

Duration is a ceiling for speech. A short line does not become five minutes of
speech or silence when you choose 5m. Each speaker turn is generated separately,
and Whisper trims unwanted speech and long silence around the requested line.
Natural pauses are inserted between turns. Alignment is conservative: uncertain
transcription leaves the segment intact instead of cutting potentially intended
words. It cannot guarantee removal of every hallucinated word.

## Prompts

Plain text is spoken by one speaker. Long monologues split automatically.
For dialogue, use one `Speaker 1:` or `Speaker 2:` block per turn. Put language
and acting directions in square brackets; those directions are not spoken.

```text
Speaker 1: [English, friendly and conversational] Welcome to Maestro. Let me show you how I organize a project and build a scene.
Speaker 2: [English, curious] Can I reuse the same character across several clips?
Speaker 1: [English, reassuring] Yes. Save your character in the library, then select that character when you create the next shot.
```

Audio Reference 1 belongs to Speaker 1; Audio Reference 2 belongs to Speaker 2.
Each uploaded reference must be at least two seconds long. References share
a 15-second budget, separate from the output duration; when two uploaded
references exceed that budget, each is trimmed to at most 7.5 seconds.
Without an uploaded reference, a speaker's first generated turn supplies the
voice reference for subsequent turns. Reference files are temporary and are
removed on completion, cancellation or failure.

A complete native H3 prompt with `<d>...</d>` dialogue still works for a
single pass up to 45 seconds. For a longer request, Maestro extracts the
literal dialogue tags and `(S1)` / `(S2)` speaker bindings and compiles separate
turns. Put important delivery directions inside the dialogue's square brackets
so they survive splitting; visual scene directions are not carried into the
audio-only turns.

For non-speech audio, start with `Sound:`:

```text
Sound: Gentle rain on a wooden roof, with distant wind through trees. No speech or music.
```

Sound requests use the requested duration, split into bounded generations when
needed. Separate sound segments can vary at their joins; this is not a
continuous soundtrack extension system. Native prompts without dialogue tags
are handled as descriptive audio in the same way.

Whisper medium is downloaded automatically through Maestro's existing model
asset system if it is missing, and participates in model offloading. It runs
after the audio generation passes to avoid reloading the H3 model between
every turn. Cancellation stops the job without publishing a partial output.

## API

`POST /api/v1/generate` accepts the same model and prompt. `duration_seconds`
is the total maximum, not the length of every segment. Use
`GET /api/v1/status/{job_id}` for progress and the final output filename.

```json
{
  "model_type": "minimax_h3_voice_audio",
  "prompt": "Speaker 1: [English, warm] Welcome to Maestro.\nSpeaker 2: [English, cheerful] Let us make something together.",
  "duration_seconds": 300,
  "num_inference_steps": 20,
  "seed": 701
}
```

The embedded audio metadata reports actual output duration and `audio_segment_count`;
restorable Studio settings retain the selected duration ceiling.
Requests over 300 seconds are rejected before queueing. The 45-second limit
is enforced again inside the individual generation path, independently of the UI.

## Validation

The focused regression suite passed all **576 tests**. Automated coverage checks
duration boundaries, complete long-script preservation,
sentence and Unicode splitting, 2.8/3 words-per-second pacing, speaker/language
bindings, reference budgets, cleanup, cancellation, conservative Whisper
trimming, five-minute sound assembly, and Studio duration selection/submission.
The TypeScript/Vite production build passed. Live browser checks confirmed the
15-second default, 45-second and five-minute presets, and a custom 20-second
duration surviving submission.

On 5 September 2026, Maestro's live queue generated the following on an RTX 4090
with 24 GB VRAM, using Windows 10 and 20 inference steps per segment:

| Smoke test | Verified result |
| --- | --- |
| Single generation, seed 701 | Exactly 45.000 seconds; one segment. |
| Five-minute sound, seed 706 | Exactly 300.000 seconds; seven bounded segments. |
| Four alternating dialogue turns, seed 707 | 29.661 seconds; four segments, with the first generated voice reference reused when each speaker returned. |

All three saved 32 kHz stereo WAV files with actual duration and segment count
in the embedded metadata. The files are in the **Wan-Port-Validation** workspace.
An in-progress cancellation produced no partial file, and the live API rejected
a 301-second request before queueing.

The dialogue transcript contained all four turns, including the final word,
but one phrase differed from the script in the transcript. This check verifies
assembly and boundary handling, not guaranteed verbatim speech or voice quality.
Whisper trimming cannot repair wording substitutions within generated speech.

The assembly and Whisper alignment are adapted from
[Wan2GP v12.71](https://github.com/deepbeepmeep/Wan2GP/blob/1e1dd2757f24923f008593d9d4ec09062234be20/models/minimax_h3/dialogue.py).
