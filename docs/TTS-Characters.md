# Saved characters in Speech

In **Studio → Audio → Speech**, open **Saved characters** and choose **Add voice**
beside Blaine or another character. This is the same persistent library used by
Video Reference mode. Maestro loads the saved voice and fills the speaker name;
the portrait identifies the character in the picker.

Use the character's name at the beginning of each dialogue line:

```text
Blaine: Welcome to Maestro. Let me show you how saved characters work.
Yoda: Ready to begin, we are.
```

Plain monologue text also works. `Speaker 1:` and `Speaker 2:` correspond to the
numbered voice slots. Prompt Enhance receives the selected character names.
Each slot has a character selector for replacing its voice, and still accepts
an uploaded audio or video file. Uploading a replacement detaches that slot from
the saved character. Removing a slot clears its audio reference.

## Model support

| Speech model | Saved voice support |
| --- | --- |
| H3 Voice Audio | Two speakers; retains the 45-second segment and five-minute total limits. |
| KugelAudio | Up to six speakers. |
| DramaBox Audio | Two speakers. |
| Scenema Audio | Two speakers through its existing SeedVC reference path. |
| IndexTTS2 | One voice, voice plus emotion reference, or two-speaker dialogue. Adding a second character defaults to dialogue; an explicitly selected voice-plus-emotion mode keeps its second slot as an emotion reference. |
| Qwen3 Base | Two speakers through voice cloning. |
| Chatterbox | One speaker. |
| Qwen3 Custom Voice / Voice Design | The picker offers **Use with clone**, which switches to Qwen3 Base while retaining the script. These variants do not accept cloned voice audio themselves. |

Model switches respect the destination's speaker limit. For example, switching
from KugelAudio to H3 retains the first two slots; switching to Chatterbox retains
the first slot. A preset-only model clears cloning slots. Output **Load settings**
restores selected character IDs, speaker names, original filenames, and references.

## Save a character from Speech

Choose **Save a new character**, enter its name, and select an image or video plus
a voice reference. If the character visual is a video, its audio supplies the
voice when no separate voice file is selected. Use at least two seconds of clear
speech. Maestro saves its own copies, so the character is available in Reference
mode and from other browsers connected to this installation.

Characters saved without voice audio are listed but cannot be used for cloning.
Save a character with a voice reference to use it in Speech. **Refresh saved
characters** reloads changes made elsewhere.

The library uses the existing `GET /api/v1/characters` and
`POST /api/v1/characters` endpoints. Generation uses each model's existing
`audio_guide`, `audio_guide2`, and additional supported voice inputs.

## Local validation

Validated on 5 September 2026 with the production UI build, eight new voice and
character tests, and the existing 576-test regression suite. Coverage includes
all shipped Speech models' reference modes and limits, complete speaker-label
mapping, slot removal and replacement, settings restoration, character creation,
and the character names supplied to Prompt Enhance.

Live browser checks selected Blaine and Yoda in H3, switched Qwen Voice Design
to Base with Blaine while preserving the script, and restored Blaine's character,
original voice filename, and named prompt using **Load settings**. A browser-submitted
H3 run loaded Blaine's library audio and saved 6.283 seconds of 32 kHz stereo WAV
on the RTX 4090. The seed-711 output is in **Wan-Port-Validation**. Other speech
models were checked for request compatibility without downloading or generating
with every checkpoint.
