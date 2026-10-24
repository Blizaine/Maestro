# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
# A two-speaker dialogue with a change of tone on every line. Directions open the lines: cues inside the lines make takes stop speaking more
# often (3 of 8 takes for a script with mid-line cues and sounds, none for plain sentences).
DEFAULT_PROMPT = "Speaker 1: [confident, almost playful] Look, it’s simple. You just have to stop trusting the cameras.\nSpeaker 2: [skeptical, raising an eyebrow] Cameras? I’ve seen satellites, Mark.\nSpeaker 1: [waving a hand dismissively, then leaning in] Manufactured images. Have you ever felt the curve?\nSpeaker 2: [frustrated, sighing] I’ve flown in planes. I’ve crossed oceans.\nSpeaker 1: [insistent, then dropping to a whisper] That’s exactly what they want you to think. The horizon is perfectly straight, isn’t it?\nSpeaker 2: [exhausted, rubbing her temples] It looks straight because we’re on top of a sphere, not because it’s flat.\nSpeaker 1: [smiling mysteriously, slowly] And who told you that... The same people selling you the round rock."

INFOS = """**Built with Fish Audio.** Drama 3 generates 44.1 kHz mono speech from an expressive script. Short directions in square brackets set tones and emotions, sounds and pauses land where written, and paired tags such as `<whisper>` apply to selected words (see prompt help). **Speech Duration** defaults to Auto (0): it gives the whole script a generous generation budget, then each part can stop naturally at its end-of-speech token. A positive value sets a maximum. Auto and manual limits are capped at one hour. Long scripts are split automatically at sentence, line or safe text boundaries into parts of at most about two minutes; **Cut Script Every N Lines** adds an earlier dialogue boundary. Each part is prompted with references, a bounded first-part voice anchor and recent history. If a part reaches its frame or context limit before EOS, generation stops and the result includes a warning and the source span completed through EOS. EOS does not verify every word, and Auto does not remove pauses. The optional silence detector is not included in the published assets, so silence can remain in the result.

**Write** drafts a speech or a dialogue from a brief in the prompt, with its directions, using the prompt enhancer.

Choose up to three **Voice References** to clone voices for Speakers 1, 2 and 3. Supply one clean recording of one person per reference. **Reference Transcripts** holds the exact words spoken in each recording, one `Speaker N: ...` line per reference (with one reference, the words alone are enough); a transcript describes its recording, not the new speech. Leave a transcript empty and Maestro uses its optional local faster-whisper service to transcribe the recording. If faster-whisper is unavailable, enter the transcript yourself; accurate words improve the voice match. Speakers without a reference get voices chosen by the model. Use a voice reference only with the speaker's permission; see `LICENSE_FISH_AUDIO`.

Short scripts stay in one part. Longer scripts are split automatically, including a long paragraph with no line breaks. **Cut Script Every N Lines** adds whole-line boundaries for dialogue. Each part sees the voice references, the first generated part as a bounded voice anchor, and recent complete turns up to **History Kept at Each Cut**; history turns stay paired with their matching audio. Setting history to zero disables both generated-history sources. Each cut processes the prompt again, which adds some time, especially with long references or history. **Speech Duration** applies to the whole result.

**Early Stop** decodes the frames already generated; **Abort** discards the current generation. Prompt enhancement is optional and off by default. Drama 3 does not use diffusion steps or classifier-free guidance.

FP8 experts occupy about 32 GB; the INT4 release occupies about 16.6 GB and is the better choice when system RAM is limited. **Legacy** keeps routed experts on the CPU, using the native CPU kernel when available and portable PyTorch as a fallback. The optional **vLLM** engine uses accelerated layer kernels when its dependencies are available; eligible full-model VRAM profiles can keep the complete expert bank on the GPU. Profiles 1 and 3 automatically use whole-frame CUDA Graph replay only when all dense decode weights and expert banks are resident on one GPU. When the complete bank does not fit, vLLM can use a bounded packed-expert GPU cache and a graph of the audio codebook tower; CPU routing and cache transfers stay outside capture. Cache sizing leaves GPU and staging-RAM headroom. FP8 GPU expert kernels require compute capability 8.9 or newer; unsupported formats retain CPU experts. Each speech part adds preparation after prompt encoding, and unavailable capture retains eager sampling. This is not the vLLM-Omni server. The console reports the actual decoder and expert placement. Dense layers and the audio decoder use the selected model precision and Maestro's normal device placement. FP8 generation can be slow and may page heavily on a machine with less available RAM. Speeds reported for other optimized runtimes are not Maestro benchmark results.

The **INT4 Experts** model uses Fish Audio's 4-bit release of the routed experts (4-bit weights with a scale per group of 128) with the same dense layers and settings. It halves expert storage compared with FP8 and may slightly change delivery.

The codebook-MTP decoder predicts the ten codebooks of each audio frame; it is not speculative decoding of future speech frames.

Weights and upstream code are subject to the Fish Audio Community License.
"""

PROMPT_INFOS = """Write the exact words to speak in their original language, with natural punctuation, and direct the performance with short cues. Square-bracket cues are performed, not spoken.

### Tones and Emotions

- **A direction at the start of a line** sets how the line is performed: `[warm and relaxed, like talking to an old friend] It's been a while.`
- **A direction in the middle** changes the delivery from that point: `I'm so happy for you. [voice breaking] Really, I am.`
- **Sounds and pauses** land where they are written: `[sigh]`, `[sighs]`, `[laughs]`, `[pause]`.
- **Paired tags** change only the words inside them and must be closed: `<whisper>Go on without me.</whisper>`, `<stress>You</stress> were there that night.`, `<slow>Read it once more.</slow>`.
- Directions are free descriptions, in English or in the script's language: `[conteniendo las lágrimas, pero sonriendo] Gracias por venir.`

```text
[whispering, terrified] Did you hear that? [pause] Something is moving downstairs.
```

```text
[bored, monotone] Yes, the meeting is at nine. [suddenly excited] Wait, is that free pizza?
```

```text
[angry, through clenched teeth] I told you three times. [sigh] [calm and patient] Let's start again, slowly.
```

```text
[sighs] Fine. If we miss it, it's on you. [laughs] Well, maybe that's fair.
```

Short, concrete directions work best; change them as the emotion of the text changes. Directions at the start of lines are the most reliable: every cue inside a line is a point where a take occasionally stops speaking, so keep those for the moments that need them.

### Dialogue

Start each line with its speaker, from `Speaker 1:` to `Speaker 16:`, and give each line its own directions:

```text
Speaker 1: [skeptical, questioning tone] Then why is your coat still wet?
Speaker 2: [defensive, slightly uneasy] I was home all night. [pause] Asleep.
Speaker 1: <stress>Nobody</stress> saw you. [sighs] Just tell me the truth.
Speaker 2: [barely holding back tears] <slow>I didn't do it.</slow>
```

Speakers 1, 2 and 3 use the voice references in their order; the others get voices chosen by the model. A line without a label continues the previous speaker, and an unlabelled script uses Speaker 1. Drama 3's own tags `<|speaker:0|>` to `<|speaker:15|>` also work (`<|speaker:0|>` is Speaker 1). Keep each character's number consistent. Do not insert conversation-control or semantic-code tokens.

**Reference Transcripts** must contain the exact words of the voice recordings, for example:

```text
Speaker 1: Thanks for calling, how can I help you today?
Speaker 2: I'd like to change my reservation, please.
```

Temperature changes variation; top-k and top-p restrict sampling. Start with temperature 0.8, top-k 0 (no top-k cap) and top-p 0.95. This is a mitigation for reported short-speech and near-silence failures, not a guarantee; EOS and the maximum duration still apply.
"""

DEEPY_INFOS = """Built with Fish Audio. Drama 3 produces 44.1 kHz mono speech. `prompt` is the script; speaker labels 1..16 map to voices, and up to three references use `audio_guide`, `audio_guide2`, and `audio_guide3` (`audio_prompt_type` A, AB, ABD). `alt_prompt` contains the exact transcript for each reference, one `Speaker N: ...` line per reference (plain words with one reference); empty entries use Maestro's optional local faster-whisper service. Long scripts split automatically at sentence, line or safe text boundaries into chunks of at most about 120 seconds; `custom_settings.dialogue_cut_lines` N>0 also cuts at every N whole lines. Each prompt preserves references and can include the first generated chunk as a bounded voice anchor plus recent complete turns up to `dialogue_history_seconds` (default 20); turns are kept with their matching audio. `duration_seconds=0` selects Auto: its whole-job maximum is `min(3600, 8 + ceil(visible_script_units / 4))` seconds, with each part also capped by text size, 120 seconds and remaining model context. A positive value remains a whole-job maximum, also capped at 3600 seconds; EOS can end sooner. A part that reaches its frame or context limit before EOS stops the job and appears as incomplete with its completed source span and warning in `speech_generation` / `generation_warnings`. EOS does not verify every word, and Auto does not pace speech or remove pauses. The published assets do not include the optional silence detector. Start with `temperature=0.8`, `top_k=0` (no top-k cap), and `top_p=0.95`. The upstream Drama 3 service recommends uncapped candidates after reports of short speech and near-silence; this is a mitigation, not a guarantee. `prompt_enhancer` T writes a speech and T1 a dialogue. Routed FP8 experts occupy about 32 GB and INT4 experts about 16.6 GB. Legacy uses native CPU expert kernels when available, with portable PyTorch as a fallback. The optional `vllm` engine uses accelerated layer kernels and can stage the complete expert bank on an eligible GPU profile. Profiles 1 and 3 attempt part-local whole-frame CUDA Graph replay after verifying complete dense/expert GPU residency. Other vLLM profiles can use a bounded packed-expert GPU cache and a separate audio-tower sampling graph with CPU routing outside capture; insufficient GPU/staging-RAM headroom or unsupported formats retain CPU experts, and unavailable capture retains eager sampling. Capture preparation restores prompt caches and RNG before real frames. It is not the vLLM-Omni server. The console reports actual backend and expert placement; other runtimes' speed claims are not Maestro benchmarks. Early Stop returns completed frames; Abort returns no audio. The codebook-MTP decoder predicts codebooks within a frame, not future speech frames. Fish Audio Community License applies; see `LICENSE_FISH_AUDIO`."""

DEEPY_PROMPT_INFOS = """Submit the actual words to speak, preserving the user's language and exact supplied wording. Direct the performance with cues that are performed, not spoken: a free-form direction in square brackets at a line start sets that line's tone/emotion (`[warm and cheerful] Good morning!`), mid-line it changes the delivery from there (`I'm fine. [voice breaking] Really.`); sounds/pauses where they happen (`[sigh]`, `[sighs]`, `[laughs]`, `[pause]`); paired tags on chosen words only, always closed (`<whisper>...</whisper>`, `<stress>...</stress>`, `<slow>...</slow>`); directions may use the script's language. Prefer one direction at the start of each line and avoid cues inside lines (each is a point where a take can stop speaking): fold changes into the opening direction (`[calm, then suddenly angry]`) and make pauses with punctuation. Dialogue: one line per turn starting `Speaker 1:` .. `Speaker 16:` (`Speaker 1: Are you ready?\nSpeaker 2: Almost.`); unlabelled lines continue the previous speaker, an unlabelled script is Speaker 1; Speakers 1-3 take the references in order. Native `<|speaker:0|>`..`<|speaker:15|>` also work (0 = Speaker 1). Do not insert `<|im_start|>`, `<|im_end|>`, `<|voice|>`, `<|text|>`, `<|interleave|>` or semantic tokens. `alt_prompt` transcribes the references exactly (`Speaker 1: ...` / `Speaker 2: ...` lines; plain words for `A`), or stays empty for Whisper; it is not a voice-style description or the new script. Without reference audio, use empty `alt_prompt`."""

_DIRECTIONS = """Direct the performance with cues that the model performs instead of speaking:
- one short direction in square brackets at the very start of every line, setting that line's tone or emotion ([warm and cheerful], [suddenly worried], [barely holding back tears]);
- when the delivery changes within a turn, say it in that opening direction ([calm, then suddenly angry], [laughing, then whispering]) or start a new line;
- sounds, whispering, stress or slowness inside the opening direction too ([sighing], [laughing softly], [whispering], [stressing each word], [slowly]). Do not write any angle-bracket tag yourself; keep the paired tags the user wrote (<whisper>...</whisper>, <stress>...</stress>, <slow>...</slow>) exactly as they are.
Never put a square-bracket cue after the start of a line, not even [pause] or a sound: a cue inside a line is where a take most often stops speaking. Make pauses and hesitations with punctuation instead (commas, "...", dashes). Write directions in English or in the script's language, keep them short and concrete, and let the emotions change from line to line. Keep the cues, tags and speaker labels the user already wrote as they are. Use natural punctuation. Do not add structural conversation tokens (such as <|im_start|>), semantic tokens, reference transcripts, Markdown fences or commentary."""

ENHANCER = """Write only the speech script of one speaker for Fish Audio Drama 3. Preserve the user's language, exact supplied wording, meaning and explicit exclusions; if the user supplied the exact words to speak, keep them and only add directions. If asked to author speech from a brief, write the finished words to speak, 3 to 8 sentences unless another length is asked, on one line per change of tone or emotion. """ + _DIRECTIONS + """ Do not add speaker labels unless the user wrote some. Return only the script. The example shows the format only: write new content for the user's request.

Example:
[hushed, nervous] Is anyone there? The lights went out an hour ago...
[relieved, laughing softly, then whispering] Oh, it's only you. You scared me half to death.
[warm] Come in, the tea is still hot."""

DIALOGUE_ENHANCER = """Write a dialogue script for Fish Audio Drama 3 from the user prompt, with two speakers unless the user asks for more (up to 16). Preserve the user's language, exact supplied wording, characters, meaning and explicit exclusions; if the user supplied exact lines, keep them and only add directions. Write 6 to 14 lines unless another length is asked. Every line starts with its speaker label, Speaker 1: to Speaker 16:, one turn per line, each character keeping the same number. """ + _DIRECTIONS + """ Return only the dialogue lines. The example shows the format only: write new content for the user's request.

Example:
Speaker 1: [skeptical, questioning tone] Then why is your coat still wet?
Speaker 2: [defensive, slightly uneasy, hesitating] I was home all night... Asleep.
Speaker 1: [sighing, then stressing each word] Nobody saw you. Just tell me the truth.
Speaker 2: [barely holding back tears, slowly] I didn't do it."""
