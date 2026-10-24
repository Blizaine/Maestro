# Fish Audio Drama 3

Enable **Fish Audio Drama 3** or **INT4 Experts** in the Audio → Speech model
picker. Models download only on explicit download or first generation. Each
uses the same expressive script format:

```text
Speaker 1: [confident] Welcome to Maestro.
Speaker 2: [whispering] Did you hear that?
Speaker 1: [laughing] Let's try another take.
```

Add up to three clean voice recordings to clone the corresponding speakers.
Reference Transcripts contains the words in those recordings, not the new
dialogue. Leave a transcript blank to transcribe that recording with Whisper
on first use. Check the transcript when words are unclear.

Advanced settings offer **Cut Script Every N Lines** and **History Kept at Each
Cut**. Each part keeps recent generated speech and text as context for the
next part. New settings allow up to 60 seconds of recent complete turns;
explicitly saved values, including 0 or 20 seconds, remain unchanged. Use
temperature 0.8, top-p 0.95 and **Top-k 0 (no cap)**
as a starting point. Diffusion steps and CFG do not apply.

### Directing the performance

Use brief square-bracket directions near the words they affect, including
within a sentence. When **Write** authors or enhances a scene, it can infer
purposeful delivery changes from the story's events and emotional movement;
the user does not need to prescribe every cue. In a long passage, place
occasional directions at meaningful turns, including mid-sentence, instead of
leaving them all at paragraph openings. Ordinary lines can stay plain when
their delivery does not change. There is no tag quota, and sound effects should
not be added just to decorate the script. A sound or pause belongs where it
should occur. Paired delivery tags target a selected word or phrase:

```text
[calm] I checked <emphasis>every</emphasis> door, [my confidence slips] and none were locked.
<whisper>Stay here.</whisper> [urgent] I'll be right back.
```

The paired tags are `<whisper>`, `<emphasis>`, `<soft>`, `<fast>`, `<slow>`,
`<stress>` and `<forceful>`. Close each tag and nest overlapping scopes
correctly. Maestro preserves these scopes across automatic cuts. For deliberate
pronunciation, the local tokenizer also has `<|phoneme_start|>` and
`<|phoneme_end|>` markers; keep a complete pronunciation span on one line. See
[Fish's Drama 3 guide](https://docs.fish.audio/features/text-to-speech/drama-3)
for its language-specific phoneme notation. Experimental language-switch tags
from the hosted service are not advertised by Maestro's local guide.

Prefer a few purposeful changes over a different voice description on every
paragraph. Keep the same speaker number, cloned identity, habitual register
and accent; change emotion, emphasis or pace when the scene needs it. Instructions
such as “raspy” followed by “deep and resonant” can ask for the very delivery
change you are trying to avoid. **Write** follows this distinction while retaining
the user's explicit performance choices. Bracket directions influence delivery;
they do not lock a voice or guarantee exact pauses.

The broader [Fish emotion guide](https://docs.fish.audio/developer-guide/core-features/emotions)
describes S2/S2.1 too. Maestro uses the Drama 3 format above; hosted-service
features should not be assumed to work identically in the local engine.

### Consistency between parts

Every part uses the same supplied voice references and transcripts. With history
enabled, the prompt also includes a bounded first-part voice anchor and recent
generated text paired with its audio. History retains complete turns: a turn
longer than the configured time allowance cannot be included as recent history.
The new 60-second default accommodates more ordinary narration turns than the
previous 20-second default. Text/audio context limits can still reduce the
history actually used; output details record the effective recent-turn count,
duration and anchor for each part. More history adds prompt processing time.
Setting history to 0 disables both generated-history sources.

A fixed seed seeds the random stream once for the entire job. Maestro does not
restart that stream at every cut. Reusing the same seed on every part would not
guarantee matching register with different text and conditioning. References,
performance cues and recent audio context remain important. These changes need
listening tests on your voice and script; they do not establish seamless delivery.

### Speech length

New Fish defaults select **Auto**. Fish chooses when to finish speaking using
its own end-of-speech token; duration is never a target length or a pacing
instruction. Auto supplies a generous script-based safety budget, bounded by
the remaining model context and a one-hour total output limit, to contain
runaway silence or repetition. The console records the effective budget and
whether synthesis stopped at EOS or its limit. Reaching a limit is not proof
that the whole script was spoken.

Choose **Maximum length** to set an explicit cap, including short values such
as six seconds. Speech can end earlier. Saved positive limits remain manual
when loading old settings; choose Auto to remove that fixed choice. Auto does
not remove pauses, force the model to read skipped text, or guarantee complete
speech. Delivery tags, the reference and sampling still affect the result.

Long scripts split automatically at safe text boundaries, with bounded recent
history and codec decoding, then assemble into one file. The manual maximum
applies to the whole file. A part that reaches its generation limit stops with
an incomplete warning rather than silently skipping its remaining words.
See [speech writing and long scripts](Speech-writing.md) for personality
instructions, long story/dialogue drafting, progress and API examples.

### Speech stops after a few words

Drama 3 can generate a short spoken opening followed by near-silence, even when
the job finishes without an error. Increasing Maximum Speech Duration does not
recover words the model skipped. In **Advanced → Generation**, start with
**Top-k 0 (no cap)**; Top-p is also available there. New Fish defaults use 0,
while explicitly saved sampling values remain available to reproduce old takes.

[Fish's official demo source](https://huggingface.co/spaces/fishaudio/Drama-3/blob/c368729ab9347dda6635c5272ad31e6521114b79/server.py#L85)
documents this failure with Top-k 30 and removes that cap. Maestro now treats
0 as unlimited candidates rather than accidentally selecting only the most
likely token. This is a sampling mitigation, not proof of complete speech or
voice fidelity on every script. EOS and the overall duration bound still apply.
For a script that stops early, simplify the directions and test changes one at
a time; Cut Script Every N Lines can add earlier whole-line boundaries.
Do not treat trimmed silence as recovered dialogue.

## Resource requirements and current engine

The INT4 expert file is about 16.6 GB; FP8 experts are about 32.2 GB, in addition
to the ordinary backbone, codebook decoder, codec and working memory. Prefer
INT4 on a 32-GB RAM machine; FP8 needs substantially more free RAM. Downloads
total roughly 21–24 GB with INT4 or 37–40 GB with FP8, depending on dense precision.
INT4 can change delivery slightly because the expert weights are quantized.

The ordinary **Legacy** engine keeps routed expert storage on the CPU outside
ordinary MMGP GPU weight blocks. It uses the bundled native AVX2/FMA/F16C expert
kernel on supported Windows/Linux x64 systems, with portable PyTorch math as
a fallback. Only selected experts execute at each call. Missing native runtime
libraries select the fallback and the console identifies the actual backend.
CPU and RAM bandwidth still matter substantially.

**vLLM** appears only when the runtime passes the Triton and FlashAttention 2
checks and has both required FLA DeltaNet kernels. Experts remain on the CPU unless a full-model
VRAM profile has enough measured free VRAM to stage the complete bank while
reserving space for caches and activations. Low-memory profiles keep CPU
experts as the fallback. When the full bank does not fit, vLLM can instead
allocate a bounded cache of packed experts after prompt encoding. Each route
loads every missing selected expert before execution; current routes are
protected from eviction. Cache sizing uses measured free VRAM and retains an
activation reserve. Cache slots and pinned staging are released before waveform
decoding. `MAESTRO_FISH_EXPERT_CACHE=0` disables this cache. Upstream speed claims
are not Maestro benchmark results.

On eligible profiles **1 or 3**, Fish automatically captures the entire audio
frame transition as a CUDA Graph when the dense decoder and complete expert
banks are resident on one GPU. The graph includes the codebook tower, sampling
and next backbone step; prompt encoding and waveform decoding stay separate.
Capture runs after prefill for each speech part and adds preparation time. It
restores the real prompt caches and random state after warmup, and releases the
graph before cache release or waveform decoding. Offloaded weights, unsupported
kernels or insufficient capture memory keep the eager decoder. The console
reports capture success or the fallback reason; per-part `speech_generation`
metadata records `decode_backend` and `graph_capture_seconds`. Set the process
environment variable `MAESTRO_FISH_CUDA_GRAPH=0` to disable this Fish-local path.
No global CUDA Graph switch is required.

When whole-frame capture is unavailable, Fish also attempts a smaller graph of
the audio codebook tower and sampling. It verifies stable physical tower weights
and MMGP co-residency; CPU expert routing and GPU cache transfers remain outside
capture. This path reports `tower_cuda_graph`, and the per-part `expert_cache`
metadata records slot count, hits, misses, transferred bytes and cache VRAM.
Unavailable capture retains eager sampling. One-frame expert execution reuses
scratch, stores directly in the activation dtype, and computes routed down
projections in parallel with an ordered FP32 sum. Multi-token prompt processing
retains its existing kernel layout.

The residency check deduplicates modules and tensors by identity, including
MMGP's child-to-parent LoRA references. Earlier local graph builds could hit a
recursion error here and fall back to eager even with no LoRAs selected.

On the RTX 4090 with Torch 2.7.1/CUDA 12.8, short INT4 speech tests passed with
both engines. Legacy with native CPU experts peaked at 3.74 GiB allocated;
vLLM with the complete GPU expert bank peaked at 19.43 GiB. Both generated the
requested sentence, independently checked with transcription. These first-use
checks are not comparative speed benchmarks. Physical 12-GB generation,
three-voice cloning fidelity and long-script media quality still need testing.

The optional upstream silence classifier was not published with these assets.
Generation uses EOS, cancellation and the chosen duration limit instead.
The published model and codec licenses are preserved with their source.

### Faster consumer-GPU engines

Maestro's `vllm` Fish option uses its native model and accelerated layer kernels;
it does not run the vLLM-Omni serving engine. Whole-frame graph replay reduces
Python and kernel-launch overhead when residency is verified. Full expert
residency alone does not establish a speed claim. Earlier workstation 01 logs,
before this graph implementation, showed about 4.77 frames/s
with the complete INT4 expert bank on the 4090, roughly 4.5 seconds of decoding
per second of audio before other overhead.

[drama-rs](https://github.com/apiplant/drama-rs) reports 8.9× real time on a 4090
using custom CUDA kernels, whole-step graph replay and different Q4/Q8 weight
layouts. Its [model files](https://huggingface.co/apiplant/drama-q4) are not a
drop-in replacement for Maestro's Fish weights. These are the author's results,
not Maestro benchmarks. A compatible optimized engine needs separate
integration and output/performance verification; increasing duration or changing
MMGP placement is not equivalent.

The native Dev path removes redundant expert-ID bound checks when IDs come
directly from its own top-k router. This avoids two CUDA scalar reads per
backbone layer (eighty per decoded frame), while direct expert calls retain
their range checks. This removes synchronization overhead without changing
the required model weights or proving a complete-generation speedup by itself.

After fixing the MMGP parent-reference traversal, one matched RTX 4090 take
(INT4 experts, INT8 ConvRot dense weights, vLLM, profile 1, fixed script, voice
reference, seed and sampling) produced 131 frames / 6.084 seconds of audio:

| Timing | Eager fallback | Whole-frame graph |
| --- | ---: | ---: |
| Speech frame decoding | 26.76 s / 4.90 frames/s | 1.66 s / 79.04 frames/s |
| Graph preparation | none | 0.76 s |
| Active sidecar time | 38 s | 14 s |
| Whole job, including loading | 45 s | 21 s |

The saved audio decoded to identical samples. The decode-only improvement was
16.1× (about 3.66× real time for that stage); loading, reference transcription,
prefill and waveform decoding remain separate costs. This short single-take
comparison does not establish long-script, multi-voice or other-GPU throughput.

The 24 October follow-up added reusable single-frame expert scratch and the
smaller-memory cache/tower-graph path. Native INT4 kernel/cache checks passed on
the 4090, 3080 Ti and 5070 Laptop. The isolated expert-kernel ratios were 2.05x,
1.25x and 1.08x; these are not whole-speech speedups. A complete-model profile4
check on the 4090 used the cached path and tower graph, produced both requested
sentences and sampled 9.68 GB peak device VRAM. Physical 8GB/12GB full speech
tests remain pending model/runtime setup. The cache is sized from each GPU's
remaining memory; this 4090 check does not establish that a complete speech
job fits an 8GB card or predict its throughput.

## API

Use `/api/v1/generate` with `model_type: "fish_drama3_int4"` (or
`"fish_drama3"`), `generation_mode: "audio"`, the script in `prompt`,
`duration_seconds: 0` for Auto (or a positive maximum in seconds), and optional
`audio_guide`, `audio_guide2`, `audio_guide3`.
Set `audio_prompt_type` to `A`, `AB`, or `ABD` for one, two, or three references;
`alt_prompt` holds their transcripts. Chunk controls live in `custom_settings`
as `dialogue_cut_lines` and `dialogue_history_seconds`.
