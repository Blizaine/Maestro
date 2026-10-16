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
next part. Maximum Speech Duration bounds the whole result; shorter scripts
can finish earlier. Use temperature 0.8, top-p 0.95 and top-k 30 as a starting
point. Diffusion steps and CFG do not apply.

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
checks and has both required FLA DeltaNet kernels. This port uses graph-free
accelerated layer execution. Experts remain on the CPU unless a full-model
VRAM profile has enough measured free VRAM to stage the complete bank while
reserving space for caches and activations. Low-memory profiles keep CPU
experts. CUDA Graphs and partial expert GPU residency are not enabled. Upstream
speed claims are not Maestro benchmark results.

On the RTX 4090 with Torch 2.7.1/CUDA 12.8, short INT4 speech tests passed with
both engines. Legacy with native CPU experts peaked at 3.74 GiB allocated;
vLLM with the complete GPU expert bank peaked at 19.43 GiB. Both generated the
requested sentence, independently checked with transcription. These first-use
checks are not comparative speed benchmarks. Physical 12-GB generation,
three-voice cloning fidelity and long-script media quality still need testing.

The optional upstream silence classifier was not published with these assets.
Generation uses EOS, cancellation and the chosen duration limit instead.
The published model and codec licenses are preserved with their source.

## API

Use `/api/v1/generate` with `model_type: "fish_drama3_int4"` (or
`"fish_drama3"`), `generation_mode: "audio"`, the script in `prompt`,
`duration_seconds`, and optional `audio_guide`, `audio_guide2`, `audio_guide3`.
Set `audio_prompt_type` to `A`, `AB`, or `ABD` for one, two, or three references;
`alt_prompt` holds their transcripts. Chunk controls live in `custom_settings`
as `dialogue_cut_lines` and `dialogue_history_seconds`.
