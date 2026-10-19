# v3.0 workflow troubleshooting

## H3 References dialogue compatibility

The v3.0 speaker-attribution checks could reject older prompts that use
`(S1) says: <d>[French] C'est super, j'adore.</d>`, even when that speaker was
declared in `subject_definitions` or established by an earlier attributed line.
Generation and prompt enhancement now share the same compatibility resolver.

An explicit declaration such as `<Subject 2> (S1) is Sam from <Picture 2>` lets
later `(S1)` dialogue keep Sam's identity. An earlier `Sam (S1) says ...` also
establishes that binding for subsequent lines in the same clip. The selected
voice follows the character's actual reference association. Subject 2 can be
Speaker 1 and use Audio 2; those numbers describe separate things.

`[Shot 2]`, scene labels, timestamps and other production headings cannot become
guest character names. Bindings are local to each prompt/clip. Contradictory
declarations or an otherwise unidentified speaker among multiple characters
still require an explicit name or Subject tag beside the line.

For new prompts, keep ownership explicit:

```text
<Subject 2> (S1) says in the voice referenced from <Audio 2>,
<d>[French] C'est super, j'adore.</d>
```

Use the Subject and Audio labels that belong to the intended character in your
actual references. Existing unambiguous legacy prompts do not need manual edits.

## Viggle Animate and Refine faces downloads

Older releases pinned model assets to Hugging Face revisions that the upstream
repository subsequently removed. This can produce a missing-revision or
file-not-found error on the first download. The corrected workflows use verified
revision `630975da14fd3c84169d6cd5092da3ce1287d137` of
[DeepBeepMeep/MiniMax-H3](https://huggingface.co/DeepBeepMeep/MiniMax-H3/tree/630975da14fd3c84169d6cd5092da3ce1287d137).
Existing downloaded files are reused; deleting your model cache is unnecessary.

Face Refiner also handles H3's newer 8-bit video output directly. Applying the
older floating-point conversion to those pixels twice washed out refined faces.
The conversion now distinguishes 8-bit pixels from signed floating-point output.
This affects both Pruned and Fused refinement.

## FLUX 3 Klein LoRAs

Both Distilled and Base expose **LoRAs & presets** in Advanced Settings. Use a
LoRA trained for the matching FLUX 3 architecture, then select it and set its
multiplier through the existing picker. Per-step multiplier schedules are
supported. FLUX 2 LoRAs are not interchangeable with FLUX 3 LoRAs.

Supported native and Diffusers target names are converted for Maestro's native
FLUX 3 runtime. Unsupported targets still undergo compatibility validation.
Successful loading does not guarantee that every third-party adapter format or
training recipe is compatible.

## Recast runtime and RAM

Fast Recast starts at 480p, eight steps and CFG 1. HQ uses a separate 40-step
recipe; manually loaded settings can retain different values. Check the actual
settings when investigating a very long estimate.

On a 32 GB RAM system, the BF16 14B transformer together with an unquantized
text encoder can cause substantial host-memory pressure. Use the supported
INT8 variants when RAM is limited. Recast now releases full-resolution tracking
arrays after saving the masks, references and optional shot files, before video
generation begins.

The console includes a **[SCAIL-2 Performance]** line with selected checkpoint
basenames, steps, CFG, resolution, frame/window lengths, planned placement and
host-memory measurements. The same summary is saved in the job's
`performance_plan.scail2_diagnostic` field. A configured budget of zero means
the MMGP profile default; it does not mean the transformer uses no VRAM.

For a performance report, include the startup/job log and saved settings. Separate
preprocessing, loading and denoising time. The reported 20-hour RTX 4080 run had
no logs or saved settings available, so its precise cause and any speedup from
these changes remain unverified.
