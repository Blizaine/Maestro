# Maestro v2.1.4 release validation

Validated on 9 September 2026. See the [release notes](RELEASE_NOTES_V2.1.4.md).

## Release boundary

Both public branches, `dev` and `main`, pointed to
`5b475160cfce4ead863d8956139bb2db9266b62e` (v2.1.3) before this patch.
The version comes from the root `VERSION` file. The patch contains H3 VAE
loading/residency changes, fractional-profile budget corrections, regression
tests and release documentation. No launcher, dependency, model-weight, private
character or local content-guide changes are included.

## Automated and local checks

- **Full Python suite:** 1,488 tests passed in 79.526 seconds in Maestro's
  Windows Python 3.11 environment. Thirteen new tests cover native checkpoint
  storage identity, legacy/native decoder math, rotary attention, multiple
  batches, FP32/FP16, ConvRot row scales and gate order, codec phase wrappers,
  GPU-capacity thresholds, handler wiring and fractional-profile budgets.
- **Real checkpoints:** tested the FP16 and INT8 ConvRot video VAEs with MMGP
  profile 5 in isolated processes on an RTX 4090 with 128 GB system RAM.
- **Generation-stage handoff:** the production residency helper returned all
  VAE weights to CPU before a tiny stand-in transformer ran. Subsequent
  encode/decode calls also released the other codec phase. No complete
  transformer generation was performed in this probe.
- **User smoke test:** the project owner confirmed the local fixes were working
  before authorizing publication. No broader hardware coverage is implied.
- **Publication checks:** the clean-repo guard passed across 2,203 staged/tracked
  files; CI's undefined-name and Python compilation checks passed, as did all
  five standalone JSON grammar checks and staged whitespace checks.

## Loading memory

Fresh-process CPU loading of the existing Viggle INT8 transformer, FP16 video
VAE and audio VAE, followed by garbage collection:

| Metric | Previous layout | Native layout | Reduction |
| --- | ---: | ---: | ---: |
| Private committed memory | 7.411 GiB | 3.064 GiB | 4.347 GiB |
| Resident working set | 8.660 GiB | 2.399 GiB | 6.261 GiB |

The transformer was largely lazy/file-backed at this point. These figures
describe component loading, not total generation requirements. Explicit MMGP
pinning can still allocate pinned memory; this removes layout-repacking copies.

## Decoder residency

Identical real weights and seeded latent input, 51 output frames at 240 × 432,
default 256-pixel VAE tiling and MMGP profile 5. Both policies already used the
new native checkpoint layout:

| VAE | Streamed decoding | Resident decoding | Peak allocated VRAM, old → new | GPU load calls, old → new |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 8.891 s | 2.219 s | 0.711 → 4.710 GiB | 207 → 1 |
| INT8 ConvRot | 5.422 s | 1.906 s | 0.874 → 2.626 GiB | 187 → 1 |

Encoder and decoder output hashes matched exactly between residency policies
for each checkpoint. An earlier FP16 repetition measured 9.063 s versus 2.219 s.
The policy follows [WanGP's H3 pipeline](https://github.com/deepbeepmeep/Wan2GP/blob/362c3467a70e1136ceb52eec95907205a8f88543/models/minimax_h3/pipeline.py#L353).

The separate legacy-versus-native-layout comparison used real VAE checkpoints,
a 17-frame 64 × 64 encoding input and a five-latent-frame 4 × 4 decode. Encoders
matched exactly. Decoder RMS differences were 0.000767 (FP16) and 0.000830
(INT8), with at most one 8-bit output color level of difference in both samples.
Fused and separate FP16 GEMMs can round differently; bit-identical output is
established for the residency change, not promised for the layout change.

## Scope and limits

These are loading and decoder-stage measurements, not a full-resolution,
end-to-end Viggle benchmark or a run of the entire WanGP application. The
reported RTX A4500/28 GB RAM environment was not available. The measurements
identify avoidable overhead but do not prove every cause of the reported
15-minute job. Low-RAM automatic profile selection and the transformer's
workspace safeguards were retained.

Python compilation and whitespace checks passed. Undefined-name checks passed
for the changed H3 modules and new tests. Checking all of `app/wgp.py` also
reports eight pre-existing findings, reproduced against the public v2.1.3
source; this patch adds none. The existing CI check covers services, H3 and
`app/launch.py`.

Private test inputs, model files, local probe scripts, logs and generated media
remain outside the published source.
