# Pika H3 Flex in Maestro

Pika H3 Flex is an experimental native H3 adapter recipe with two model entries:
**Frames** for text-to-video and optional first/last images, and **References**
for ordered image, video, and audio references. Both generate video with
synchronized audio. Start at **480p / 5.17 seconds** (`832x480`, 124 frames at 24 fps).

The adapters are the official Comfy BF16 exports from the pinned
[Pika H3 Flex repository](https://huggingface.co/Pika-Labs/Pika-H3-Flex/tree/0ddf55ad7067ca1e76812253bd7b4600df87f759),
revision `0ddf55ad7067ca1e76812253bd7b4600df87f759`. Maestro checks each exact
file size and SHA-256 before loading the managed adapter. Frames and References
use separate adapter files and pins.

The initial experimental stack uses the approved stock Comfy scaled-FP8 H3
transformer, the pinned BF16 Pika adapter, and the native INT8 ConvRot video VAE.
The definitions expose only that FP8 checkpoint; BF16 and INT8 base aliases need
a separately verified adapter conversion before they are qualified.

## Fixed recipe

Pika Flex uses six model evaluations at knot values `1000, 750, 500, 250, 100,
40, 0`, with video/audio shifts `12/3`, CFG `1`, and adapter strength `1`. The H3
wrapper advertises its sampler as Euler. Flex transitions use their own fresh
noise recipe between nonterminal evaluations; this is not an ordinary six-step
H3 Euler schedule.

Steps, guidance, adapter strength, and shifts are locked. Extra LoRAs,
accelerators, control video, timed frame injection, First Block
Cache, and Audio Refinement are unavailable. References are ordered; changing
their order changes the request meaning.

### Attention choices

Experimental SLA is the default on supported GPUs and safely falls back to the
runtime's supported dense attention when unavailable. Sage2 and SDPA remain
explicit dense choices; SOL remains optional experimental sparse attention.
These choices leave the six evaluations, sampling knots, shifts, and managed
adapter unchanged. Sparse attention can change motion, reference fidelity and
audio, so compare outputs before relying on it for a project. Hardware/runtime
availability and fallback status are reported beside each option.

First Block Cache remains disabled for this freshly re-noised six-step recipe.
A single native pass retains its selected output duration; older saved window
lengths must not shorten it. Longer timelines use the experimental continuation
workflow described below, with the same six evaluations in every pass.

## Inputs and output sizes

Frames supports text-to-video and optional first and last images. References
accepts up to 9 images, 3 videos, and 3 audio clips, with 12 items total; at
least one image or video is required. Audio references cannot be used alone.
Video/audio references are limited to 2–15 seconds per item, with at most 15
seconds combined for each media type.

The native video VAE requires `17*n + 5` frames at 24 fps. Maestro defaults to
124 frames (about 5.17 seconds) and permits 124–362 frames (up to about 15.1
seconds) per native pass.

## Experimental sliding windows

Frames and References can use Maestro's shared H3 continuation workflow for
longer timelines. Set the total Duration and the per-pass Window Length using
the same controls as other H3 models. Each pass keeps the fixed Pika sampler;
only the amount of new timeline generated per pass changes.

Continuation carries multiple recent video frames, a boundary frame and the
matching generated stereo audio into the next pass. References retains the
original ordered reference manifest as well. This preserves motion context
instead of starting each pass with an isolated still image. Experimental Pika
References sequences keep motion/audio continuity enabled; the independent
hard-cut sequence option is not exposed for this adapter.

The overlap follows H3's `17*n+1` grid. For example, two 124-frame passes with
18-frame overlap produce 230 joined frames (about 9.58 seconds). Native passes
are aligned separately; the joined total stays on the requested integer-frame
timeline and only its final excess tail is trimmed. Saved sequence settings
retain their total, window length and overlap when loaded again.

Start with short windows, especially on smaller GPUs. Longer total durations
increase the number of passes rather than requiring all frames in one pass.
Memory advice for this experimental adapter is conservative and must not be
read as a measured Pika performance guarantee. Motion, voices, music and object
positions can still drift across windows; inspect joins for the actual project.

The resolution menus use the two native tiers and fixed aspect-ratio buckets
from the pinned [Pika H3 Flex repository](https://huggingface.co/Pika-Labs/Pika-H3-Flex/tree/0ddf55ad7067ca1e76812253bd7b4600df87f759):

| Tier | Width × height buckets |
| --- | --- |
| 480p | `960x416`, `832x480`, `704x480`, `640x480`, `608x480`, `480x480`, `480x608`, `480x640`, `480x704`, `480x832`, `416x960` |
| 768p | `1536x672`, `1344x768`, `1152x768`, `1024x768`, `960x768`, `768x768`, `768x960`, `768x1024`, `768x1152`, `768x1344`, `672x1536` |

## Local limits and claims

Maestro does not include Pika's hosted 1080p/2K/4K upsampler, separate
soundtrack stems, prompt enhancement, or private serving API. A four-step mode
is not exposed here. The FP8 base and INT8 ConvRot VAE can produce different
numerical results from a Pika run using unquantized weights.

Maestro has not verified real-time generation on a consumer GPU, so this entry
makes no consumer-GPU real-time claim. Begin at 480p and 124 frames, then measure
the full generation and decode path on the target hardware.

Styles remain user-directed. The adapter does not force a realistic look;
describe the visual style and shot language you want in the prompt.

## License and attribution

Maestro is distributed under the WanGP Non-Commercial Evaluation License 1.1;
see the [license notice](../../LICENSE) and [full license text](../../app/LICENSE.txt).
Pika's upstream inference code is Apache-2.0 under its own terms; that does not
relicense Maestro. MiniMax H3 base weights use the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE).
Preserve Pika and MiniMax attribution and review the pinned repository's terms
for its adapter weights before redistribution. See [the adapter notice](../../app/models/minimax_h3/licenses/pika_flex/NOTICE.txt)
for provenance.
