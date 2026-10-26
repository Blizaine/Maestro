# Qwen Image 2.1

Choose **Studio → Image → Qwen Image 2.1 7B**. It is enabled in Model Visibility
once on update; your selected image model is preserved. Disable it there if you
prefer. Model weights and the Qwen3-VL encoder download on first generation.

Start with **1024 × 1024, 40 steps, CFG 4**, matching the Wan2GP base recipe.
Existing saved settings are preserved; **Advanced → Image acceleration → Apply
recommended steps and guidance** resets the recipe when needed.
Larger images and more references use more GPU memory. Maestro uses MMGP
offloading and tiled VAE encoding/decoding with BF16 or FP32 arithmetic, keeping
the new architecture separate from the older Qwen Image/Edit 20B models.

- **Generate:** describe the finished picture, including composition, materials,
  lighting and any literal text. Enhance uses a dedicated 2.1 guide.
- **Edit/combine:** attach up to ten reference images and describe the change.
  Use `<image1>`, `<image2>`, etc. in their displayed order and specify what to
  preserve. The same model handles generation without references.
- **Transparent images:** request “an RGBA image with transparency,” with
  “an alpha channel and a transparent background.” Maestro preserves the alpha
  channel and saves PNG output automatically.
- **CFG:** the base recipe uses 4. Values above 1 enable the negative-prompt path
  and add work per step; Turbo profiles use 1.
- **LoRAs:** only adapters for the new 2.1 7B architecture are compatible. Put
  those in `app/loras/qwen21`; old Qwen 20B adapters remain in their own directory.

**Official Qwen Image 2.1 Turbo 7B:** choose this separate image model for
Qwen's accelerated checkpoint. It downloads through Maestro normally, using
BF16 or INT8 ConvRot according to your precision setting. It reuses the base
model's processor, Qwen3-VL encoder and VAE; those components are not duplicated.
The checkpoint uses **8 steps and CFG 1**, with Qwen's exact published sigma
schedule and dynamic shifting disabled. Maestro enforces that recipe on queued,
restored and API jobs. Viggle acceleration profiles remain available on the base
model only. Ordinary compatible 2.1 LoRAs are preserved, but their quality with
this distilled checkpoint depends on the adapter. The shared architecture exposes
the same reference, control, inpaint, outpaint, transparency and 2K tools.
Start at 1024px; higher resolutions and references need more memory.

This option is enabled once on update without replacing your current model;
you can hide it in the model picker. Weights download only when requested.
Use `model_type: "qwen_image_21_7B_turbo"`, `sample_solver: "qwen21_turbo"`,
`num_inference_steps: 8` and `guidance_scale: 1` through the generation API.
See the [official Turbo model card](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo).

**Control images:** under Advanced, choose a control-image process and supply an
image for pose, depth, edges, grayscale or raw-image transfer. Reference images
can accompany the control image. The source/control image occupies the first
conditioning slot, so at most nine additional references fit the ten-image limit.

**Inpaint / Outpaint:** choose the corresponding Image workflow. Inpaint takes a
source and black-and-white mask (white changes, black stays protected). Outpaint
takes a source and canvas-expansion amounts. Additional references can guide
identity or appearance. For Inpaint, Advanced offers Masked Denoising with edit
strength or LanPaint at 2, 5, 10 or 15 inner steps; more inner steps take longer.
Outpaint uses the model's canvas-extension path.

**Native 2K:** choose **2K** in Resolution. Square is 2048 × 2048, widescreen is
2752 × 1536, portrait reverses those dimensions, and 4:3 is 2400 × 1792. These
use roughly 4.2 megapixels; “2K” is not a 2048-pixel longest-side limit. Auto
aspect matches the source within that pixel budget. This is native generation,
not a post-generation upscale. See the [official aspect presets](https://huggingface.co/Qwen/Qwen-Image-2.1#supported-aspect-ratios).

**Viggle Turbo:** choose a profile under **Advanced → Image acceleration**.
v0.1 uses 4 steps, v0.2 uses 5, and v0.2.1 uses 6. The matching adapter downloads
on first generation and runs at strength 1 with CFG 1 and its fixed sigma
schedule. Returning to Base removes the managed Turbo adapter while retaining
your ordinary style LoRAs. Complex edits can favor the base model; Turbo's
authors have limited validation for masks, RGBA, many references and 2K output.
See the [Viggle model card](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo).

**Memory:** reference KV caching is opt-in in Advanced. Maestro still checks its
memory budget and recomputes conditioning when caching would exceed it.
Numbered denoising steps and final VAE decoding have separate working sets;
low VRAM use while streaming transformer weights does not establish that a
large decoder tile will fit.

The image decoder omits unused temporal feature caches, moves finished tiles
to CPU, and blends their reliable interiors in FP32 before quantizing once.
Tiles are full-sized against image edges, avoiding narrow last-row/column
tiles. Full selects up to 1024px tiles; Medium and Aggressive retain at least
512px of decoder context with 256px overlap, so the lower-memory setting no
longer trades that context for 256px tiles. Images within one tile decode in
one pass. Larger overlap can add decode work; it does not repeat denoising.
These changes target the thin tile-boundary lines reported with Qwen 2.1;
they do not remove lines already present in a reference image.

CPU regressions cover exact single-image cache parity, RGBA/batched output,
edge coverage and simulated padding seams. A decoder-only comparison using
the same encoded 1088×1920 portrait latent reduced Full's peak allocated VRAM
on a 4090 from 6.85GiB to 3.26GiB and reduced its worst 64px-window difference
from an untiled reference. This is a decoder measurement, not a claim that
every resolution, reference count or complete render fits an 8GB GPU.
An isolated decoder check on the 8GB RTX 5070 Laptop GPU with the VMM
allocator also completed Full at that size: 3.26GiB peak allocated,
4.22GiB reserved, and 6.4 seconds. This check used a private candidate copy;
it did not replace or restart the installed app or rerun denoising.

The model uses the **Qwen Research License** (non-commercial research/evaluation).
Commercial use requires a separate license from Qwen. Read the
[official model card and license](https://huggingface.co/Qwen/Qwen-Image-2.1)
before using it commercially.

API: submit `/api/v1/generate` with `model_type: "qwen_image_21_7B"`,
`image_mode: 1`, `prompt`, `resolution`, `num_inference_steps: 40`, and
`guidance_scale: 4`. For references also supply `video_prompt_type: "I"` and
the uploaded image paths in `image_refs`, in the intended order.

Control transfers use `image_guide` and `video_prompt_type` (`PV` pose, `DV`
depth, `EV` Canny edges, `SV` shapes, `CV` grayscale, `V` raw), plus `I` when
references are present.
Inpainting uses `image_mode: 2`, `image_guide`, `image_mask`, `video_prompt_type:
"VAG"` (or `VAGI` with references), and `model_mode: 0` for masked denoising or
2/3/4/5 for LanPaint. Turbo uses `sample_solver: "viggle_v01"`, `"viggle_v02"`,
or `"viggle_v021"`. Cache control is
`custom_settings: {"qwen21_kv_cache": "Enabled"}` for 16-bit storage, `"INT8"`
for quantized VRAM storage, or `"INT8_RAM"` for quantized CPU-RAM storage.
Omit it or select `"Disabled"` to keep the low-memory default. Caches are
request-local and disabled when LoRA weights change across steps. Both CFG
branches have separate caches. Memory guards may recompute the full prefix
instead; they retain every reference and the requested output dimensions.

The INT8 layout uses 32-channel groups and FP32 scales. With 128-channel heads,
storage is 56.25% of the 16-bit K/V cache, with a small precision tradeoff.
This does not halve total model or job memory. The portable path materializes
one cached layer before the existing attention call, and RAM mode adds CPU/GPU
transfers. The native staged Sage cache kernels are not included. GPU speed and
visual fidelity need matched local testing; caching can be slower for a short
prefix, especially from RAM.
