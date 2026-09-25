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
memory budget and recomputes conditioning when caching would exceed it. VAE
tiling now follows GPU capacity with wider overlap, replacing the former fixed
384-pixel tiles. This addresses a plausible source of seams reported in
[#153](https://github.com/Blizaine/Maestro/issues/153), but the report supplied no
sample or settings, so visual confirmation is still needed. Speed and memory
depend on resolution, references, precision and hardware.

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
`custom_settings: {"qwen21_kv_cache": "Enabled"}`; omit it to use the low-memory
default.
