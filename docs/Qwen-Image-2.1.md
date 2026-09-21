# Qwen Image 2.1

Choose **Studio → Image → Qwen Image 2.1 7B**. It is enabled in Model Visibility
once on update; your selected image model is preserved. Disable it there if you
prefer. Model weights and the Qwen3-VL encoder download on first generation.

Start with **1024 × 1024, 40 steps, CFG 1**. Resolution controls also support 2K.
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
- **CFG:** the official recommended default is 1 (no classifier-free guidance).
  Increasing CFG above 1 enables the negative-prompt path and adds work per step.
- **LoRAs:** only adapters for the new 2.1 7B architecture are compatible. Put
  those in `app/loras/qwen21`; old Qwen 20B adapters remain in their own directory.

This first integration does not expose masked inpainting or outpainting modes;
ordinary reference-image editing is available. Existing Qwen models retain their
original controls and sampling defaults.

The model uses the **Qwen Research License** (non-commercial research/evaluation).
Commercial use requires a separate license from Qwen. Read the
[official model card and license](https://huggingface.co/Qwen/Qwen-Image-2.1)
before using it commercially.

API: submit `/api/v1/generate` with `model_type: "qwen_image_21_7B"`,
`image_mode: 1`, `prompt`, `resolution`, `num_inference_steps: 40`, and
`guidance_scale: 1`. For references also supply `video_prompt_type: "I"` and
the uploaded image paths in `image_refs`, in the intended order.
