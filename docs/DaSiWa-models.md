# DaSiWa checkpoint imports

DaSiWa models are downloaded through **Model Browser → Checkpoints** rather
than offered as eight built-in selections. Maestro keeps its checkpoint loaders,
H3 recipe verification and native workflow support. Existing imported definitions
and downloaded weights are preserved; re-importing a matching H3 file verifies
and reuses the installed weights.

## MiniMax H3

1. Open **Model Browser → Checkpoints**, filter to **MiniMax H3** and select
   the creator's DaSiWa model and version.
2. Choose a supported file and inspect its compatibility. If needed, supply the
   CivitAI API key in **Settings → Services**. The browser reports unavailable
   formats or missing workflow, recipe and QKV-layout evidence before downloading.
3. Download and install the checkpoint. After size, SHA-256 and layout checks,
   native Ref2VA/Hybrid checkpoints register paired **Frames** and **References**
   entries sharing one transformer. A Frames-only file registers only Frames.
4. Test a short 480p clip with the verified recipe before a longer project.
   Import verification does not certify creative quality or every LoRA combination.

The pinned DaSiWa Hybrid v3 standard exports start at **25 Euler steps** with
video/audio shifts **11/4**. Baked Turbo v3 starts at **8 steps**, accepts **4–8**
and uses shifts **9/4**. Supported INT8 ConvRot and W4A8 INT4 exports are covered
by the import pipeline. Other files must independently pass compatibility checks.

Baked acceleration rejects additional Turbo/PDD adapters and disables incompatible
sampling caches and audio refinement. Ordinary compatible H3 LoRAs remain
available. Dense Auto attention follows the configured backend. Compatible imports
also offer **H3 Optimizations → Sol Engine** on supported hardware; compare the
same prompt, seed and inputs because sparse attention can change the result.

**Frames** supports text, start/end/timed images and Control Video editing.
**References** supports ordered image, video and audio references. Switching
between companion workflows preserves the imported checkpoint and its recipe.

See [H3 checkpoint import](H3-checkpoint-import.md) for formats, API calls,
verification and installed-weight reuse.

## Other families

The browser's exact base-model compatibility rules still apply:

- **Krea 2:** supported files can target RAW or Turbo. Select the recipe matching
  the creator's file; CivitAI does not distinguish the two in its base-model label.
- **LTX-2.3:** compatible checkpoint structure can be imported through the LTX-2.3
  architecture. The generic import template uses Maestro's Dev recipe; it does
  not automatically reproduce the retired DragonLeap distilled preset.
- **Wan 2.2:** the browser does not yet have a verified high/low checkpoint-pair
  import pipeline. Removing the curated Lightspeed preset does not add that
  pipeline. Keep matching high/low versions when using a separately installed
  custom definition.

Maestro reports unsupported combinations instead of assigning a checkpoint to an
unrelated model family. Creator schedules and underlying model licenses continue
to apply. Shared encoders and VAEs may require additional first-use downloads.

## Creator sources

- [DaSiWa MiniMax H3](https://civitai.com/models/2877206/dasiwa-minimax-h3)
- [DaSiWa Krea 2](https://civitai.com/models/2760803/dasiwa-krea2-or-turbo-or-raw)
- [DaSiWa Wan 2.2 SafeTensors](https://civitai.com/models/1981116/dasiwa-wan-22-i2v-14b-lightspeed-synthseduction-v9)
- [DaSiWa LTX-2.3](https://civitai.com/models/2543443/dasiwa-ltx-23)

The former curated definitions remain only as test fixtures for legacy recipe
compatibility. They are not scanned by the application model registry.
