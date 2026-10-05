# MiniMax H3 checkpoint imports

Open **Model Browser → Checkpoints**, filter to **MiniMax H3**, and open a model. Choose the creator's version and file. Maestro checks the selected file before enabling its **Import** button, then verifies the complete download before registering the model.

Check the file selection: one CivitAI version can contain several formats, or even VAE components. The selected file's actual tensor index determines compatibility. Imported model names include their quantization so different files from the same release are distinguishable.

## Supported files and workflows

The importer supports individual joint video/audio H3 transformer checkpoints that match Maestro's loader:

- Pruned rank-8 20B or full 33B layouts.
- BF16/FP16 weights with proven QKV row order.
- Comfy INT8 ConvRot with consistent group-64 or group-256 descriptors. All 200 transformer linear layers must be described, either by per-layer `.comfy_quant` tensors or by the file header's `_quantization_metadata`. Header-based exports do not need duplicate descriptor tensors; if both representations are present, they must agree.
- Scaled E4M3 FP8 with supported scalar/column scales and proven grouped QKV row order. Published Comfy-Org FP8 file checksums are recognized, including their CivitAI copies.
- DaSiWa/WanGP asymmetric **W4A8 INT4** SafeTensors (`asym_w4a8_int8`): packed four-bit weights, FP8 relative scales, FP32 channel scales, weight groups of 16, and ConvRot groups of 64 or 256. Codebook and correction tensors are preserved. Maestro uses chunked Triton decoding and INT8 multiplication on compatible CUDA devices, with a portable fallback.
- **GGUF v2/v3** with Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0 linear weights. Mixed supported types and retained floating-point weights are allowed. Norms, biases and the AdaLN curve must remain floating point. Packed QKV and MLP weights remain fused during loading.

The rotary-position frequency buffer can be stored in FP32, BF16 or FP16. Maestro calculates rotary positions in FP32. Integer or quantized rotary buffers and incorrect shapes remain unsupported.

Native **FL2VA** checkpoints receive a **Frames** entry. Native **Ref2VA/hybrid** checkpoints receive **Frames and References** entries sharing one transformer file. Both use the checkpoint's native Ref2VA LoRA basis, including when the UI is in Frames mode. Full and pruned files retain their actual architectures.

Entries are labeled with the creator's model and version and appear in the Studio selectors after import. Version and file IDs distinguish standard/Turbo releases and quantizations. Switching between imported companion workflows retains the same checkpoint edition. Import completion refreshes the catalog without replacing a current prompt, selected model or generation settings.

Other INT4 encodings (including Nunchaku/NF4), INT6, NVFP4, unsupported GGUF types such as IQ/MXFP, archives, workflow JSON, VAEs, standalone LoRAs, VDN/custom attention exports and unknown layouts remain blocked. An INT4 label alone does not prove W4A8 compatibility. Files without a published SHA-256 remain blocked until that verification data is available.

Some GGUF exports omit QKV row-order metadata. The browser asks for the creator's **grouped Q/K/V** or **head-interleaved** layout before importing; an unknown layout is never silently assigned. Follow the creator's exporter instructions. Maestro supports both orders in its fused attention paths. Already-fused native LoRAs must use the checkpoint's logical row order; independently named Diffusers Q/K/V adapters are converted to the loaded layout.

## Sampling recipes

Maestro detects documented native workflows and standard, baked Turbo or fused sampling recipes. When that information is ambiguous or incomplete, the browser asks you to select the creator's workflow/recipe explicitly. The resulting verification profile records the choices and defaults. A selection cannot override a conflicting concrete file identity.

When a model page offers both Turbo and non-Turbo files, its shared acceleration instructions do not identify every file. Maestro asks for the selected file's recipe unless its header or version name proves one, and does not borrow step counts or shifts from a sibling edition.

DaSiWa Hybrid v3 standard INT8 and W4A8 INT4 use the documented 25-step Euler recipe, video shift 11 and audio shift 4. Their baked Turbo v3 versions use 8 steps by default, accept 4–8, and use video shift 9 and audio shift 4. These releases are identified by their exact CivitAI model/version/file IDs and published checksums.

Baked acceleration disables managed Turbo, extra Turbo/PDD adapters, First Block Cache and audio refinement. Compatible imports allow optional Sol Engine on supported hardware. Its stored sampler, shifts, CFG and supported step range are enforced at generation time. Ordinary compatible H3 character/style LoRAs remain available with the existing adapter checks. Creator-published schedules take precedence; an explicitly confirmed recipe uses Maestro's documented fallback defaults for missing fields.

“Verified” means the selected file's structure, quantization, workflow identity, sampling profile and downloaded checksum passed the import checks. It does not certify the creator's output quality or every combination of LoRAs and conditioning inputs. Test a short clip before a long production run.

## Downloads and reuse

Preflight reads a bounded SafeTensor header or GGUF tensor index (at most 8 MiB), the small AdaLN curve, and any quantization markers. It does not load transformer weights onto the GPU. Nearby markers share a read; distant marker reads have bounded concurrency. Leading GGUF curves are read from the same bounded response, including when a server ignores Range.

The download goes to a temporary file. Maestro checks its expected size, SHA-256 and H3 layout before publishing it, then writes provenance and workflow definitions. A failed check preserves an existing checkpoint.

If an identical checkpoint is already installed, including a previously downloaded DaSiWa file with another local filename, Maestro verifies its full SHA-256 and reuses it. Frames and References do not download duplicate transformer weights. The usual H3 text encoder and VAEs remain shared and download on first use if needed. An imported checkpoint does not silently fall back to stock H3 weights.

## API

Use the URL reported by the running Maestro launcher as `base`. The CivitAI key stays in Maestro's service settings; clients do not need to put it in request payloads.

`POST /api/v1/civitai/checkpoint-inspect` accepts `model_id`, `version_id`, `file_id`, and optional `h3_sampling_profile`, `h3_native_workflow`, and `h3_qkv_layout` (`auto` by default; explicit layouts are `grouped` or `interleaved`). It returns `supported`, `status`, an explanation or verification `profile`, and compatible `architectures`. `needs_selection` identifies choices that must be confirmed before importing. Send the same selections with the download request; they are retained for local revalidation and restart.

JavaScript:

```javascript
const selection = {model_id: 2877206, version_id: 3374439, file_id: 3263048};
const result = await fetch(`${base}/api/v1/civitai/checkpoint-inspect`, {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(selection),
}).then(response => response.json());
```

Python:

```python
selection = {"model_id": 2877206, "version_id": 3374439, "file_id": 3263048}
result = requests.post(f"{base}/api/v1/civitai/checkpoint-inspect",
                       json=selection, timeout=180).json()
```

Curl:

```sh
curl "$MAESTRO_URL/api/v1/civitai/checkpoint-inspect" \
  -H 'Content-Type: application/json' \
  --data '{"model_id":2877206,"version_id":3374439,"file_id":3263048}'
```

After a supported response, `POST /api/v1/civitai/download` with the selection plus `kind: "checkpoint"`, `base_model: "MiniMax H3"`, and a returned `target_architecture`. The server resolves the official file URL and metadata itself. Poll `GET /api/v1/civitai/downloads`; completion includes `model_type`, companion `model_types`, and a reuse/import message. `POST /api/v1/models/reload` refreshes the server registry; the browser does this automatically when an import completes.

## Validation

Baked Turbo imports retain their verified sampler, step bounds and shifts
while allowing the normal dense Auto backend or an explicit SDPA choice.
**H3 Optimizations → Sol Engine** is an optional sparse-attention speed setting
on supported hardware; it can change the result. Unsupported Sol requests
fall back to an available dense backend. This does not add another Turbo
adapter, enable First Block Cache, or switch the imported sampling recipe to
the bundled fused model's SLA recipe.

Focused CPU tests cover architecture anchors, quantization descriptors, finite AdaLN curves, wrapped names, full/pruned detection, source/file matching, authentication errors, bounded byte reads, checksum reuse, atomic publication, native workflow registration, loader revalidation and baked acceleration guards. Isolated tests of the actual UI store cover arbitrary imported companion IDs, step memory, model visibility and preservation of current generation state.

Real CivitAI preflight was exercised for DaSiWa standard/Turbo INT8, DaSiWa Turbo v3 W4A8 INT4, native FL2VA INT8, both published Comfy-Org pruned FP8 workflows, and community Q4_0 GGUF checkpoints. Public Unsloth Q2_K FL2VA and Q4_K Ref2VA indexes were also checked. Unsupported or mislabeled quantizations and missing checksums are rejected. These checks do not substitute for rendering a newly imported creator checkpoint.

Eros Max beta5 standard/Turbo and beta3 Turbo INT8 headers also passed bounded inspection with their matching recipe selections. Header-based INT8 loading is tested through MMGP using tiny real SafeTensor files on the CPU, checking group-64/group-256 rotation, QKV splits and row scales against an independent dense reference.

The existing live INT8 import was verified to reuse installed weights and register both workflows. New W4A8 CUDA tests compare real Triton decoding and chunked INT8 multiplication against an independent CPU reference, including FP32/BF16, both ConvRot groups, codebooks and corrections. Tiny real GGUF files exercise the registered MMGP loader, all ten supported quantizations against the official GGUF dequantizer, CUDA linear execution, BF16 activations, both fused H3 attention layouts and CPU/CUDA tensor offloading. These imports retain their original GGUF container; conversion to MMGP's serialized Quanto SafeTensor format is outside this pipeline. GGUF parser tests cover nonseekable streams, malformed metadata, shapes, types, offsets, overlap, truncation and bounded reads. Workflow/receipt tests cover layout selection, restart revalidation, atomic publication and preservation of existing SafeTensor imports. The backend regressions, UI regression suites, production build and changed-file lint remain part of verification. Full new-format generation quality should be checked with a short creator checkpoint render.
