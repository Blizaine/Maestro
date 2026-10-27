# Hy-Nomas world runtime

## Scope

Worlds is experimental and hidden by default. Enable **Settings → Integrations → Beta Features → Worlds mode (Experimental)** to show **Worlds** below **Blend** in Studio's **Video → Create** workflow menu. This preference is saved per installation. Disabling it closes the drawer without cancelling queued work or removing saved worlds.

This port adds Hy-Nomas, a bounded `.wgpworld` v1 reader/writer, and a dedicated Worlds drawer connected to Maestro's durable generation queue. Create a world, continue its events, or request discrete camera moves; save its state, capture its last frame, and record the saved clip chain. Each move is a queued job. Interactive latency and model quality still require GPU validation; this integration does not promise real-time navigation.

The model runtime returns a `.wgpworld` payload from generation and retains the complete session state in `model_factory.session`. It preserves the latents, segment text embeddings, camera poses, manifest timeline, and the rebuilt/active history archive. The `.wgpworld` file intentionally does not contain KV history; loading it replays each stored segment's history write.

The application connects the durable queue, immutable sidecar storage, ownership-checked cancellation, and license acknowledgement. Worlds stays outside Studio's ordinary model selectors. The handler advertises `world_model=true` and no latent-preview support because this branch does not contain the correct DCAE-V preview factors. Original Studio, Director, and Editor choices remain independent of the drawer.

## Session integration contract

The model runtime exposes these methods for the app adapter:

| Method | Contract |
| --- | --- |
| `generate(...)` | Runs a new world or a continuation when `input_custom` is a saved world path. Returns `{"x": video, "side_files": {".wgpworld": bytes}}`; the `world_video` mode also returns `overridden_inputs.video_length` equal to the actual decoded frame count. |
| `open_world(path)` / `reload_world(path)` | Validates the file, loads its tensors, and reconstructs segment history before returning a `WorldSession`. |
| `continue_world(world_path, input_prompt, **kwargs)` | Calls the normal segmented generation path with the saved world as its base. |
| `move(keys, text, seconds, on_step, steps, memory)` | Extends the in-memory session and returns RGB pixels as `[3, frames, height, width]` in `[-1, 1]`. It stores camera poses and segment metadata; the last history write is completed by `finish_move()`. |
| `save_world()` | Completes any pending move history write and returns new, validated `.wgpworld` bytes, including all moves so far. |
| `recording_output()` | Returns `{"frames": uint8[frames, height, width, 3], "fps": 24, "pixel_format": "rgb24"}` for the app's video encoder. Frames include the world's video and recorded moves. |
| `close_session()` | Releases the current working-set tensors and session references. |

The model also supplies a `WorldExplorationAdapter` with a `.world_path`, `next_request(timeout)`, and `publish(event, **payload)` for model-local tests and other internal adapters. The user-facing first slice instead submits every create/continue/move through the durable generation queue. It does not keep a long-lived hidden GPU owner between actions.

The application still needs to route generation results and saved bytes into its media store, open/reload a session before movement, stream `move()` output, call `finish_move()` after playback, map `recording_output()` to the normal video encoder, and close sessions on exit/cancellation. Resolve media IDs to local paths in the app layer before calling `open_world`; the model only accepts a filesystem path or bytes.

## `.wgpworld` v1 format

The container is an uncompressed ZIP with exactly these entries:

- `manifest.json`: `format`, `version`, model geometry and frame rate, generation list, ordered segment timeline, world description, history budget, and a session ID.
- `tensors.safetensors`: `latents` as BF16 `[1, 32, latent_frames, height/32, width/32]`; one `text_<index>` BF16 `[tokens, 3584]` tensor per segment; and optional finite FP32 `camera` poses `[4 * latent_frames, 4, 4]`.
- `thumbnail.jpg`: the gallery-sized still.
- `poster.jpg`: the full-size still.

The reader rejects unknown or duplicate ZIP entries, compressed/encrypted entries, path traversal names, symlinks, oversized files, malformed JSON, unsupported versions, invalid tensor headers, wrong tensor names/shapes/dtypes, non-contiguous latent timelines, inconsistent generation counts, invalid camera matrices, and invalid JPEGs. Bounds are 3 GiB total, 2 GiB tensors, 8 MiB manifest, 16 MiB safetensors header, 8 MiB thumbnail, 32 MiB poster, 8192 pixels per image edge, and 10,000 latent frames. The file is fully validated before the model replays history.

Early v1 files can omit `history_frames` and `world_id`; the reader supplies the default 18-frame history and a new session ID. The v1 structural fields and segment tensors remain required.

## Runtime compatibility

The current Maestro branch ships MMGP 4.0. It does not expose the newer `mmgp.runtime.state_ring`, `PrefixCache`, or `joint_qkv_attention` helpers. The model therefore keeps its bounded Q8 history in CPU RAM, transfers one block at a time for attention through the model-local adapter, and uses Maestro's `shared.attention.pay_attention`. Q8 is symmetric with 32-channel groups and FP16 scales.

This branch also has no `shared.qtypes.block_scaled`, `shared.kernels.fp8_backend`, or `int8_backend.wgp_group_linear`. The local [`block_scaled.py`](../app/models/hy_nomas/block_scaled.py) adapter recognizes only the pinned DiT and decoder metadata formats and preserves their INT8/FP8 block scales. Its correctness fallback dequantizes one requested linear weight in bounded chunks for PyTorch linear; it is expected to be much slower than the upstream dedicated kernels and has not been measured on a GPU. No speed, memory, or visual-quality claims are made. The adapter is intentionally model-local and does not modify shared quantization or kernel code.

The runtime's cancellation checks run between DiT blocks, decoder layers, and decoder windows. Decoder fused Triton dispatch remains disabled until the newer dispatch API is integrated and hardware-tested. No model weights are downloaded by these tests.

## Licenses and model assets

The Hy-Nomas inference code, DiT checkpoint, and LoRAs are Apache-2.0 under Tencent's notice. The complete license and the source distribution's bundled dependency-license text are retained in [`LICENSE-Tencent-Apache-2.0.txt`](../app/models/hy_nomas/LICENSE-Tencent-Apache-2.0.txt). Its source was fetched from Tencent Hy-Nomas revision `18974d41ada7af4aa3a026480197ab0ac0702682` (SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`). The associated source notice is included in [`NOTICE`](../app/models/hy_nomas/NOTICE). See the [official Hy-Nomas repository](https://github.com/Tencent-Hunyuan/Hy-Nomas).

The PixelUMM decoder code is Apache-2.0; its full code license from nv-tlabs/PixelUMM revision `e19e91aabb99784f7b5524268ed42ec59c1ad660` is retained in [`LICENSE-PixelUMM-Apache-2.0.txt`](../app/models/hy_nomas/LICENSE-PixelUMM-Apache-2.0.txt) (SHA-256 `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`). The Hy-Nomas fine-tuned decoder weights and derivatives use NVIDIA's [One-Way Noncommercial License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf); the exact 2022 PDF is retained in [`LICENSE-PixelUMM-One-Way-Noncommercial.pdf`](../app/models/hy_nomas/LICENSE-PixelUMM-One-Way-Noncommercial.pdf) (SHA-256 `4ff203c3f7997c7fed287a463d733f794934a79cfabb2936008fca0bcc8ad3d6`). Section 3.3 limits use to non-commercial research or evaluation. The official [PixelUMM model card](https://huggingface.co/nvidia/PixelUMM) confirms the source-code/checkpoint license split. Since those weights decode every world, generation and exploration are subject to that restriction. The runtime refuses model loading unless the app passes `acknowledge_pixelumm_noncommercial=True`; the app must present the notice and link before passing that acknowledgement.

The Qwen2.5-VL-7B text encoder is the existing Qwen-Image component and inherits the [upstream Qwen2.5-VL Apache-2.0 license](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). The converted text encoder and tokenizer files are served from `DeepBeepMeep/Qwen_image` at immutable revision `75c70695c563a235a5d14e37f110d499b3ecbd1c`.

### Immutable asset pins and header audit

The Hugging Face API metadata at the time of this audit resolved both `main` refs to immutable commits. Defaults and download definitions now use those commit SHAs; asset hashes are the Hub-reported LFS SHA-256 values. These pins describe the exact files and do not fetch them.

| Asset | Immutable revision | File SHA-256 | Size |
| --- | --- | --- | ---: |
| `DeepBeepMeep/Worlds/hy_nomas_distilled_int8.safetensors` | `545ad7b2393ad5e606c821fb04e07f8fec151046` | `973fa409f3b009232db3c83feddc89c5326fc3a5672feb791f0989ea8bf83252` | 9,059,126,120 bytes |
| `DeepBeepMeep/Worlds/hy_nomas/pixelumm_decoder_fp8.safetensors` | `545ad7b2393ad5e606c821fb04e07f8fec151046` | `5528323e171afd302cece26b05a8dfea8dc8d89e0501734c732b8ef36e098b12` | 7,301,500,088 bytes |
| `DeepBeepMeep/Qwen_image/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_bf16.safetensors` | `75c70695c563a235a5d14e37f110d499b3ecbd1c` | `09a25316d7d9a34686a44ded496e95af32404f95c2fb7e6ee02f4b53b153695e` | 16,584,431,274 bytes |
| `DeepBeepMeep/Qwen_image/Qwen2.5-VL-7B-Instruct/Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors` | `75c70695c563a235a5d14e37f110d499b3ecbd1c` | `18966ca6ba18261ec0465442580ad30510a758ecc0314871a0bbd30032716568` | 9,432,992,295 bytes |

The immutable Worlds commit's DiT header identifies `nomas-int8-bf16-single-v1`, group size 128, INT8 `weight_q` matrices, FP32 `weight_scale` tensors, compute dtype BF16, and source hash `73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70`. Its 775 quantized layer names and shapes all match the DiT metadata. After canonicalizing quantized keys, all 2,106 local meta-state entries are present with matching shapes. The header has 162 additional camera tensors: 54 each of `action_modules.{i}.img_attn_q.weight`, `action_modules.{i}.key_attn_q_norm.weight`, and `action_modules.{i}.key_attn_k_norm.weight` (`i=0..53`). They are omitted only under the pinned singleton-key attention profile proved below; the complete key/shape list and checkpoint provenance are persisted in [`camera_checkpoint_audit.json`](../app/models/hy_nomas/camera_checkpoint_audit.json). The decoder header identifies `pixelumm_fp8_bf16_single_v1`, FP8 E4M3 matrices, FP32 128-by-128 block scales, BF16 unquantized tensors, and source hash `0bbfe8fd44bd2ebf1059fe65079caba50fd6f52ea8e1b5b35393e3050b9b8378`. It is marked as the decoder-only tensor subset of the FP8 PixelUMM release; all 476 canonical tensor names and shapes match the local meta-state exactly. These header checks do not validate numerical quality or GPU performance.

### Camera attention checkpoint audit

The original Tencent inference preset at [revision `18974d41ada7af4aa3a026480197ab0ac0702682`](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/18974d41ada7af4aa3a026480197ab0ac0702682/configs/experiment/inference.yaml) sets `per_segment_camera_attn=false`, `per_chunk_camera_attn=true`, and `camera_chunk_size=1`. In Tencent's [`ActionModule`](https://github.com/Tencent-Hunyuan/Hy-Nomas/blob/18974d41ada7af4aa3a026480197ab0ac0702682/hymm/models/camera/action_module.py), per-chunk ranges associate each latent frame's spatial image queries with the matching single keyboard token. For one key, softmax is exactly 1: the query, key, their learned RMS norms, and camera RoPE cannot change the output. The active camera result is therefore `proj_keyboard(V(camera_token))`, broadcast to the spatial image tokens of that frame. The Maestro adapter retains and uses the keyboard embedding, K/V projection's value half, output projection, and pose embedding; it does not create unused Q or Q/K norm modules. `ignore_unused_weights=True` ignores just the 162 audited tensors listed in the JSON file.

The model definition carries this fixed camera attention profile into transformer construction. Unsupported requests for per-segment, multi-key, or larger-chunk camera attention raise an error. A CPU test compares the actual Maestro attention dispatcher against the one-key V-only reduction with seven image queries and three temporal keys split into 2/4/1-query frame groups, then checks the camera module's broadcast path. No multi-key camera attention is claimed or implemented.

Companion readiness must include the selected Qwen text-encoder weight file, all five tokenizer/configuration files, and the PixelUMM decoder file. The model checkpoint itself remains the base model URL. Primary integration still owns verifying asset presence in its real download/readiness flow before registering the model. Tests use synthetic tensors and do not download model weights.

## Standalone Worlds workflow foundation

`app/services/worlds_workflow.py` supplies the mounted API and pure request helpers. `app/services/worlds_storage.py` verifies completed media and saves immutable state under the separate Worlds workspace. The **Video → Create → Worlds** menu item opens `ui/src/components/Worlds/WorldExplorer.tsx`; it does not change the current Director, Studio or Editor mode. Hy-Nomas is registered for this dedicated workflow and filtered from ordinary Studio model choices.

The router accepts only an operation, its prompt/action fields, a canonical client UUID, a SHA-256 digest, and the explicit PixelUMM non-commercial acknowledgement. It constructs every generation request with `model_type=hy_nomas`, 832x480, 97 frames, 8 steps, guidance 1.0, and the allowlisted nested `custom_settings` values `world_output=world_video` and `acknowledge_pixelumm_noncommercial=true`; `_client_submission_id` equals the client UUID. A continuation or move resolves the world ID through the injected world store and passes only the app-resolved `.wgpworld` path as `custom_guide`. The client cannot choose paths, model IDs, geometry, steps, or backend settings. Moves use bounded integer milliseconds (100 ms through 10 s), default to 800 ms, and map to Hy-Nomas bracket instructions. Event actions map to `[Event] text`.

The browser writes the exact request UUID, digest, and payload to local storage before its single POST. An ambiguous result and a reloaded browser reconcile the server receipt through GET. The router writes its durable receipt before calling the normal generation submit callback and never submits an existing request ID again. The receipt is associated with the exact `_client_submission_id`. Cancellation first reads the latest job status, verifies both job ID and submission ID, writes a one-shot cancellation marker, and calls the injected cancellation callback at most once. It does not cancel work without that ownership proof.

The app's publication callback must be idempotent by request UUID. The router invokes it only after the exact owned job is terminal `completed`; it exposes a world only when that callback confirms the real video and `.wgpworld` sidecar with `verified`, `video_verified`, and `world_file_verified` all true. The storage callbacks must return sanitized public metadata; absolute filesystem paths and manifests do not cross the API boundary. The save route serves only a bounded `.wgpworld` path under the injected app workspace. Capture accepts a bounded JPEG. Recording calls a synchronous CPU callback in a worker thread; the callback concatenates already-saved creation/move clips. Imported worlds without that saved clip chain report recording unavailable. The HTTP route never calls model `recording_output()` and never launches an extra generation for recording.

The drawer presents the [NVIDIA PixelUMM non-commercial license](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf) before generation or file download and requires an explicit acknowledgement, including when downloading a recorded MP4. It labels the workflow experimental: each move is a queued generation and is not real-time movement. No inference speed, latency, or visual-quality claims are made. The current CPU workflow tests mock queued jobs and storage, use synthetic files, and do not download weights or access a GPU.

### Source provenance

The copied WanGP model source snapshot is revision `9905f0df941e5ec7b0c80e32555d3a61fe415726`. The following SHA-256 values were computed from the exact raw-source files retained during the port review; the table records their public provenance:

| Raw source file | Source SHA-256 | Maestro file matches raw source |
| --- | --- | --- |
| `models/hy_nomas/camera.py` | `96cd0d1ad1e49ea27b0fbe0af1ab6babf37a81299a2f354397917545d080968c` | No; camera-profile validation and lint adaptation |
| `models/hy_nomas/exploration.py` | `16fd7b4eacbb4256f0d7a8b79d23583910c6a9f577f36ca938f6193d952731c7` | No; adapted for model-local queue contract |
| `models/hy_nomas/history.py` | `9eef6f3059047bbe2dd6cb6be34d836d493791de4046735f735387fb13b3e956` | No; adapted to local Q8 history fallback |
| `models/hy_nomas/hy_nomas_handler.py` | `605858843198216c58bffb3904c0ecb6eb686402db7b63cc9c2678926827ead2` | No; local handler has pinned assets and license gate |
| `models/hy_nomas/hy_nomas_main.py` | `a42fab677beacff77fd46162f609dcb009bfc2eb877c037df60fdac24a54006d` | No; adapted to local session and cancellation APIs |
| `models/hy_nomas/pixelumm.py` | `bbb9c461542dc7cbb99e57e4e42d708d641e99d6c92e2f10e1ca8a598e2c0376` | No; adapted to local attention and ops adapters |
| `models/hy_nomas/prompts.py` | `5487893c1a423fc7e4804aa31fff627256e4b990058237f04684ba84bfc7b3d8` | Yes |
| `models/hy_nomas/texts.py` | `cb060166a33a074b8072e998983b94f1a8654b0a59cf32813186e41ed6c9663f` | Yes |
| `models/hy_nomas/transformer.py` | `a253c53abd6086f4820030e3f07b3118ca513628cb89329c2a348bb1d321bc30` | No; adapted to MMGP 4.0 attention and ops |
| `models/hy_nomas/triton_kernels.py` | `61db2ea42d9097d9aeaa8cefdb25cf75ee2749088490770dead7f542568a24a7` | Yes |
| `models/hy_nomas/world_file.py` | `87b2c7b7f5410f98fa90ab8c4b5105c1195cc2f4cbfb910bab7367b8ddd1a238` | No; replaced with bounded ZIP/tensor validation |

The block-scaled adapter source came from `shared/qtypes/block_scaled.py` at that exact commit (original SHA-256 `76f5d5f9bf1144fc5feaa6f48813b53bd93f466faa29e794aac50bf574be28e2`) and was adapted because this Maestro branch does not have the newer shared FP8/INT8 kernel APIs. The Tencent model code source notice pins `18974d41ada7af4aa3a026480197ab0ac0702682`; the PixelUMM code license pin is `e19e91aabb99784f7b5524268ed42ec59c1ad660`. No model-weight files are included in this source tree or its CPU tests; tests use synthetic tensors.
