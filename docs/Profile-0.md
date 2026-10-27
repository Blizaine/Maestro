# Experimental Profile 0

Profile 0 is staged behind an explicit process-start runtime choice. The
current default remains Maestro MMGP 4.0; an absent or `legacy` preference
continues to resolve `app/mmgp`. The opt-in key is `wangp9905`, displayed as
**Experimental Profile0 (WanGP9905)**. Restart the server to change it.

`read_saved_runtime(config_path)` reads this preference before Torch or MMGP
imports. A missing config file or missing `memory_runtime` key selects the
legacy backend; an explicit value must be exactly `legacy` or `wangp9905`.
Invalid JSON and invalid explicit values fail clearly. The reader is
read-only and does not migrate or rewrite the saved configuration.

The selector API is `services.memory_runtime.select_runtime(app_dir,
saved_preference)`. Pass the result of `read_saved_runtime()` and call the
selector immediately after adding `app` to `sys.path` and
before importing `shared.cuda_memory`, `mmgp`, or any service that might import
MMGP. `shared.cuda_memory.apply_startup_settings()` can import `mmgp.allocator`
before the ordinary `import mmgp` later in `wgp.py`. After import,
`verify_runtime_import(selection)` confirms the selected package path and real
module version. It fails with a restart message if a different backend has
already loaded; the two versions must never share a process through aliases.
The selector does not modify server settings.

`runtime_snapshot(selection, cfg)` is a read-only status view for saved versus
active backend, restart requirement, selected label, static VMM and RAM
companion-file presence, and Fish capability for both the saved and active
backend. It does not probe CUDA or load an allocator. If the already-imported
MMGP module cannot be matched to either local package, active status remains
unverified rather than being inferred from the saved preference.
`validate_runtime_updates(body, cfg, selection)` and
`validate_system_memory_updates(body, cfg, selection)` are pure preflights for
the runtime, allocator, and profile fields. The System API runs these and its
other field/folder/notification checks before any config or locator mutation.
These functions never mutate `body` or `cfg`.
Profile 0 can be saved only when the resulting backend is `wangp9905`, the
resulting allocator is exactly `vmm`, and the pinned VMM file is present in
the staged backend package. Existing profiles are kept as saved; choosing a
runtime does not remap or reset them. Saving these choices does not activate
the backend in the current process: restart is required, and generation still
fails closed until startup reports the VMM allocator active.

The snapshot mapping exposes `memory_runtime`, `saved_runtime`, `active_runtime`,
`restart_required`, `selected_label`, `active_label`,
`experimental_runtime_selectable`, `experimental_runtime_capability_reason`,
`profile0_native_available`, `profile0_native_abi_verified`,
`profile0_native_validation_reason`, `profile0_capability_reason`,
`profile0_ram_native_available`, `profile0_ram_capability_reason`,
`profile0_saved_configuration_eligible`, `fish_supported`, `fish_limitation`,
`active_fish_supported`, and `active_fish_limitation`. The combined preflight
returns memory keys and runtime/profile keys separately for the API caller to
include in its all-fields validation before persisting anything.

The System settings selector stores `memory_runtime` without changing profile,
allocator, pinning, preload, or Auto-tune settings. The experimental backend
choice is disabled until its pinned platform VMM file is present. Switching the
runtime requires a full restart. Experimental Profile0 is called out as
unsupported for Fish Audio and as having different placement behavior for
Profiles 1 and 3. The Profile 0 choices appear only after the saved runtime is
WanGP9905, the saved VRAM allocator is `vmm`, and the platform VMM binary is
present. Existing saved Profile 0 values are displayed disabled if those
checks are not met so a user can change them away without an implicit reset.
The snapshot reports file presence only: it does not validate the native ABI,
driver, or GPU behavior, and a present file must not be treated as proof of
runtime safety.

Read-ahead is rejected when the experimental runtime is selected; users must
turn it off themselves before switching. The experimental RAM allocator is
also rejected until its companion native binary is present, and the API asks
the user to change that preference explicitly. The UI leaves Smart Memory
Pinning as saved and explains that Profile 0's shared pool determines staging.

`profile_options(selection, 0, manual_options)` returns the upstream defaults
`pinPool=True`, `cpuCompute="auto"`, `autoPreload=True`, and `cudaGraphs=False`.
The caller must still pass profile number `0` to the selected backend. Explicit
manual options are preserved over defaults. `require_profile()` rejects
Profile 0 on the default backend and preserves the existing values 1, 2, 3,
3.5, 4, 4.5, and 5 without remapping. Profile 0 is not a renamed legacy
profile. Its actual settings come from the upstream `offload.profile(...,
profile_no=0, ...)` implementation.

For every WanGP9905 profile, `profile_options()` strips Maestro's disabled
`readAhead=False` kwarg because MMGP 4.4 removed that option; an enabled
read-ahead request fails closed. It also drops an empty `budgets={}` sentinel
that Maestro's `init_pipe()` adds when a model supplied no budget override,
preserving the selected MMGP profile's own budget defaults. Non-empty manual
budgets and legacy backend options pass through unchanged. WanGP9905 graph
requests remain rejected; Maestro's existing Fish graph behavior is unchanged
on the legacy backend.

## Activation requirements

The package omits upstream prebuilt allocator binaries. Its Windows x64
candidate includes a local VMM build from the exact pinned C++ source;
`app/mmgp_backends/v43/mmgp/allocator/VMM_BUILD.json` records the build and
artifact hashes. It also includes the optional Windows x64 OOM companion,
built from the same pinned upstream source with PyTorch 2.10 headers.
`OOM_BUILD.json` records its source and binary hashes, compiler, dependencies,
and CPU exception checks with installed PyTorch 2.7.1 and 2.10.0. This restores
the upstream native OOM exception type and avoids a missing-companion warning
that Pinokio otherwise mistakes for a startup error. Those CPU checks do not
certify recovery from an actual GPU OOM. Other platforms need their matching build before this option
becomes available. File presence alone does not establish ABI or GPU behavior;
no allocator is loaded by the settings API.
`require_profile_runtime()` fails closed until the matching VMM allocator is
present and reports `active == "vmm"`. The upstream 4.3 metadata requires
PyTorch 2.1+, but VMM hook registration in the source requires PyTorch 2.3+;
the optional OOM and RAM allocator companion libraries are only checked for
PyTorch 2.6–2.15. On Windows, the required file is
`mmgp/allocator/vmm_alloc_win_amd64.dll`. The source build uses Visual Studio
Build Tools with C++17 and x64 tools; it does not need the CUDA toolkit. The
runtime still needs an NVIDIA CUDA 11.2+ driver and hardware supporting CUDA
VMM and memory pools. Install the allocator before the first CUDA context is
created, then restart. No allocator is installed by this staged helper.
The pinned package declares `torch>=2.1.0`, `optimum-quanto`, `accelerate`,
`safetensors`, and `psutil`; no dependency was installed for this staging.

Profile 0 graph capture is rejected. The upstream graph work, Maestro's
whole-frame graph hooks, cancellation, and teardown have not been exercised
together on GPU. Fish Audio is also rejected on this runtime: Maestro's Fish
adapter currently uses the old `formats=` / `dequantize(tensors)` expert
contract and optional native CPU/Triton paths; the pinned MMGP source uses a
different `ExpertFormat` and expert residency contract. Fish must continue on
MMGP 4.0 until that adapter and its CPU, GPU, graph, and cleanup behavior are
ported and verified.

## Verification boundary

CPU/offline checks cover import resolution, the package's Python modules,
profile arguments, missing allocator behavior, graph rejection, MMGP manager
teardown, API preflight purity, saved-choice preservation, and UI capability
gates. Both application entry points select the saved runtime before allocator
imports, and generation checks the active allocator before loading a model.
No GPU allocation safety, memory or speed improvement,
automatic-preload quality, CUDA graph parity, or compatibility with every
Maestro pipeline has been established. Do not generate with this backend until
the pinned native allocator and restart/generation gates have been separately
reviewed and tested.

The immutable source is recorded in
`app/mmgp_backends/v43/UPSTREAM_SOURCE.md`. Its `pyproject.toml` says 4.3.0,
while `__init__.py` reports 4.4.0; the UI label and import check follow the
runtime source version rather than hiding that drift.
