import { create } from 'zustand'
import type { GenerateParams, OutputFile, MediaFilter, AspectRatio, ResolutionPreset, ScailResolutionProfile, GenerationJob, ModelFamily, ModelDef, GenerationMode, ModelOptions, SystemConfig, SettingsTab, OutputMetadata, MultiClip, ServicesConfig, LlmStatus, LlmModelOption, AudioAnalysisResult, PlannedClip, ClipPlan, DirectorClipImage, DirectorImageGenProgress, SpeakerMapping, DirectorSkill, DirectorShotImageGuidance, ShortFilmCharacter, ShortFilmPath, CivitAIModel, CivitAIDownload, PipelineListItem, PipelineRepairState, SavedPipelineState, SystemDetectResponse, SystemStats, RecastCharacterMapping, RepaintRegionMapping, H3WindowPlan } from '../types'
import * as api from '../api/client'
import { applyThemePrefs, getStoredPrefs, type FamilyId, type ThemeMode, type ThemePrefs } from '../lib/theme'
import {
  effectiveH3OmniSequenceFrames,
  h3WindowOverrideKey,
  h3OmniSequenceWindowCount,
  h3SlidingWindowCount,
  normalizeH3ClipFrameSchedule,
  normalizeH3ClipFrames,
  normalizeH3NativeFrames,
  recommendedH3PassProfile,
  recommendedH3OmniSequenceProfile,
} from '../lib/h3Memory'

const CIVIT_DOWNLOAD_POLL_MS = 2000
const CIVIT_DOWNLOAD_COMPLETED_VISIBLE_MS = 30_000
let _civitDownloadPollTask: Promise<void> | null = null
let _civitDownloadPollController: AbortController | null = null
let _civitDownloadPollRequested = false
const _civitRefreshedCheckpointDownloads = new Set<string>()
const DIRECTOR_REPAIR_POLL_MS = 2000
const DIRECTOR_REPAIR_ACTIVE = new Set(['queued', 'running', 'cancelling'])
type DirectorRepairPoll = {
  operationId: string
  timer: number | null
}
const _directorRepairPolls = new Map<string, DirectorRepairPoll>()
const _directorRepairDiscoveries = new Map<string, object>()
let _dashboardPipelineLoadToken = 0
let _dashboardPipelineListLoadToken = 0
let _h3WindowOverridesHydrated = false
let _h3WindowOverrideSaveTask: Promise<void> = Promise.resolve()

function _saveH3WindowOverrides(overrides: Record<string, number>) {
  _h3WindowOverrideSaveTask = _h3WindowOverrideSaveTask
    .catch(() => { /* a later save should still run */ })
    .then(async () => {
      await api.updateH3WindowOverrides(overrides)
    })
    .catch(error => {
      console.warn('Failed to save H3 window overrides:', error)
    })
}

type OutpaintAspect = 'source' | '16:9' | '9:16' | '1:1' | '4:3' | '3:4'

function _normalizeSlidingWindowOverlap(
  value: number,
  defaults?: Record<string, number> | null,
): number {
  if (!defaults) return Math.max(0, Math.round(value))
  const minimum = defaults.overlap_min ?? 1
  const maximum = defaults.overlap_max ?? Math.max(minimum, value)
  const step = Math.max(1, defaults.overlap_step ?? 1)
  const offset = defaults.overlap_offset ?? minimum
  const normalized = offset + Math.round((value - offset) / step) * step
  return Math.max(minimum, Math.min(maximum, normalized))
}

const _OUTPAINT_ASPECT_RATIOS: Array<[Exclude<OutpaintAspect, 'source'>, number]> = [
  ['16:9', 16 / 9],
  ['9:16', 9 / 16],
  ['1:1', 1],
  ['4:3', 4 / 3],
  ['3:4', 3 / 4],
]

function _inferOutpaintAspect(width: number, height: number): OutpaintAspect | null {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return null
  const ratio = width / height
  let nearest: Exclude<OutpaintAspect, 'source'> | null = null
  let nearestError = Number.POSITIVE_INFINITY
  for (const [aspect, target] of _OUTPAINT_ASPECT_RATIOS) {
    const relativeError = Math.abs(ratio - target) / target
    if (relativeError < nearestError) {
      nearest = aspect
      nearestError = relativeError
    }
  }
  // Grid alignment can move either dimension by several pixels. Four percent
  // safely recognizes those canvases without pretending an arbitrary ratio
  // is one of the six choices supported by the composer.
  return nearestError <= 0.04 ? nearest : null
}

function _repairNeedsPolling(repair: PipelineRepairState | null | undefined): boolean {
  return !!repair && DIRECTOR_REPAIR_ACTIVE.has(repair.status)
}

function _stopDirectorRepairPoll(pid: string): void {
  const poll = _directorRepairPolls.get(pid)
  if (poll?.timer != null) window.clearTimeout(poll.timer)
  _directorRepairPolls.delete(pid)
}

function _downloadTimestampMs(value: number | null | undefined): number | null {
  const timestamp = Number(value)
  if (!Number.isFinite(timestamp) || timestamp <= 0) return null
  return timestamp < 1_000_000_000_000 ? timestamp * 1000 : timestamp
}

function _downloadNeedsPolling(download: CivitAIDownload, now: number): boolean {
  if (download.status === 'downloading') return true
  if (download.status !== 'completed') return false
  const completedAt = _downloadTimestampMs(download.completed_at)
  return completedAt !== null && now - completedAt < CIVIT_DOWNLOAD_COMPLETED_VISIBLE_MS
}

function _waitForDownloadPoll(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise(resolve => {
    if (signal.aborted) {
      resolve()
      return
    }
    const timer = window.setTimeout(done, ms)
    function done() {
      window.clearTimeout(timer)
      signal.removeEventListener('abort', done)
      resolve()
    }
    signal.addEventListener('abort', done, { once: true })
  })
}

// Vite can replace this module without a full page unload. Abort the old
// async loop so HMR never leaves an orphaned polling timer behind.
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    _civitDownloadPollController?.abort()
    _civitDownloadPollController = null
    _civitDownloadPollTask = null
    _civitDownloadPollRequested = false
    _civitRefreshedCheckpointDownloads.clear()
    for (const pid of _directorRepairPolls.keys()) {
      _stopDirectorRepairPoll(pid)
    }
    _directorRepairDiscoveries.clear()
  })
}

// --- LocalStorage persistence for per-mode settings ---
const STORAGE_KEY = 'maestro_mode_settings'

// Persistence schema version. Bump when changing the LoRA-key strategy or
// adding fields that need migration. Currently:
//   v1: savedLoraPerMode is keyed by lora_id (e.g. `civitai:12345`) instead
//       of filename, so settings survive LoRA version bumps. A snapshot of
//       lora_id → filename at save time is embedded for fast load-time
//       translation; reconciliation against the fresh map (fetched from
//       /api/v1/loras/installed) happens after boot in `loadModels()`.
const _PERSIST_VERSION = 1

type LoraModeBlob = { activated_loras: string[]; loras_multipliers: string; loraWeights: Record<string, number[]>; availableLoras: string[] }

/** Per-mode params snapshot stored in localStorage. Holds whatever
 *  GenerateParams the user had set in that mode, plus a couple of
 *  top-level store fields (filmGrain*) that conceptually belong to
 *  the mode but live outside `params`. Each mode keeps its own
 *  complete snapshot so settings don't leak between modes — this
 *  fixed bugs where e.g. `repeat_generation: 10` set in image mode
 *  would queue up 10 videos when the user switched to video mode,
 *  or `video_prompt_type: 'KFI'` (frames injection) would persist
 *  on a mode where it didn't apply. Partial<GenerateParams> because
 *  the user almost never sets every field. */
type SavedModeParams = Partial<GenerateParams> & {
  filmGrainIntensity?: number
  filmGrainSaturation?: number
  /** Top-level store field (NOT in GenerateParams), saved per-mode so
   *  audio's 600/1800 slider.max doesn't leak into video on mode switch.
   *  See setGenerationMode for the save/restore wiring. */
  durationSeconds?: number
}

interface PersistedModeSettings {
  generationMode: GenerationMode
  selectedModelPerMode: Partial<Record<GenerationMode, string>>
  savedParamsPerMode: Partial<Record<GenerationMode, SavedModeParams>>
  /** Runtime shape (filename-keyed). The on-disk shape is lora_id-keyed
   *  starting with v1; the persistence layer translates transparently. */
  savedLoraPerMode: Partial<Record<GenerationMode, LoraModeBlob>>
  /** Per-mode main prompt (lyrics in audio mode). Tracked separately from
   *  the params snapshot in memory. Still written for shape stability but
   *  NO LONGER rehydrated on boot — a refresh starts with a clean prompt
   *  (see the partial-hydration note in loadModels). */
  savedPromptPerMode?: Partial<Record<GenerationMode, string>>
  /** Snapshot of lora_id → filename captured at last save. Returned by
   *  `_loadSettings` for use in mid-session reconciliation when the fresh
   *  lora map arrives, so we can rewrite filenames that changed since save. */
  _loraFilenameSnapshot?: Record<string, string>
}

/** Build a lora_id-keyed copy of a single LoraModeBlob using filename → lora_id.
 *
 *  Multi-version disambiguation: if two filenames in the same blob share a
 *  lora_id (e.g. user keeps v1 + v2 of the same CivitAI model on disk for
 *  A/B testing), use a `{lora_id}#{filename}` suffix for the collision so
 *  each file's settings persist independently. Without this, the second
 *  file's loraWeights overwrite the first's via Object.fromEntries, and
 *  cross-session A/B silently loses one version's weights. */
function _modeBlobToLoraIdKeyed(
  m: LoraModeBlob,
  filenameToLoraId: Record<string, string>
): LoraModeBlob {
  const baseId = (fname: string) => filenameToLoraId[fname] || `local:${fname}`
  // Detect collisions across the whole blob: count how many filenames in
  // (activated_loras ∪ loraWeights ∪ availableLoras) map to each base id.
  const idCounts: Record<string, number> = {}
  const seen = new Set<string>([
    ...(m.activated_loras || []),
    ...Object.keys(m.loraWeights || {}),
    ...(m.availableLoras || []),
  ])
  for (const fname of seen) {
    const bid = baseId(fname)
    idCounts[bid] = (idCounts[bid] || 0) + 1
  }
  const id = (fname: string): string => {
    const bid = baseId(fname)
    return (idCounts[bid] || 0) > 1 ? `${bid}#${fname}` : bid
  }
  return {
    ...m,
    activated_loras: (m.activated_loras || []).map(id),
    loraWeights: Object.fromEntries(
      Object.entries(m.loraWeights || {}).map(([fname, w]) => [id(fname), w])
    ),
    availableLoras: (m.availableLoras || []).map(id),
  }
}

/** Reverse: lora_id-keyed blob → filename-keyed using lora_id → filename map.
 *
 *  Disambiguated keys (`{loraId}#{filename}`) carry the filename in the
 *  suffix — extract it directly so multi-version A/B state round-trips
 *  losslessly. */
function _modeBlobToFilenameKeyed(
  m: LoraModeBlob,
  loraIdToFilename: Record<string, string>
): LoraModeBlob {
  const fname = (id: string): string => {
    const hashIdx = id.indexOf('#')
    if (hashIdx > 0) return id.slice(hashIdx + 1)
    return loraIdToFilename[id] || (id.startsWith('local:') ? id.slice(6) : id)
  }
  return {
    ...m,
    activated_loras: (m.activated_loras || []).map(fname),
    loraWeights: Object.fromEntries(
      Object.entries(m.loraWeights || {}).map(([id, w]) => [fname(id), w])
    ),
    availableLoras: (m.availableLoras || []).map(fname),
  }
}

/**
 * Persist mode settings. The on-disk shape is lora_id-keyed (so that
 * filename changes from LoRA version bumps are transparent on reload),
 * with an embedded `_loraFilenameSnapshot` so the next load can translate
 * back to filenames immediately without waiting for the fresh map.
 *
 * If no map is provided (e.g. very early in boot before /installed has
 * returned), we skip translation and write the legacy filename-keyed shape
 * with no version flag. The next save with a populated map will upgrade it.
 */
/** Fields in SavedModeParams that hold file paths or per-gen ephemeral
 *  inputs which should NEVER persist across browser sessions. Persisting
 *  these caused the "ghost reference" bug: on page reload the cached
 *  paths would rehydrate from localStorage and the next generation
 *  would submit them, so users would silently get image-to-image edits
 *  against stale uploads they no longer had selected. Same pattern hit
 *  frame-injection positions in LTX-2 video mode and audio guide refs
 *  in TTS modes.
 *
 *  Rule of thumb: anything pointing to a path under app/uploads/ or any
 *  ephemeral per-job input belongs here. Anything the user genuinely
 *  wants remembered (model settings, slider values, video_prompt_type
 *  letter codes, etc.) stays out of this list and continues to persist.
 *
 *  Workaround for users on a Maestro version before this fix: use a
 *  private/incognito browser window (skips localStorage rehydration).
 */
const EPHEMERAL_PARAM_FIELDS: ReadonlyArray<keyof SavedModeParams> = [
  'image_start',
  'image_end',
  'image_refs',
  'video_guide',
  'video_mask',
  'video_source',
  'audio_guide',
  'audio_guide2',
  'frames_positions',
]

function _stripEphemeralParams(perMode: Partial<Record<GenerationMode, SavedModeParams>>): Partial<Record<GenerationMode, SavedModeParams>> {
  const cleaned: Partial<Record<GenerationMode, SavedModeParams>> = {}
  for (const [mode, params] of Object.entries(perMode || {})) {
    if (!params) continue
    const copy: SavedModeParams = { ...params }
    for (const field of EPHEMERAL_PARAM_FIELDS) {
      delete copy[field]
    }
    // The "T" temporal-alignment flag only means something alongside a
    // video_source — which is ephemeral-stripped above. Persisting a lone
    // "T" produced a ghost Advanced badge (counts as an active process
    // choice while displaying as nothing). Strip it on the way in AND out
    // so existing users' stale snapshots heal on next load. Only a TRAILING
    // "T" is the flag — an internal "T" is the depth_temporal control letter
    // (TVG/PTVG/TEVG) and a global strip silently downgraded those to plain
    // pose/spatial, so use /T$/.
    if (typeof copy.video_prompt_type === 'string' && copy.video_prompt_type.endsWith('T')) {
      copy.video_prompt_type = copy.video_prompt_type.replace(/T$/, '')
    }
    cleaned[mode as GenerationMode] = copy
  }
  return cleaned
}

function _saveSettings(
  state: PersistedModeSettings,
  filenameToLoraId?: Record<string, string>,
) {
  try {
    // Strip file-bearing / ephemeral fields BEFORE serializing so they
    // never round-trip through localStorage. The in-memory store keeps
    // them for the current session; only the persisted snapshot is
    // pruned. See EPHEMERAL_PARAM_FIELDS comment for the full rationale.
    const sanitizedParamsPerMode = _stripEphemeralParams(state.savedParamsPerMode || {})

    if (filenameToLoraId && Object.keys(filenameToLoraId).length > 0) {
      // Translate savedLoraPerMode → lora_id keys
      const translatedPerMode: Partial<Record<GenerationMode, LoraModeBlob>> = {}
      for (const [mode, m] of Object.entries(state.savedLoraPerMode || {})) {
        if (m) translatedPerMode[mode as GenerationMode] = _modeBlobToLoraIdKeyed(m, filenameToLoraId)
      }
      // Snapshot: lora_id → filename (so load can translate back instantly)
      const snapshot: Record<string, string> = {}
      for (const [fname, id] of Object.entries(filenameToLoraId)) snapshot[id] = fname
      const payload = {
        _version: _PERSIST_VERSION,
        _loraFilenameSnapshot: snapshot,
        generationMode: state.generationMode,
        selectedModelPerMode: state.selectedModelPerMode,
        savedParamsPerMode: sanitizedParamsPerMode,
        savedLoraPerMode: translatedPerMode,
        savedPromptPerMode: state.savedPromptPerMode,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
    } else {
      // No map yet — write legacy filename-keyed shape, no version. Will be
      // upgraded on next save with a populated map. Still apply the ephemeral
      // strip on the way out.
      const sanitizedState = { ...state, savedParamsPerMode: sanitizedParamsPerMode }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizedState))
    }
  } catch { /* quota exceeded or private browsing */ }
}

function _loadSettings(): PersistedModeSettings | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    // v1+: savedLoraPerMode is lora_id-keyed; use the embedded snapshot to
    // translate back to filenames immediately. Reconciliation against the
    // fresh map happens in loadModels() once /installed returns.
    if (parsed && parsed._version === _PERSIST_VERSION && parsed._loraFilenameSnapshot) {
      const snapshot: Record<string, string> = parsed._loraFilenameSnapshot
      const translated: Partial<Record<GenerationMode, LoraModeBlob>> = {}
      for (const [mode, m] of Object.entries(parsed.savedLoraPerMode || {})) {
        if (m) translated[mode as GenerationMode] = _modeBlobToFilenameKeyed(m as LoraModeBlob, snapshot)
      }
      return {
        generationMode: parsed.generationMode,
        selectedModelPerMode: parsed.selectedModelPerMode || {},
        // Strip ephemeral file-bearing fields at load too — protects existing
        // users whose localStorage was written by a pre-fix version and still
        // contains stale image_start / image_refs / etc. paths. New saves will
        // be already-clean from _saveSettings; this is the migration safety
        // net so the first post-update page load can't immediately rehydrate
        // ghost references.
        savedParamsPerMode: _stripEphemeralParams(parsed.savedParamsPerMode || {}),
        savedLoraPerMode: translated,
        savedPromptPerMode: parsed.savedPromptPerMode || {},
        _loraFilenameSnapshot: snapshot,
      }
    }
    // Legacy (no version): blob is already filename-keyed, return as-is —
    // but still strip ephemeral fields out for the same migration-safety reason.
    const legacy = parsed as PersistedModeSettings
    return {
      ...legacy,
      savedParamsPerMode: _stripEphemeralParams(legacy.savedParamsPerMode || {}),
    }
  } catch { return null }
}

/** Fetch a model's defaults from the backend and merge primary fields
 *  into params. Shared between `selectModel` (explicit model pick) and
 *  `setGenerationMode` (mode switch where the per-mode active model
 *  may change). Without this, switching from LTX-2 (8 steps) to Flux 2
 *  Klein 9B (4 steps) or HiDream Dev (28 steps) would silently keep
 *  the slider at the previous model's value.
 *
 *  Only overrides "primary" model-tuned numeric fields. Leaves
 *  user-intent fields (prompt, seed, negative_prompt, resolution,
 *  repeat_generation, activated_loras) alone — those should survive
 *  model switches.
 *
 *  Race-safe: applies only if the same model is still active when
 *  the fetchDefaults promise resolves. Guards against rapid model
 *  switching from leaving a stale model's defaults applied.
 */
// String list — some of these (sample_solver, embedded_guidance_scale,
// audio_guidance_scale) aren't declared on GenerateParams but the
// params object is loose enough to carry them through to the backend.
const _PRIMARY_MODEL_DEFAULT_FIELDS: ReadonlyArray<string> = [
  'num_inference_steps',
  'guidance_scale',
  'flow_shift',
  'sample_solver',
  'embedded_guidance_scale',
  'audio_guidance_scale',
  // Perturbation config for the STG slider. These are inert unless
  // perturbation_switch === 2, which startGeneration derives from the
  // STG slider — the server-side fallback layers ([9]) are wrong for
  // LTX-2 22B (needs [28] from the model's settings file), so the
  // model-correct values must ride along with the request.
  // Deliberately NOT copied: perturbation_switch and stg_scale — older
  // generated settings files carry perturbation_switch: 2 / stg_scale: 1.0
  // from the settings-file era, and copying them would silently re-enable
  // STG on every generation.
  'perturbation_layers',
  'perturbation_start_perc',
  'perturbation_end_perc',
  // Default state of the Reference Pipeline toggle (10Eros defs set it to
  // true). Only copied when the model's settings carry the key, so models
  // without it keep whatever the user last chose — and startGeneration
  // strips it for models that lack the capability anyway. Unchecking the
  // toggle holds until the model is re-selected, same as steps/guidance.
  'reference_pipeline',
  // LM sampling knobs for the ACE-Step 1.5 family (and other LM-staged
  // audio models). Their handlers seed tuned values (temperature 0.85,
  // top_p 0.9, top_k off, LM CFG 2.5); without hydration the UI showed
  // and SENT its generic temperature 1.0. Only models whose defaults
  // carry these keys are affected — video model settings don't include
  // them, so nothing changes there.
  'temperature',
  'top_p',
  'top_k',
  'alt_guidance_scale',
  // Sliding-window geometry. The UI only writes sliding_window_size when
  // the user touches the Advanced slider, so without hydration a request
  // carries NO window size and the backend inherits one from unrelated
  // primary settings — SCAIL-2 (window default 81) then ran a 10s
  // generation as a single 160-frame window and overflowed VRAM at
  // resolutions that fit fine per-window. LTX-2's defaults carry 481
  // (~19s), so typical LTX generations stay single-window as before.
  'sliding_window_size',
  'sliding_window_overlap',
  // Native video-to-video editing defaults. These are model-owned controls,
  // while the uploaded source and mask paths remain ephemeral.
  'denoising_strength',
  'masking_strength',
  // Control-video coupling for the SCAIL-2 / Wan-Animate class:
  // force_fps "control" makes the output follow the guide video's frame
  // rate (user-reported: 25fps source came out 16fps without it), and
  // audio_prompt_type "R" remuxes the guide's audio track into the
  // output (user-reported: outputs were silent). Only the scail2 model
  // settings carry force_fps; every other model's audio_prompt_type
  // defaults to "" which matches the UI default, so nothing changes
  // elsewhere.
  'force_fps',
  'audio_prompt_type',
]

// Monotonic sequence for loadModelOptions staleness detection — only the
// most recently requested model's options may touch the store.
let _modelOptionsSeq = 0

function _applyModelDefaults(
  storeGet: () => { selectedModelPerMode: Partial<Record<GenerationMode, string>>; generationMode: GenerationMode; params: GenerateParams },
  storeSet: (fn: (s: { params: GenerateParams }) => { params: GenerateParams }) => void,
  modelType: string,
): void {
  api.fetchDefaults(modelType).then((d) => {
    if (!d || typeof d !== 'object') return
    // Race guard: model may have been switched again while this fetch
    // was in flight. Apply only if still the active model in current mode.
    const state = storeGet()
    const active = state.selectedModelPerMode[state.generationMode]
    if (active !== modelType) return
    const overrides: Record<string, unknown> = {}
    for (const field of _PRIMARY_MODEL_DEFAULT_FIELDS) {
      // A one-click Full -> Pruned Turbo recommendation switches models and
      // then restores the managed 6-step preset. Do not let the asynchronous
      // base-model defaults response race in afterward and put it back at
      // 20 steps. loadModelOptions applies the same Turbo contract.
      if (
        field === 'num_inference_steps'
        && modelType.startsWith('minimax_h3')
        && state.params.minimax_h3_turbo_mode === true
      ) {
        continue
      }
      if ((d as Record<string, unknown>)[field] !== undefined) {
        overrides[field] = (d as Record<string, unknown>)[field]
      }
    }
    if (Object.keys(overrides).length > 0) {
      storeSet(s => ({ params: { ...s.params, ...overrides } as GenerateParams }))
    }
  }).catch(() => { /* fetch failure shouldn't break model switch */ })
}

// Family → generation mode mapping
const familyModeMap: Record<string, GenerationMode> = {
  flux: 'image',
  flux2: 'image',
  qwen: 'image',
  z_image: 'image',
  krea2: 'image',
  hidream: 'image',
  wan: 'video',
  wan2_2: 'video',
  hunyuan: 'video',
  hunyuan_1_5: 'video',
  ltxv: 'video',
  ltx2: 'video',
  kandinsky5: 'video',
  tts: 'audio',
  longcat: 'avatar',
}

// Model types classified as Avatar even though their family is primarily Video
const avatarModelTypes = new Set([
  'multitalk',
  'multitalk_720p',
  'fantasy',
  'infinitetalk',
  'infinitetalk_multi',
  'steadydancer',
  'i2v_2_2_multitalk',
  'animate',
  'hunyuan_avatar',
])

// Model types classified as Video Edit (Kiwi Edit, Chrono Edit)
const videoEditModelTypes = new Set([
  'kiwi_edit',
  'kiwi_edit_instruct_only',
  'kiwi_edit_reference_only',
  'chrono_edit',
  'chrono_edit_distill',
  'lucy_edit_fastwan',
  'lucy_edit_fastwan_1_1',
  // Dedicated to Edit → Recast; the general SCAIL Fast profile remains in
  // Studio Video/Animate.
  'scail2_14B_recast_fast',
])

// Audio sub-families: split the single "tts" family into Speech, Music, SFX
const audioSubFamilies: ModelFamily[] = [
  { id: 'tts_speech', label: 'Text to Speech', order: 200 },
  { id: 'tts_music', label: 'Music', order: 201 },
  { id: 'tts_sfx', label: 'Sound Effects', order: 202 },
]

// Model types that belong to the Music sub-family (everything else in
// tts → Speech). Membership is prefix-based for the known music model
// lines so newly added variants (e.g. new ACE-Step checkpoints)
// classify correctly without touching this file — the XL SFT models
// were invisible in the Music group because an id list here missed
// them. Keep the explicit set for one-off ids that don't share a
// prefix with their line.
const musicModelTypes = new Set<string>([])
const musicModelPrefixes = ['ace_step', 'heartmula', 'minimax_music3']

function isMusicModelType(modelType: string): boolean {
  if (musicModelTypes.has(modelType)) return true
  return musicModelPrefixes.some(p => modelType.startsWith(p))
}

// Model types that belong to the SFX sub-family (MMAudio variants)
const sfxModelTypes = new Set([
  'mmaudio_v2',
  'mmaudio_nsfw',
])

// Virtual MMAudio model entries (injected into model list alongside backend models)
const SFX_VIRTUAL_MODELS: ModelDef[] = [
  { model_type: 'mmaudio_v2', name: 'MMAudio v2', family: 'tts', architecture: 'mmaudio', is_i2v: false, is_t2v: false, guidance_max_phases: 1, fps: 0, is_downloaded: true },
  { model_type: 'mmaudio_nsfw', name: 'MMAudio NSFW', family: 'tts', architecture: 'mmaudio', is_i2v: false, is_t2v: false, guidance_max_phases: 1, fps: 0, is_downloaded: false },
]

// Default enabled models (shown by default in selectors)
const DEFAULT_ENABLED_MODELS = new Set([
  // Image
  // Keep the general-purpose Flux default plus the complete Krea 2 family:
  // base RAW/Turbo generation and their identity-preserving Edit variants.
  // Other image models remain opt-in through Model Visibility.
  'flux2_klein_9b',
  'krea2_raw',
  'krea2_turbo',
  'krea2_raw_edit',
  'krea2_turbo_edit',
  // Video
  // Default to just the LTX-2.3 Distilled 1.1 22B checkpoint (newer /
  // better quality). The FP8 build and every other video model
  // (Wan 2.2 t2v/i2v, GGUF quants, dev variants) stay available via
  // Settings → System → Model Visibility but off by default so the
  // first-launch picker isn't overwhelming.
  'ltx2_22B_distilled_1_1',
  // LTX-2.5's official split Distilled workflow. The large gated component
  // pack downloads only when selected for the first time.
  'ltx2_25',
  // SCAIL-2 character animation (Animate a character with a control
  // video). Fast = lightx2v distill bundled (6 steps, no CFG, ~13x).
  'scail2_14B',
  'scail2_14B_fast',
  'scail2_14B_recast_fast',
  // MiniMax H3 Base: text, first/last-frame video, and native stereo audio.
  'minimax_h3',
  'minimax_h3_full',
  // MiniMax H3 Ref2VA: ordered image, video, and audio references.
  'minimax_h3_ref2va',
  'minimax_h3_ref2va_full',
  // Audio — Speech
  'kugelaudio_0_open',
  'qwen3_tts_base',
  'qwen3_tts_customvoice',
  'qwen3_tts_voicedesign',
  // Audio — Music
  'ace_step_v1_5_turbo_lm_4b',
  'ace_step_v1_5_xl',
  'ace_step_v1_5_xl_turbo_lm_4b',
  'ace_step_v1_5_xl_sft',
  'ace_step_v1_5_xl_sft_lm_4b',
  'minimax_music3',
  // Audio — SFX
  'mmaudio_v2',
  'mmaudio_nsfw',
  // Avatar
  'animate',
])

/* Version of the curated defaults list above. enabledModels is a stored
 * whitelist, so existing installs never re-read DEFAULT_ENABLED_MODELS —
 * without this, entries added to the curated list in an update stay
 * invisible for everyone who ever opened the app before. Bump the
 * version when adding entries and list them under that version below:
 * they get merged into existing installs' whitelists exactly ONCE, so
 * a user who then disables them stays disabled forever. (This is
 * deliberately narrower than auto-enabling every unknown model — only
 * the curated list's own additions are pushed.) */
const DEFAULTS_VERSION = 10
const DEFAULTS_ADDED_IN: Record<number, string[]> = {
  // v1.2.0: the ACE-Step XL SFT pair; LM_4B becomes the music default.
  2: ['ace_step_v1_5_xl_sft', 'ace_step_v1_5_xl_sft_lm_4b'],
  // v1.3.0: SCAIL-2 character animation, base + lightx2v-distilled Fast.
  3: ['scail2_14B', 'scail2_14B_fast'],
  // Dedicated Recast recipe: native replacement + official I2V LightX point.
  4: ['scail2_14B_recast_fast'],
  // Krea 2 image generation + identity-preserving image editing.
  5: ['krea2_raw', 'krea2_turbo', 'krea2_raw_edit', 'krea2_turbo_edit'],
  // MiniMax H3 Base native audio-video generation.
  6: ['minimax_h3'],
  // MiniMax H3 Base Omni Reference (Ref2VA).
  7: ['minimax_h3_ref2va'],
  // Full 33B H3 variants alongside the recommended Pruned 20B entries.
  8: ['minimax_h3_full', 'minimax_h3_ref2va_full'],
  // LTX-2.5 official Distilled T2V/I2V with synchronized native audio.
  9: ['ltx2_25'],
  // MiniMax-Music3 long-form stereo song generation.
  10: ['minimax_music3'],
}
const DEFAULTS_VERSION_KEY = 'maestro_defaults_version'

/* The music default changed in v1.2.0 (Turbo LM_4B -> SFT LM_4B).
 * A saved selection equal to the OLD default means the user was riding
 * the default rather than expressing a preference — follow them to the
 * new one, once, at the same version transition. Users who picked any
 * other model keep their choice. */
const OLD_MUSIC_DEFAULT = 'ace_step_v1_5_xl_turbo_lm_4b'
const NEW_MUSIC_DEFAULT = 'ace_step_v1_5_xl_sft_lm_4b'

const ENABLED_MODELS_KEY = 'maestro_enabled_models'
let _initializedMatureModels = new Set<string>()
let _modelVisibilityHydrated = false
let _modelVisibilityDefaultsVersion = 1
let _modelVisibilitySaveTask: Promise<void> = Promise.resolve()

function _saveEnabledModels(models: Set<string>) {
  try {
    localStorage.setItem(ENABLED_MODELS_KEY, JSON.stringify([...models]))
  } catch { /* quota exceeded */ }
  const payload = {
    enabled_models: [...models],
    initialized_mature_models: [..._initializedMatureModels],
    defaults_version: _modelVisibilityDefaultsVersion,
  }
  _modelVisibilitySaveTask = _modelVisibilitySaveTask
    .catch(() => { /* a later save should still run */ })
    .then(async () => {
      try {
        await api.updateModelVisibility(payload)
      } catch (error) {
        console.warn('Failed to persist model visibility:', error)
      }
    })
}

function _loadEnabledModels(): Set<string> | null {
  try {
    const raw = localStorage.getItem(ENABLED_MODELS_KEY)
    if (raw) return new Set(JSON.parse(raw))
  } catch { /* ignore */ }
  return null
}

function _markMatureModelsInitialized(
  models: ModelDef[],
  modelTypes?: Iterable<string>,
) {
  const requested = modelTypes ? new Set(modelTypes) : null
  for (const model of models) {
    if (
      model.nsfw_only
      && (requested == null || requested.has(model.model_type))
    ) {
      _initializedMatureModels.add(model.model_type)
    }
  }
}

function _enableUninitializedMatureModels(
  models: ModelDef[],
  enabledModels: Set<string>,
): Set<string> | null {
  const next = new Set(enabledModels)
  let changed = false
  for (const model of models) {
    if (
      model.nsfw_only
      && !_initializedMatureModels.has(model.model_type)
    ) {
      _initializedMatureModels.add(model.model_type)
      next.add(model.model_type)
      changed = true
    }
  }
  return changed ? next : null
}

// Default model_type per generation mode
const modeDefaultModel: Record<GenerationMode, string> = {
  image: 'flux2_klein_9b',
  video: 'ltx2_22B_distilled_1_1',
  audio: 'kugelaudio_0_open',
  avatar: '',  // will fallback to first available
  tools: '',   // Tools is non-generative post-processing — owns no model
}

export function getFamilyMode(familyId: string): GenerationMode {
  return familyModeMap[familyId] || 'video'
}

/** Get the effective generation mode for a specific model (respects per-model overrides) */
export function getModelMode(modelType: string, familyId: string): GenerationMode {
  if (avatarModelTypes.has(modelType)) return 'avatar'
  if (familyId === 'longcat') return 'avatar'
  return getFamilyMode(familyId)
}

/** Director models whose image/audio conditioning strengths are fixed at 1.0. */
export function directorModelUsesFixedMediaStrength(
  modelType: string | undefined,
  architecture?: string | null,
): boolean {
  return [modelType, architecture].some(value => {
    const normalized = String(value || '').toLowerCase()
    return normalized.startsWith('minimax_h3') || normalized.startsWith('ltx2_25')
  })
}

export function getFamiliesForMode(mode: GenerationMode, allFamilies: ModelFamily[], editSubMode?: string, audioSubMode?: string): ModelFamily[] {
  if (mode === 'avatar') {
    // Recast and Repaint run on SCAIL-2, which lives under the Wan 2.1
    // family. The remaining edit sub-modes use LTX models.
    if (editSubMode === 'recast' || editSubMode === 'restyle') {
      return allFamilies.filter(f => f.id === 'wan')
    }
    return allFamilies.filter(f => f.id === 'ltx2' || f.id === 'ltxv')
  }
  if (mode === 'audio') {
    // Filter to the active audio sub-mode family
    if (audioSubMode === 'speech') return audioSubFamilies.filter(f => f.id === 'tts_speech')
    if (audioSubMode === 'music') return audioSubFamilies.filter(f => f.id === 'tts_music')
    if (audioSubMode === 'sfx') return audioSubFamilies.filter(f => f.id === 'tts_sfx')
    if (audioSubMode === 'mixer') return []  // Mixer has no model selector
    return audioSubFamilies
  }
  return allFamilies.filter(f => getFamilyMode(f.id) === mode)
}

/** Get models for a family ID, optionally filtered by generation mode */
export function getModelsForFamily(familyId: string, allModels: ModelDef[], mode?: GenerationMode, editSubMode?: string): ModelDef[] {
  if (familyId === 'tts_speech') {
    return allModels.filter(m => m.family === 'tts' && !isMusicModelType(m.model_type) && !sfxModelTypes.has(m.model_type))
  }
  if (familyId === 'tts_music') {
    return allModels.filter(m => m.family === 'tts' && isMusicModelType(m.model_type))
  }
  if (familyId === 'tts_sfx') {
    return allModels.filter(m => m.family === 'tts' && sfxModelTypes.has(m.model_type))
  }
  const familyModels = allModels.filter(m => m.family === familyId)
  // When mode is specified and the family spans multiple modes, filter to matching models
  if (mode === 'avatar') {
    // Recast exposes its dedicated native-replacement Fast recipe plus HQ.
    if (editSubMode === 'recast') {
      return familyModels.filter(m =>
        m.model_type === 'scail2_14B_recast_fast'
        || m.model_type === 'scail2_14B'
      )
    }
    // Repaint intentionally mirrors Studio Video/Frames SCAIL Animate:
    // the edited first frame is the primary image and the source video
    // supplies motion/camera movement.
    if (editSubMode === 'restyle') {
      return familyModels.filter(m =>
        m.model_type === 'scail2_14B_fast'
        || m.model_type === 'scail2_14B'
      )
    }
    return familyModels.filter(m => !avatarModelTypes.has(m.model_type) && !videoEditModelTypes.has(m.model_type))
  }
  if (mode === 'video') {
    // For video mode: exclude models that are classified as avatar or video edit
    return familyModels.filter(m => !avatarModelTypes.has(m.model_type) && !videoEditModelTypes.has(m.model_type))
  }
  return familyModels
}

/** Get the display family ID for a model (handles audio sub-families) */
export function getDisplayFamily(model: ModelDef): string {
  if (model.family === 'tts') {
    if (sfxModelTypes.has(model.model_type)) return 'tts_sfx'
    if (isMusicModelType(model.model_type)) return 'tts_music'
    return 'tts_speech'
  }
  return model.family
}

// Transient: the LTX model selected before entering either SCAIL-2 edit
// workflow, so leaving Recast/Repaint restores the user's prior edit model.
let _preScail2AvatarModel = ''

const DEFAULT_RECAST_MAPPING: RecastCharacterMapping = {
  id: 'recast-a',
  target: 'person',
  refFile: null,
  refPath: '',
  refUrl: '',
  additionalRefs: [],
  referenceAlignedToSource: false,
}

function getDefaultModelForMode(mode: GenerationMode, families: ModelFamily[], models: ModelDef[]): string {
  // Try the preferred default first
  const preferred = modeDefaultModel[mode]
  if (preferred && models.some(m => m.model_type === preferred)) {
    return preferred
  }
  // Fallback: first model in first family of this mode
  const modeFamilies = getFamiliesForMode(mode, families)
  if (modeFamilies.length > 0) {
    const firstModel = getModelsForFamily(modeFamilies[0].id, models, mode)[0]
    if (firstModel) return firstModel.model_type
  }
  return ''
}

interface AppState {
  // Generation mode (top-level: image/video/audio/avatar)
  generationMode: GenerationMode
  setGenerationMode: (mode: GenerationMode) => void
  editSubMode: import('../types').EditSubMode
  setEditSubMode: (mode: import('../types').EditSubMode) => void
  // Edit mode state (persists across sub-mode switches)
  editVideoPath: string
  editVideoUrl: string
  editVideoFile: File | null
  editVideoDuration: number
  editVideoResolution: string  // "WxH" from source video
  editStartTime: number
  editEndTime: number
  editRetakeStrength: number
  /** CFG scale for prompt-driven edit modes. 1.0 = no CFG (the retake
   *  pipeline's legacy default — prompt barely influences the output).
   *  3.0-5.0 = strong prompt guidance (required for inpaint to actually
   *  replace content with prompt-specific pixels). */
  editPromptStrength: number
  /** LoRA strength for Edit Anything mode. 1.0 is the recommended start
   *  per the LoRA card; bump to 1.2 if the edit is too weak; lower below
   *  1.0 if the edit distorts unrelated content. */
  editAnythingLoraStrength: number
  /** Optional boundary-anchor images for Edit Anything. When set, the
   *  retake pipeline pins frame 0 / last frame of the edit range to these
   *  images instead of auto-extracting them from the source clip. Empty
   *  slots fall back to source frames — so if only the end anchor is set,
   *  the model morphs from source's actual start frame into the user's
   *  edited end frame across the range (the "Ironman suit forms over the
   *  man" effect). */
  editAnythingStartAnchor: string | null
  editAnythingEndAnchor: string | null
  /** SCAIL-2 Repaint edited first frame (uploaded or returned from Image mode). */
  editRepaintFrameFile: File | null
  editRepaintFramePath: string
  editRepaintFrameUrl: string
  /** Optional source-video → edited-frame semantic correspondences. */
  editRepaintMappings: RepaintRegionMapping[]
  /** Spatial quality profile shared with Recast's SCAIL-2 canvas logic. */
  editRepaintResolutionProfile: ScailResolutionProfile
  setEditRepaintFrame: (file: File | null, path: string, url: string) => void
  setEditRepaintMappings: (mappings: RepaintRegionMapping[]) => void
  /** Recast (SCAIL-2 Replace): who to swap out, as a SAM3 keyword. */
  editRecastTarget: string
  /** Number of matching people to track and replace (SCAIL-2 supports 1-5). */
  editRecastPersonCount: number
  /** Recast reference character image (uploaded path + preview URL). */
  editRecastRefFile: File | null
  editRecastRefPath: string
  editRecastRefUrl: string
  /** Explicit source-person → replacement mappings in stable SCAIL color order. */
  editRecastMappings: RecastCharacterMapping[]
  setEditRecastMappings: (mappings: RecastCharacterMapping[]) => void
  /** True when the reference preserves the selected source frame's layout. */
  editRecastRefAligned: boolean
  /** Remove unrelated reference scenery before SCAIL-2 encodes identity. */
  editRecastIsolateReference: boolean
  /** Derive a tighter same-character identity view when none is supplied. */
  editRecastAutoFaceDetail: boolean
  /** Rewrite and append Maestro's Recast identity/scene prompt guidance. */
  editRecastEnhancePrompt: boolean
  /** Strict source-pixel composite outside the tracked Recast target. */
  editRecastProtectBystanders: boolean
  /** Native SCAIL-2 color mapping for other visible identities. */
  editRecastPreserveBystanders: boolean
  /** Apply the official SCAIL-2 replacement Relighting LoRA. */
  editRecastUseRelighting: boolean
  /** Spatial quality profile, independent from the selected SCAIL-2 model. */
  editRecastResolutionProfile: ScailResolutionProfile
  setEditRecastRef: (file: File | null, path: string, url: string, aligned?: boolean) => void
  /** Round-trip marker for the "Edit Anchor in Image Mode" workflow.
   *  Populated when the user clicks "Edit Start" or "Edit End" on a
   *  boundary anchor slot. A banner at the top of the sidebar lets them
   *  apply the latest Image-mode output to that single anchor, then
   *  return to Edit Anything. Each anchor is its own independent
   *  round-trip — start and end can't both be in flight at once, but
   *  the user does them sequentially. */
  editReturnTarget: {
    /** Which anchor slot we're populating on return. */
    anchor: 'start' | 'end' | 'recast' | 'repaint'
    /** The pre-extracted source frame at the corresponding trim handle.
     *  This is the frame the user is editing in Image mode; if they
     *  cancel without applying, no anchor is set and the model falls
     *  back to extracting this same frame at generation time. */
    framePath: string
    /** The clip the user came from, so we can re-link them on return. */
    clipPath: string
    startTime: number
    endTime: number
    /** User's image-mode reference images / type before we hijacked the
     *  slot for the round-trip — restored on return so we don't nuke
     *  their existing image-mode workflow state. */
    savedImageRefs: File[]
    savedImageRefType: string
  } | null
  setEditAnythingStartAnchor: (path: string | null) => void
  setEditAnythingEndAnchor: (path: string | null) => void
  /** Extract one boundary frame from the source clip and switch the
   *  sidebar to Studio Image mode (using the proper setGenerationMode
   *  so the model + LoRA + image-mode params all swap correctly) with
   *  that frame loaded as image_start. */
  sendFrameToImageMode: (which: 'start' | 'end' | 'recast' | 'repaint') => Promise<void>
  /** Apply the latest Image-mode output to the requested anchor/reference,
   *  then return to Edit Anything or Recast. */
  applyOutputAsAnchor: () => Promise<void>
  /** Skip applying — return to Edit Anything with the anchor unset
   *  (model will fall back to source-extracted frame at generation time,
   *  giving the morph-from-source effect when only the OTHER anchor is
   *  set). */
  skipAnchorPhase: () => void
  /** Cancel the round-trip and return to Edit Anything. Same effect as
   *  skipAnchorPhase, but exposed separately for UI clarity. */
  cancelAnchorReturn: () => void
  editRetakeEngine: 'native' | 'legacy'
  editRegenerateAudio: boolean
  editSamTarget: string  // separate SAM segmentation target (noun phrase)
  editInvertMask: boolean  // invert SAM mask (select everything EXCEPT the target)
  editMasksPath: string | null  // cached SAM mask for inpaint
  editMaskPreview: string | null
  editDetectedTarget: string
  // Continue video state
  continueVideo: File | null
  continueVideoPath: string
  continueVideoUrl: string
  continueVideoDuration: number
  setContinueVideo: (file: File, path: string, url: string, duration: number) => void
  clearContinueVideo: () => void
  // Per-sub-mode working sets (Studio Video). Keyed by image_mode
  // (0 Frames / 2 Multi-Shot / 3 Extend / 4 Blend) — each sub-mode keeps
  // its own prompt, input tiles, and settings. See setParam('image_mode').
  videoSubModeStash: Partial<Record<number, VideoSubModeStash>>
  // Blend state
  blendClipA: File | null
  blendClipAPath: string
  blendClipAUrl: string
  blendClipADuration: number
  blendClipB: File | null
  blendClipBPath: string
  blendClipBUrl: string
  blendClipBDuration: number
  blendTransitionSec: number
  blendStrengthA: number
  blendStrengthB: number
  /** Seconds of Clip A's overlap tail used as video_source (motion prefix) for VE mode.
   *  0 = pure SE (single start-frame anchor, no motion continuity from A).
   *  1-2 = model extrapolates A's motion through the blend. */
  blendMotionPrefixSec: number
  /** Seconds of Clip B's overlap head used as video_end (motion suffix) —
   *  symmetric counterpart to motion prefix. 0 = single still anchor at
   *  blend end. 1-2 = model lands at B with real jogger stride/speed. */
  blendMotionSuffixSec: number
  /** input_video_strength for the VE anchors (video_source + image_end).
   *  1.0 = hard-lock both anchors → model averages between them (crossfade).
   *  0.5-0.8 = weaker anchors, model invents motion in between. */
  blendAnchorStrength: number
  setBlendClipA: (file: File, path: string, url: string, duration: number) => void
  setBlendClipB: (file: File, path: string, url: string, duration: number) => void
  clearBlendClipA: () => void
  clearBlendClipB: () => void
  setBlendTransitionSec: (sec: number) => void
  setBlendStrengthA: (v: number) => void
  setBlendStrengthB: (v: number) => void
  setBlendMotionPrefixSec: (v: number) => void
  setBlendMotionSuffixSec: (v: number) => void
  setBlendAnchorStrength: (v: number) => void
  blendMode: 'insert' | 'overlap'
  blendOverlapSec: number
  setBlendMode: (mode: 'insert' | 'overlap') => void
  setBlendOverlapSec: (sec: number) => void
  // Outpaint state
  // Padding kept in pixels (server contract: pad_top/bottom/left/right).
  // The new OutpaintCanvas computes these from canvas aspect + video position
  // on submit, but the store still surfaces the raw values so legacy callers
  // and metadata sidecars stay compatible.
  outpaintPadding: { top: number; bottom: number; left: number; right: number }
  setOutpaintPadding: (padding: { top: number; bottom: number; left: number; right: number }) => void
  outpaintResolutionPreset: 'auto' | '480p' | '540p' | '720p' | '1080p'
  setOutpaintResolutionPreset: (preset: 'auto' | '480p' | '540p' | '720p' | '1080p') => void
  // Canvas aspect ratio for the outpaint composer. 'source' means keep the
  // source clip's native aspect (no canvas extension — only useful when the
  // user wants to outpaint a single side via drag).
  outpaintAspect: '16:9' | '9:16' | '1:1' | '4:3' | '3:4' | 'source'
  setOutpaintAspect: (a: '16:9' | '9:16' | '1:1' | '4:3' | '3:4' | 'source') => void
  // Video frame position+size inside the canvas, normalized to canvas
  // dimensions (0–1). Default = centered, fully fit (no crop). User drags
  // to reposition; resize handles scale the source within the canvas.
  outpaintVideoBox: { x: number; y: number; w: number; h: number }
  setOutpaintVideoBox: (box: { x: number; y: number; w: number; h: number }) => void
  // Film-strip trim times (seconds). When end > start, server pre-trims
  // the source via ffmpeg before outpainting.
  outpaintTrimStart: number
  outpaintTrimEnd: number
  setOutpaintTrimStart: (t: number) => void
  setOutpaintTrimEnd: (t: number) => void
  outpaintSourcePreservation: number
  setOutpaintSourcePreservation: (v: number) => void
  outpaintLoraStrength: number
  setOutpaintLoraStrength: (v: number) => void
  // Official LTX-2.3 binary-mask conditioning plus multiscale source blend.
  // Enabled by default; false keeps the legacy black-sentinel path for A/B.
  outpaintMaskPreserving: boolean
  setOutpaintMaskPreserving: (v: boolean) => void
  outpaintPreserveSourceAudio: boolean
  setOutpaintPreserveSourceAudio: (v: boolean) => void
  // Lock source pixels: composite original source clip back into the source
  // rectangle of the outpainted output (post-process ffmpeg overlay).
  // Default OFF — the model's regenerated source area actually preserves
  // lip detail well, and a hard overlay creates a visible rectangle seam.
  // Kept for opt-in use cases that need pixel-perfect source area.
  outpaintLockSourcePixels: boolean
  setOutpaintLockSourcePixels: (v: boolean) => void
  // Trim sliding-window smear: cut the per-window-overlap frames at the
  // window 1→2 boundary in the output, where the IC-LoRA's prefix
  // conditioning produces a constant ~9-frame lag for the rest of the
  // clip. Default ON — fixes lip sync on multi-window outpaint.
  outpaintTrimSmear: boolean
  setOutpaintTrimSmear: (v: boolean) => void
  // Sliding-window controls for long-clip outpainting (auto-engages when
  // total_frames > windowSize). 0 = use model default (LTX-2: 241 frames).
  outpaintWindowSize: number
  setOutpaintWindowSize: (v: number) => void
  outpaintWindowOverlap: number
  setOutpaintWindowOverlap: (v: number) => void
  setEditVideoPath: (path: string) => void
  setEditVideo: (file: File | null, path: string, url: string, duration: number, resolution: string) => void
  clearEditVideo: () => void
  audioSubMode: import('../types').AudioSubMode
  setAudioSubMode: (mode: import('../types').AudioSubMode) => void
  // Music mode (ACE-Step): describe + LLM writes, or type Style/Lyrics directly.
  musicDescription: string
  setMusicDescription: (s: string) => void
  musicInstrumental: boolean
  setMusicInstrumental: (b: boolean) => void
  selectedModelPerAudioSubMode: Partial<Record<import('../types').AudioSubMode, string>>
  selectedModelPerMode: Partial<Record<GenerationMode, string>>
  savedLoraPerMode: Partial<Record<GenerationMode, { activated_loras: string[]; loras_multipliers: string; loraWeights: Record<string, number[]>; availableLoras: string[] }>>
  savedParamsPerMode: Partial<Record<GenerationMode, SavedModeParams>>
  savedPromptPerMode: Partial<Record<string, string>>
  /** Snapshot of lora_id → filename loaded from localStorage at boot.
   *  Used by `refreshLoraIdMap` reconciliation to rewrite filenames that
   *  changed since save (LoRA version updates). Internal-only; not part
   *  of the persisted runtime state. */
  _loraFilenameSnapshotAtLoad?: Record<string, string>

  // Generation params
  params: GenerateParams
  setParam: <K extends keyof GenerateParams>(key: K, value: GenerateParams[K]) => void
  setParams: (partial: Partial<GenerateParams>) => void

  // UI state
  settingsOpen: boolean
  toggleSettings: () => void
  setSettingsOpen: (open: boolean) => void
  sidebarOpen: boolean
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void

  // Theme — see lib/theme.ts. Two-dimensional: a dark/light/auto mode
  // plus a theme family (each family has a dark and a light variant).
  // Persisted to localStorage; an inline script in index.html applies
  // the resolved theme to <html> BEFORE React mounts to avoid a flash
  // of the default theme.
  themePrefs: ThemePrefs
  setThemeMode: (mode: ThemeMode) => void
  setThemeFamily: (family: FamilyId) => void

  // Retake Dialog
  retakeDialogOpen: boolean
  retakeSourceFile: string | null
  openRetakeDialog: (filename: string) => void
  closeRetakeDialog: () => void

  // CivitAI LoRA Browser
  // Director Pipeline Dashboard
  dashboardOpen: boolean
  dashboardPipelineList: PipelineListItem[]
  dashboardSelectedPipeline: SavedPipelineState | null
  dashboardLoading: boolean
  setDashboardOpen: (open: boolean) => void
  loadPipelineList: () => Promise<void>
  loadSavedPipeline: (pid: string) => Promise<void>
  tagClip: (pid: string, clipIndex: number, tag: string | null) => Promise<void>
  startPipelineRepair: (pid: string) => Promise<PipelineRepairState>
  cancelPipelineRepair: (pid: string) => Promise<PipelineRepairState>
  pollPipelineRepair: (pid: string, operationId: string) => void
  rerunClipImage: (pid: string, clipIndex: number, prompt?: string) => Promise<unknown>
  rerunClipVideo: (pid: string, clipIndex: number, prompt?: string) => Promise<unknown>
  rejoinPipelineClips: (pid: string) => Promise<unknown>
  resumePipeline: (pid: string) => Promise<void>
  deletePipeline: (pid: string) => Promise<void>
  loadDirectorFromPipeline: (pid: string) => Promise<void>

  // Recipes (one-click Studio presets)
  recipesOpen: boolean
  setRecipesOpen: (open: boolean) => void
  recipes: import('../api/client').RecipeCard[]
  recipesLoading: boolean
  loadRecipes: () => Promise<void>
  applyRecipe: (id: string) => Promise<{ missing: import('../api/client').RecipeLora[] }>
  saveRecipeFromOutput: (outputName: string, name: string, description: string, nsfw: boolean) => Promise<void>
  deleteRecipe: (id: string) => Promise<void>
  downloadRecipeLora: (lora: import('../api/client').RecipeLora, modelType: string) => Promise<void>

  loraBrowserOpen: boolean
  loraBrowserArch: string | null
  loraBrowserDefaultDir: string | null
  setLoraBrowserOpen: (open: boolean, arch?: string) => void
  setLoraBrowserDefaultDir: (dir: string | null) => void
  civitSearchResults: CivitAIModel[]
  civitSearchCursor: string | null
  civitSearchLoading: boolean
  civitSearchError: string | null
  civitSelectedModel: CivitAIModel | null
  civitDownloads: CivitAIDownload[]
  searchCivitAI: (params: Record<string, unknown>, append?: boolean) => Promise<void>
  selectCivitAIModel: (modelId: number) => Promise<void>
  clearCivitSelection: () => void
  startCivitAIDownload: (params: Record<string, unknown>) => Promise<void>
  pollCivitAIDownloads: () => void

  // Models & families (from API)
  families: ModelFamily[]
  models: ModelDef[]
  loadModels: () => Promise<void>
  modelsLoaded: boolean

  // Model visibility (favorites)
  enabledModels: Set<string>
  toggleModelEnabled: (modelType: string) => void
  resetEnabledModels: () => void
  setAllModelsEnabled: (enabled: boolean) => void
  /** Bulk-toggle a list of models (family-level enable/disable, issue #14). */
  setModelsEnabled: (modelTypes: string[], enabled: boolean) => void
  // ModelSelector "+N more" hint → open Settings and expand Enabled Models.
  modelVisibilityFocus: GenerationMode | null
  openModelVisibility: (mode: GenerationMode) => void
  clearModelVisibilityFocus: () => void

  // Resolution helpers
  resolutionPreset: ResolutionPreset
  setResolutionPreset: (preset: ResolutionPreset) => void
  aspectRatio: AspectRatio
  setAspectRatio: (ratio: AspectRatio) => void

  // Duration
  durationSeconds: number
  setDurationSeconds: (s: number) => void

  // Sliding window
  slidingWindowSeconds: number
  setSlidingWindowSeconds: (s: number) => void
  slidingWindowOverlap: number
  setSlidingWindowOverlap: (frames: number) => void
  slidingWindowLocked: boolean
  setSlidingWindowLocked: (locked: boolean) => void
  /** Durable H3 pass lengths keyed by exact model type and resolution. */
  h3WindowOverrides: Record<string, number>
  saveH3WindowOverride: (modelType: string, resolution: string, frames: number) => void
  clearH3WindowOverride: (modelType: string, resolution: string) => void

  // Real frame rate of the uploaded guide/control video (probed server-side
  // at upload). Used by force_fps="control" models (SCAIL-2 class) to
  // convert durationSeconds to frames at the rate the output will actually
  // play at, instead of the model's nominal fps.
  guideVideoFps: number | null
  setGuideVideoFps: (fps: number | null) => void

  // Output count
  outputCount: number
  setOutputCount: (n: number) => void

  // Image uploads
  startImage: File | null
  endImage: File | null
  setStartImage: (f: File | null) => void
  setEndImage: (f: File | null) => void

  // Image references (for models with image_ref_choices)
  imageRefs: File[]
  imageRefType: string
  removeBackgroundRefs: boolean
  addImageRef: (file: File) => void
  removeImageRef: (index: number) => void
  reorderImageRefs: (from: number, to: number) => void
  setImageRefType: (type: string) => void
  setRemoveBackgroundRefs: (v: boolean) => void

  // Post-processing (shared for Studio mode)
  spatialUpsampling: string
  setSpatialUpsampling: (v: string) => void
  filmGrainIntensity: number
  setFilmGrainIntensity: (v: number) => void
  filmGrainSaturation: number
  setFilmGrainSaturation: (v: number) => void

  // Voice clone postprocessing (SeedVC). Replaces 1 or 2 voices in
  // a generated video's audio with user-supplied reference voice(s).
  // Applied after generation as a postprocessing step. See
  // app/postprocessing/voice_clone.py for backend logic.
  voiceCloneEnabled: boolean
  setVoiceCloneEnabled: (v: boolean) => void
  voiceCloneMode: 'single' | 'two'
  setVoiceCloneMode: (v: 'single' | 'two') => void
  // Up to 2 reference voices. Each entry tracks the uploaded filename
  // (display) + the server-side path the backend uses.
  voiceCloneRefs: { filename: string; path: string }[]
  setVoiceCloneRef: (index: number, ref: { filename: string; path: string } | null) => void

  // ── Tools area (standalone post-processing on an existing clip) ──────
  // Apply FlashVSR upscale or SeedVC revoice to any gallery output or an
  // uploaded clip, independent of a generation. See ToolsPanel.tsx + the
  // /api/v1/tools/* endpoints.
  toolsTool: 'upscale' | 'revoice'
  setToolsTool: (t: 'upscale' | 'revoice') => void
  /** Gallery filename (resolved against the workspace) OR an absolute upload path. */
  toolsSourcePath: string | null
  toolsSourceName: string | null
  toolsSourceUrl: string | null
  setToolsSource: (src: { path: string; name: string; url: string | null } | null) => void
  toolsUpscaleMethod: string
  setToolsUpscaleMethod: (m: string) => void
  toolsRevoiceMode: 'single' | 'two'
  setToolsRevoiceMode: (m: 'single' | 'two') => void
  toolsRevoiceRefs: ({ filename: string; path: string } | null)[]
  setToolsRevoiceRef: (index: number, ref: { filename: string; path: string } | null) => void
  runTool: () => Promise<void>
  /** Gallery one-click: upscale a specific clip now, with the configured method. */
  quickUpscaleClip: (name: string, url: string | null) => Promise<void>
  /** Gallery one-click: load a clip into the Tools panel for a tool that needs
   *  setup before running (e.g. revoice needs voice references), and switch to it. */
  sendClipToTools: (name: string, url: string | null, tool: 'upscale' | 'revoice') => void

  // Director-mode post-processing (separate image/video)
  directorImageSpatialUpsampling: string
  setDirectorImageSpatialUpsampling: (v: string) => void
  directorImageFilmGrainIntensity: number
  setDirectorImageFilmGrainIntensity: (v: number) => void
  directorImageFilmGrainSaturation: number
  setDirectorImageFilmGrainSaturation: (v: number) => void
  directorVideoSpatialUpsampling: string
  setDirectorVideoSpatialUpsampling: (v: string) => void
  directorVideoFilmGrainIntensity: number
  setDirectorVideoFilmGrainIntensity: (v: number) => void
  directorVideoFilmGrainSaturation: number
  setDirectorVideoFilmGrainSaturation: (v: number) => void
  directorVideoSelfRefiner: number
  setDirectorVideoSelfRefiner: (v: number) => void
  directorAudioScale: number
  setDirectorAudioScale: (v: number) => void

  // Audio guide (pre-filled by Director or manual upload)
  audioGuideFilename: string | null
  setAudioGuideFilename: (name: string | null) => void
  audioGuide2Filename: string | null
  setAudioGuide2Filename: (name: string | null) => void
  ttsSpeakerName1: string
  ttsSpeakerName2: string
  ttsSpeakerNamesManual: boolean
  setTtsSpeakerName1: (name: string) => void
  setTtsSpeakerName2: (name: string) => void
  _autoParseSpkeakerNames: (text: string, force?: boolean) => void
  // Dynamic multi-speaker (1-6 voices)
  ttsVoiceCount: number  // 0=text only, 1-6=voice clone count
  ttsVoices: { name: string; filename: string | null; path: string | null }[]
  setTtsVoiceCount: (count: number) => void
  setTtsVoiceName: (index: number, name: string) => void
  setTtsVoiceFile: (index: number, filename: string | null, path: string | null) => void
  addTtsVoice: () => void
  removeTtsVoice: (index: number) => void

  // Multi-clip state
  clips: MultiClip[]
  singlePromptMode: boolean
  setClipPrompt: (index: number, prompt: string) => void
  setClipStartImage: (index: number, file: File | null) => void
  setSinglePromptMode: (v: boolean) => void
  syncClipCount: () => void

  // Generation state (queue)
  jobs: GenerationJob[]
  isGenerating: boolean
  startGeneration: () => Promise<void>
  stopGeneration: (jobId?: string) => void
  dismissJob: (jobId: string) => void
  reconnectJobs: () => Promise<void>

  // LoRA state
  availableLoras: string[]
  lorasLoading: boolean
  loraWeights: Record<string, number[]>
  /** Map of LoRA filename → stable lora_id (e.g. `civitai:12345` for a
   *  CivitAI-sourced LoRA, `local:foo.safetensors` for hand-installed).
   *  Populated from /api/v1/loras/installed at boot and refreshed when
   *  LoRAs are added/removed. Used by the localStorage persistence layer
   *  to write update-resilient keys. */
  loraIdByFilename: Record<string, string>
  /** Reverse: lora_id → current filename. Used by reconciliation to
   *  detect when a saved filename has been renamed by a LoRA update. */
  filenameByLoraId: Record<string, string>
  /** Refresh `loraIdByFilename` / `filenameByLoraId` from the backend.
   *  Triggers reconciliation of savedLoraPerMode against the fresh map. */
  refreshLoraIdMap: () => Promise<void>
  loadLoras: (modelType: string) => Promise<void>
  toggleLora: (filename: string) => void
  /** Ensure the LTX-2.3 transition LoRA is downloaded and activated for
   *  blend mode. Called when blend mode is opened. Idempotent: no-op if
   *  the LoRA is already installed and activated. */
  ensureTransitionLoraForBlend: () => Promise<void>
  /** Ensure the Alissonerdx Edit Anything LoRA is downloaded. Called when
   *  the Edit Anything sub-mode is opened. Idempotent — no-op if already
   *  installed. Unlike the transition LoRA, this one is activated
   *  server-side by the /api/v1/edit-anything endpoint, not client-side,
   *  so the user's global LoRA list isn't touched. */
  ensureEditAnythingLora: () => Promise<void>
  setLoraWeight: (filename: string, phaseIndex: number, value: number) => void

  // Presets
  presets: import('../api/client').GenerationPreset[]
  presetsLoading: boolean
  loadPresets: () => Promise<void>
  savePreset: (name: string) => Promise<void>
  loadPreset: (preset: import('../api/client').GenerationPreset) => void
  deletePreset: (id: string) => Promise<void>

  // Model options
  modelOptions: ModelOptions | null
  modelOptionsLoading: boolean
  loadModelOptions: (modelType: string) => Promise<void>

  // System config
  systemConfig: SystemConfig | null
  systemConfigLoading: boolean
  loadSystemConfig: () => Promise<void>
  updateSystemConfig: (partial: Partial<SystemConfig>) => Promise<void>

  // Hardware detect — populated lazily when Settings → System opens.
  // Shared between AutoPerformanceCard (the readout) and the rest of
  // the System panel (e.g. the VRAM coefficient subtext that needs to
  // know the user's actual VRAM size, not a hardcoded 24GB).
  systemDetect: SystemDetectResponse | null
  loadSystemDetect: () => Promise<void>
  systemStats: SystemStats | null
  loadSystemStats: () => Promise<void>

  // Settings tab
  settingsTab: SettingsTab
  setSettingsTab: (tab: SettingsTab) => void

  // Select model (triggers side effects)
  selectModel: (modelType: string) => void

  // Workspaces
  workspaces: Array<{ name: string; path: string; file_count?: number }>
  activeWorkspace: string
  /** Gallery is showing the virtual "Uploads" view (browse-only — the
   *  server-side active workspace, and where generations save, is
   *  untouched). Entered via switchWorkspace('__uploads__'). */
  browsingUploads: boolean
  loadWorkspaces: () => Promise<void>
  switchWorkspace: (name: string) => Promise<void>
  createWorkspace: (name: string) => Promise<void>
  deleteWorkspace: (name: string) => Promise<void>

  // Storage Manager overlay
  storageDashboardOpen: boolean
  setStorageDashboardOpen: (open: boolean) => void

  // LoRA picker sort order — store-backed (not per-component state) so
  // simultaneously mounted pickers (e.g. Director's Image + Video
  // accordions) stay in sync; persisted to localStorage.
  loraPickerSort: 'name' | 'newest'
  setLoraPickerSort: (sort: 'name' | 'newest') => void

  // Outputs
  outputs: OutputFile[]
  outputsTotal: number
  selectedOutput: number
  setSelectedOutput: (i: number) => void
  mediaFilter: MediaFilter
  outputSearchQuery: string
  setMediaFilter: (f: MediaFilter) => void
  setOutputSearchQuery: (q: string) => void
  filteredOutputs: () => OutputFile[]
  outputsLoading: boolean
  loadOutputs: () => Promise<void>
  loadMoreOutputs: () => Promise<void>
  refreshOutputs: () => Promise<void>
  toggleFavorite: (name: string) => Promise<void>

  // Output metadata (lazy-loaded for selected output)
  selectedOutputMeta: OutputMetadata | null
  metadataLoading: boolean
  loadOutputMetadata: (name: string) => Promise<void>
  loadSettingsFromOutput: () => Promise<void>
  rerollGeneration: () => Promise<void>
  deleteSelectedOutput: () => Promise<void>
  rejoinClipGroup: (groupId: string) => Promise<void>

  // Services config
  servicesConfig: ServicesConfig | null
  servicesConfigLoading: boolean
  loadServicesConfig: () => Promise<void>
  updateServicesConfig: (partial: Partial<ServicesConfig>) => Promise<void>

  // LLM state
  llmStatus: LlmStatus | null
  llmLoading: boolean
  llmModels: LlmModelOption[]
  loadLlmStatus: () => Promise<void>
  loadLlmModels: () => Promise<void>
  loadLlm: () => Promise<void>
  unloadLlm: () => Promise<void>

  // Prompt enhancement
  isEnhancing: boolean
  promptEnhanceError: string | null
  enhancePrompt: (ttsMode?: string) => Promise<void>
  h3WindowPlan: H3WindowPlan | null
  updateH3WindowPrompt: (index: number, prompt: string) => void
  clearH3WindowPlan: () => void

  // Director (Music Video Director)
  sidebarMode: 'director' | 'studio'
  directorStep: 'upload' | 'analyze' | 'structure' | 'style' | 'plan' | 'review' | 'generate_images' | 'plan_video' | 'review_video'
  directorAudioFile: File | null
  directorAudioPath: string | null
  directorAnalysis: AudioAnalysisResult | null
  directorPlannedClips: PlannedClip[]
  directorEnergyBias: number
  directorClipPlans: ClipPlan[]
  directorSceneDescription: string
  directorLoading: boolean
  /** Sub-status for the current loading phase (e.g. "Loading
   *  transcription model (first use downloads ~300MB)..."). Set by
   *  the analyze polling loop in directorUploadAndAnalyze; read by
   *  the sidebar loading spinner. Falls back to a default like
   *  "Analyzing audio..." in the UI when null. */
  directorLoadingMessage: string | null
  directorError: string | null
  directorReferenceImage: File | null
  directorReferenceImagePath: string | null
  directorCharacterRefs: File[]
  directorCharacterRefPaths: string[]
  directorCharacterRefLabels: string[]
  directorLocationRefs: File[]
  directorLocationRefPaths: string[]
  directorLocationRefLabels: string[]
  directorVoiceRef: File | null
  directorVoiceRefPath: string | null
  directorIdentityGuidanceScale: number
  /** Experimental: bypass the safety check that disables ID-LoRA reference
   *  audio concatenation on the distilled LTX-2.3 pipeline. The base
   *  distilled model produces noise when ref tokens are prepended, but
   *  newer ID-LoRA variants (e.g. AviadDahan CelebVHQ-3K) claim distilled
   *  compatibility — this flag lets users test those LoRAs.
   *
   *  REMOVED 2026-05-26: Per WanGP v11.77 testing, the CelebVHQ ID-LoRA
   *  works on both dev and distilled. The block-on-distilled gate and
   *  this experimental override are both gone. The comment is preserved
   *  for historical context only. */
  setDirectorVoiceRef: (file: File | null) => void
  setDirectorIdentityGuidanceScale: (v: number) => void
  directorClipImages: DirectorClipImage[]
  /** Set or clear an optional user-supplied start image for one manually
   *  reviewed Director scene. */
  directorSetClipImage: (clipIndex: number, file: File | null) => void
  directorImageGenProgress: DirectorImageGenProgress | null
  directorSpeakers: string[]
  directorSpeakerMappings: SpeakerMapping[]
  directorAutoMode: boolean
  directorSeamless: boolean
  directorShotImageGuidance: DirectorShotImageGuidance
  /** Completed LLM stream outputs, kept so the thinking/output boxes stay
   *  in the chat history after each stage finishes instead of vanishing. */
  directorLlmLog: { stage: string; text: string }[]
  directorAppendLlmLog: (stage: string, text: string) => void
  directorSkill: DirectorSkill | null
  directorResolution: ResolutionPreset
  directorAspectRatio: AspectRatio
  /** Director-owned inference-step choices, keyed by video model. Keeping
   *  these separate from Studio prevents one surface from silently changing
   *  the other and lets each Director model retain its own valid recipe. */
  directorVideoInferenceStepsByModel: Record<string, number>
  /** Optional expert override of Director's hardware-safe native-shot cap. */
  directorVideoMaxShotFramesByModel: Record<string, number>
  /** Director-owned H3 Turbo choices, separate from Studio's active mode. */
  directorH3TurboModeByModel: Record<string, boolean>
  /** Director-owned managed H3 Turbo checkpoint choice. */
  directorH3TurboPresetByModel: Record<string, string>
  /** Director-owned experimental H3 Sol Engine choices. */
  directorH3SolModeByModel: Record<string, boolean>
  /** Director-owned H3 First Block Cache choices and tuning. */
  directorH3FirstBlockCacheByModel: Record<string, boolean>
  directorH3FirstBlockCacheMultiplierByModel: Record<string, number>
  directorH3FirstBlockCacheWarmupByModel: Record<string, number>
  setDirectorAutoMode: (v: boolean) => void
  setDirectorSeamless: (v: boolean) => void
  setDirectorShotImageGuidance: (v: DirectorShotImageGuidance) => void
  setDirectorSkill: (skill: DirectorSkill) => void
  setDirectorResolution: (preset: ResolutionPreset) => void
  setDirectorAspectRatio: (ratio: AspectRatio) => void
  setDirectorVideoInferenceSteps: (modelType: string, steps: number | null) => void
  setDirectorVideoMaxShotFrames: (modelType: string, frames: number | null) => void
  setDirectorH3TurboMode: (modelType: string, enabled: boolean) => void
  setDirectorH3TurboPreset: (modelType: string, presetId: string) => void
  setDirectorH3SolMode: (modelType: string, enabled: boolean) => void
  setDirectorH3FirstBlockCache: (modelType: string, enabled: boolean) => void
  setDirectorH3FirstBlockCacheMultiplier: (modelType: string, value: number) => void
  setDirectorH3FirstBlockCacheWarmup: (modelType: string, value: number) => void
  selectDirectorImageModel: (modelType: string) => void
  selectDirectorVideoModel: (modelType: string) => void
  directorSetLora: (mode: 'image' | 'video', activated_loras: string[], loras_multipliers: string, loraWeights: Record<string, number[]>, availableLoras: string[]) => void
  setSidebarMode: (mode: 'director' | 'studio') => void
  directorSetSpeakerMapping: (speakerId: string, name: string, role: SpeakerMapping['role']) => void
  directorInsertSpeakerMention: (speakerId: string) => void
  directorUploadAndAnalyze: (file: File) => Promise<void>
  // Music Video: generate-the-track source + song setup
  directorMusicSource: 'upload' | 'generate' | null
  directorMusicModel: string
  directorSongDescription: string
  directorSongInstrumental: boolean
  directorSongStyle: string
  directorSongLyrics: string
  directorSongDuration: number
  directorTrackGenerating: boolean
  setDirectorMusicSource: (s: 'upload' | 'generate' | null) => void
  setDirectorMusicModel: (modelType: string) => void
  setDirectorSongDescription: (v: string) => void
  setDirectorSongInstrumental: (v: boolean) => void
  setDirectorSongStyle: (v: string) => void
  setDirectorSongLyrics: (v: string) => void
  setDirectorSongDuration: (v: number) => void
  directorWriteSong: () => Promise<void>
  directorGenerateTrack: () => Promise<void>
  directorAnalyzeAndPlan: (audioPath: string, opts?: { transcribe?: boolean; lyricsHint?: string }) => Promise<void>
  directorSetEnergyBias: (bias: number) => Promise<void>
  directorConfirmStructure: () => void
  directorSetSceneDescription: (prompt: string) => void
  directorSetReferenceImage: (file: File | null) => void
  directorAddCharacterRef: (file: File) => void
  directorRemoveCharacterRef: (index: number) => void
  directorSetCharacterRefLabel: (index: number, label: string) => void
  directorReorderCharacterRefs: (from: number, to: number) => void
  directorAddLocationRef: (file: File) => void
  directorRemoveLocationRef: (index: number) => void
  directorSetLocationRefLabel: (index: number, label: string) => void
  directorReorderLocationRefs: (from: number, to: number) => void
  directorPlanPrompts: () => Promise<void>
  directorPlanVideoPrompts: () => Promise<void>
  directorGenerateStartImages: () => Promise<void>
  directorApplyToClips: () => void
  directorGenerate: () => void
  directorReset: () => void
  directorEditClipPlan: (index: number, field: 'video_prompt' | 'image_prompt', value: string) => void
  _uploadDirectorRefs: () => Promise<{ refImagePath: string | null; charPaths: string[]; locPaths: string[] }>

  // Short Film Director
  shortFilmCharacters: ShortFilmCharacter[]
  shortFilmPath: ShortFilmPath | null
  shortFilmTargetDuration: number
  shortFilmNarrative: boolean
  shortFilmSetCharacters: (characters: ShortFilmCharacter[]) => void
  shortFilmSetPath: (path: ShortFilmPath) => void
  shortFilmSetTargetDuration: (duration: number) => void
  shortFilmSetNarrative: (v: boolean) => void
  shortFilmUploadAndAnalyze: (file: File) => Promise<void>
  shortFilmSetPacingBias: (bias: number) => Promise<void>
  shortFilmPlanPrompts: () => Promise<void>
  shortFilmPlanVideoPrompts: () => Promise<void>
  shortFilmPlanFromStory: () => Promise<void>

  // LLM streaming
  llmStreamText: string
  llmStreamDone: boolean

  // Director Pipeline (server-side)
  pipelineId: string | null
  pipelineStatus: import('../api/client').PipelineStatus | null
  pipelinePolling: boolean
  startDirectorPipeline: () => Promise<void>
  continuePipeline: (updates?: { clip_plans?: Array<{ video_prompt: string; image_prompt: string }> }) => Promise<void>
  stopPipeline: () => Promise<void>
  pollPipelineStatus: () => void
}

const defaultParams: GenerateParams = {
  prompt: '',
  model_type: 'ltx2_22B_distilled_1_1',
  resolution: '1280x720',
  video_length: 251,
  num_inference_steps: 8,
  guidance_scale: 1.0,
  seed: -1,
  image_mode: 0,
  negative_prompt: '',
  repeat_generation: 1,
  activated_loras: [],
  loras_multipliers: '',
  skip_steps_cache_type: '',
  skip_steps_multiplier: 0.08,
  skip_steps_start_step_perc: 25,
  settings_version: 2.52,
}

// ── Per-sub-mode working sets (Studio Video) ─────────────────────────
// Frames, Multi-Shot, Extend, and Blend each keep their OWN prompt,
// input tiles, and settings. Switching the ModeToggle stashes the
// outgoing sub-mode's full working set and restores the incoming one —
// so a Frames setup with a dozen injected keyframes survives a
// round-trip through Extend untouched. First visit to a sub-mode keeps
// the generic settings (steps, resolution, ...) but blanks the input
// spec, so Extend starts clean instead of inheriting Frames' inputs.
// In-memory only: after a reload the active sub-mode is restored (via
// savedParamsPerMode) and the others start blank again.
interface VideoSubModeStash {
  params: GenerateParams
  startImage: File | null
  endImage: File | null
  continueVideo: File | null
  continueVideoPath: string
  continueVideoUrl: string
  continueVideoDuration: number
  audioGuideFilename: string | null
  imageRefs: File[]
  imageRefType: string
  removeBackgroundRefs: boolean
  durationSeconds: number
  slidingWindowSeconds: number
  slidingWindowOverlap: number
  clips: MultiClip[]
  singlePromptMode: boolean
}

const captureVideoSubModeStash = (s: AppState): VideoSubModeStash => ({
  params: { ...s.params },
  startImage: s.startImage,
  endImage: s.endImage,
  continueVideo: s.continueVideo,
  continueVideoPath: s.continueVideoPath,
  continueVideoUrl: s.continueVideoUrl,
  continueVideoDuration: s.continueVideoDuration,
  audioGuideFilename: s.audioGuideFilename,
  imageRefs: s.imageRefs,
  imageRefType: s.imageRefType,
  removeBackgroundRefs: s.removeBackgroundRefs,
  durationSeconds: s.durationSeconds,
  slidingWindowSeconds: s.slidingWindowSeconds,
  slidingWindowOverlap: s.slidingWindowOverlap,
  clips: s.clips,
  singlePromptMode: s.singlePromptMode,
})

// The "input spec" — everything the Inputs panel + prompt box write into
// params. Blanked when entering a sub-mode with no stash yet; the
// generic generation settings (steps, resolution, guidance, ...) carry
// over and only diverge per-sub-mode once the user changes them there.
const BLANK_VIDEO_INPUT_PARAMS: Partial<GenerateParams> = {
  prompt: '',
  image_start: undefined,
  image_end: undefined,
  image_refs: undefined,
  frames_positions: undefined,
  injection_strength: undefined,
  video_prompt_type: '',
  image_prompt_type: '',
  audio_prompt_type: '',
  audio_guide: undefined,
  video_guide: undefined,
  video_mask: undefined,
  minimax_h3_control_visual_mode: 'prompt',
  video_source: undefined,
  input_video_strength: undefined,
}

const resolutionMap: Record<ResolutionPreset, Record<AspectRatio, string>> = {
  'auto': {
    'auto': 'auto',
    '16:9': 'auto',
    '9:16': 'auto',
    '1:1': 'auto',
    '4:3': 'auto',
    '3:4': 'auto',
  },
  '480p': {
    'auto': 'auto_480p',
    '16:9': '848x480',
    '9:16': '480x848',
    '1:1': '672x672',
    '4:3': '736x544',
    '3:4': '544x736',
  },
  '540p': {
    'auto': 'auto_540p',
    '16:9': '960x544',
    '9:16': '544x960',
    '1:1': '736x736',
    '4:3': '832x608',
    '3:4': '608x832',
  },
  '720p': {
    'auto': 'auto_720p',
    '16:9': '1280x720',
    '9:16': '720x1280',
    '1:1': '1024x1024',
    '4:3': '1104x832',
    '3:4': '832x1104',
  },
  '768p': {
    'auto': 'auto_768p',
    '16:9': '1344x768',
    '9:16': '768x1344',
    '1:1': '768x768',
    '4:3': '1024x768',
    '3:4': '768x1024',
  },
  '1080p': {
    'auto': 'auto_1080p',
    '16:9': '1920x1088',
    '9:16': '1088x1920',
    '1:1': '1024x1024',
    '4:3': '1920x1088',
    '3:4': '1088x1920',
  },
}

export function resolveResolution(
  modelOptions: ModelOptions | null,
  preset: ResolutionPreset,
  ratio: AspectRatio,
): string {
  const modelValues = modelOptions?.resolution_presets?.[preset]?.values
  return modelValues?.[ratio]
    || modelValues?.['16:9']
    || resolutionMap[preset]?.[ratio]
    || resolutionMap[preset]?.['16:9']
    || '1280x720'
}

function findResolutionSelection(
  resolution: string,
  modelOptions: ModelOptions | null,
): { preset: ResolutionPreset; ratio: AspectRatio } | null {
  const maps: Array<Partial<Record<ResolutionPreset, { values: Partial<Record<AspectRatio, string>> }>>> = []
  if (modelOptions?.resolution_presets) maps.push(modelOptions.resolution_presets)
  maps.push(Object.fromEntries(
    Object.entries(resolutionMap).map(([preset, values]) => [preset, { values }]),
  ) as Partial<Record<ResolutionPreset, { values: Partial<Record<AspectRatio, string>> }>>)

  for (const presetMap of maps) {
    for (const [preset, config] of Object.entries(presetMap)) {
      for (const [ratio, value] of Object.entries(config?.values || {})) {
        if (value === resolution) {
          return {
            preset: preset as ResolutionPreset,
            ratio: ratio as AspectRatio,
          }
        }
      }
    }
  }
  return null
}

// Memoization cache for filteredOutputs — ensures stable references
let _foCachedOutputs: OutputFile[] = []
let _foCachedFilter: MediaFilter = 'all'
let _foCachedResult: OutputFile[] = []

function computeFilteredOutputs(outputs: OutputFile[], mediaFilter: MediaFilter): OutputFile[] {
  if (outputs === _foCachedOutputs && mediaFilter === _foCachedFilter) {
    return _foCachedResult
  }
  _foCachedOutputs = outputs
  _foCachedFilter = mediaFilter
  if (mediaFilter === 'all') {
    _foCachedResult = outputs
  } else if (mediaFilter === 'videos') {
    _foCachedResult = outputs.filter(o => o.type === 'video')
  } else if (mediaFilter === 'images') {
    _foCachedResult = outputs.filter(o => o.type === 'image')
  } else if (mediaFilter === 'audio') {
    _foCachedResult = outputs.filter(o => o.type === 'audio')
  } else if (mediaFilter === 'avatars') {
    // "Edits" filter — show outputs from any of the Edit tab sub-modes.
    // Filter by `edit_sub_mode` (set by retake/inpaint/outpaint/restyle/
    // edit_anything endpoints) rather than `mode === 'avatar'`, because
    // those endpoints write `mode: 'video'` for backwards compatibility
    // and the old check produced an empty list. Falls back to mode check
    // for any legacy outputs that predate the edit_sub_mode tagging.
    _foCachedResult = outputs.filter(o => !!o.edit_sub_mode || o.mode === 'avatar')
  } else if (mediaFilter === 'multiclip') {
    // Backend already filters to multiclip + sliding window finals — pass through
    _foCachedResult = outputs
  } else if (mediaFilter === 'favorites') {
    _foCachedResult = outputs.filter(o => o.favorite)
  } else {
    _foCachedResult = outputs
  }
  return _foCachedResult
}

/** Resolve whether the current Director selection owns generated per-shot
 *  images. This mirrors services/director_video_strategy.py so the manual
 *  browser flow and the durable server pipeline take the same branch. */
function _directorUsesGeneratedShotImages(state: AppState): boolean {
  const videoModel = state.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
  const support = state.models.find(
    model => model.model_type === videoModel,
  )?.director?.shot_image_support
  const guidance = state.directorShotImageGuidance
  if (guidance === 'prompt_only') return false
  if (guidance === 'generate') return true
  if (!support || support === 'required') return true
  if (support === 'direct_references') return false
  return Boolean(
    state.directorReferenceImage
    || state.directorReferenceImagePath
    || state.directorCharacterRefs.length
    || state.directorCharacterRefPaths.length
    || state.directorLocationRefs.length
    || state.directorLocationRefPaths.length
  )
}

export const useStore = create<AppState>((set, get) => ({
  // Generation mode
  generationMode: 'video',
  editSubMode: 'retake' as import('../types').EditSubMode,
  setEditSubMode: (mode: import('../types').EditSubMode) => {
    const s = get()
    const prev = s.editSubMode
    set({ editSubMode: mode })
    if (mode === prev || s.generationMode !== 'avatar') return
    // Recast uses SCAIL-2 Replace; Repaint uses the proven SCAIL-2 Animate
    // path from Studio Video/Frames. Swap recipes when moving between those
    // modes and restore the previous LTX edit model when leaving both.
    const current = (s.params.model_type as string) || ''
    const isScail2 = (mt: string) => s.models.find(m => m.model_type === mt)?.architecture === 'scail2_14B'
    const enteringScail2Edit = mode === 'recast' || mode === 'restyle'
    const leavingScail2Edit = prev === 'recast' || prev === 'restyle'
    if (enteringScail2Edit) {
      const valid = mode === 'recast'
        ? current === 'scail2_14B_recast_fast' || current === 'scail2_14B'
        : current === 'scail2_14B_fast' || current === 'scail2_14B'
      if (!valid) {
        if (!leavingScail2Edit && !isScail2(current)) {
          _preScail2AvatarModel = current
        }
        const preferred = mode === 'recast'
          ? 'scail2_14B_recast_fast'
          : 'scail2_14B_fast'
        const target = s.models.some(m => m.model_type === preferred)
          ? preferred
          : s.models.some(m => m.model_type === 'scail2_14B')
            ? 'scail2_14B'
            : undefined
        if (target) get().selectModel(target)
      }
    } else if (leavingScail2Edit && isScail2(current)) {
      const restore = _preScail2AvatarModel && s.models.some(m => m.model_type === _preScail2AvatarModel)
        ? _preScail2AvatarModel
        : getDefaultModelForMode('avatar', s.families, s.models)
      if (restore) get().selectModel(restore)
    }
  },
  editVideoPath: '',
  editVideoUrl: '',
  editVideoFile: null,
  editVideoDuration: 0,
  editVideoResolution: '',
  editStartTime: 0,
  editEndTime: 5,
  editRetakeStrength: 0.85,
  editPromptStrength: 3.5,
  editAnythingLoraStrength: 1.0,
  editAnythingStartAnchor: null,
  editAnythingEndAnchor: null,
  editRepaintFrameFile: null,
  editRepaintFramePath: '',
  editRepaintFrameUrl: '',
  editRepaintMappings: [],
  editRepaintResolutionProfile: '480p',
  setEditRepaintFrame: (file, path, url) => set({
    editRepaintFrameFile: file,
    editRepaintFramePath: path,
    editRepaintFrameUrl: url,
  }),
  setEditRepaintMappings: mappings => set({
    editRepaintMappings: mappings.slice(0, 5),
  }),
  editRecastTarget: 'person',
  editRecastPersonCount: 1,
  editRecastRefFile: null,
  editRecastRefPath: '',
  editRecastRefUrl: '',
  editRecastMappings: [{ ...DEFAULT_RECAST_MAPPING }],
  editRecastRefAligned: false,
  editRecastIsolateReference: true,
  editRecastAutoFaceDetail: true,
  editRecastEnhancePrompt: false,
  editRecastProtectBystanders: false,
  editRecastPreserveBystanders: true,
  editRecastUseRelighting: false,
  editRecastResolutionProfile: '480p',
  setEditRecastMappings: mappings => set({
    editRecastMappings: mappings,
    editRecastTarget: mappings[0]?.target || 'person',
    editRecastPersonCount: Math.min(5, Math.max(1, mappings.length || 1)),
    editRecastRefFile: mappings[0]?.refFile || null,
    editRecastRefPath: mappings[0]?.refPath || '',
    editRecastRefUrl: mappings[0]?.refUrl || '',
    editRecastRefAligned: mappings[0]?.referenceAlignedToSource === true,
  }),
  setEditRecastRef: (file, path, url, aligned = false) => set(s => ({
    editRecastRefFile: file,
    editRecastRefPath: path,
    editRecastRefUrl: url,
    editRecastRefAligned: aligned,
    editRecastMappings: [
      {
        ...(s.editRecastMappings[0] || DEFAULT_RECAST_MAPPING),
        refFile: file,
        refPath: path,
        refUrl: url,
        referenceAlignedToSource: aligned,
      },
      ...s.editRecastMappings.slice(1),
    ],
  })),
  editReturnTarget: null,
  setEditAnythingStartAnchor: (path: string | null) => set({ editAnythingStartAnchor: path }),
  setEditAnythingEndAnchor: (path: string | null) => set({ editAnythingEndAnchor: path }),
  sendFrameToImageMode: async (which: 'start' | 'end' | 'recast' | 'repaint') => {
    const state = get()
    const clipPath = state.editVideoPath
    if (!clipPath) {
      console.error('Edit Anything: no source video loaded')
      return
    }
    const startTime = state.editStartTime || 0
    const endTime = state.editEndTime || state.editVideoDuration || 0
    if (endTime <= startTime) {
      console.error('Edit Anything: invalid trim range')
      return
    }

    // Snapshot user's current image-mode reference state BEFORE the
    // hijack so we can restore it on return / skip / cancel and not
    // disturb their non-Edit-Anything Image-mode workflow.
    const savedImageRefs = state.imageRefs
    const savedImageRefType = state.imageRefType

    // Decide which timestamp to grab. End frame is one frame INSIDE the
    // exclusive end (at -0.04s = ~one frame at 25fps) so it matches what
    // the retake pipeline will pin during inference.
    const tStart = which === 'end' ? Math.max(0, endTime - 0.04) : startTime
    try {
      let framePath = ''
      let frameUrl = ''
      // Repaint can refine its existing edited frame. The first trip starts
      // from the source trim frame; later trips start from the applied result.
      if (which === 'repaint' && state.editRepaintFramePath) {
        framePath = state.editRepaintFramePath
        const frameName = framePath.replace(/\\/g, '/').split('/').pop() || ''
        frameUrl = state.editRepaintFrameUrl || api.getFileUrl(frameName)
      } else {
        const res = await fetch('/api/v1/extract-frames', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            video_path: clipPath,
            ...(which === 'end' ? { end_time: tStart } : { start_time: tStart }),
          }),
        })
        if (!res.ok) throw new Error(`extract-frames failed: ${res.status}`)
        const data = await res.json()
        framePath = (which === 'end' ? data.end_path : data.start_path) as string
        frameUrl = (which === 'end' ? data.end_url : data.start_url) as string
      }

      // Use setGenerationMode rather than poking generationMode directly.
      // This is the proper switch — it picks the right model for image
      // mode (auto-restoring the user's last image-mode model or the
      // family default), reloads LoRAs, and resets image_mode + the
      // resolution/aspect presets that go with image generation. Without
      // this, the model stays on whatever LTX-2 video model was active.
      get().setGenerationMode('image')

      // Load the extracted frame into Image mode's REFERENCE images list
      // (the "Reference Images" drop zone in the sidebar). This is the
      // i2i / image-edit input slot — distinct from video mode's
      // image_start (which is i2v's "first frame"). ImageRefSection's
      // own useEffect picks the right imageRefType when imageRefs goes
      // from empty to populated; we leave that to it.
      const blob = await fetch(frameUrl).then(r => r.blob())
      const file = new File([blob], `${which}_frame.png`, { type: blob.type || 'image/png' })
      set(s => ({
        // Replace any pre-existing refs with just our extracted frame
        // for the duration of the round-trip. Restored from the
        // editReturnTarget snapshot when we return.
        imageRefs: [file],
        imageRefType: '',  // let ImageRefSection re-set the default for the new model
        // Make sure no stale i2v fields are populated — those would land
        // in video mode's i2v slot, which isn't what we want here.
        startImage: null,
        params: { ...s.params, image_start: '', image_mode: 1 },
        editReturnTarget: {
          anchor: which,
          framePath,
          clipPath,
          startTime,
          endTime,
          savedImageRefs,
          savedImageRefType,
        },
      }))
    } catch (e) {
      console.error('Failed to send frame to Image mode:', e)
    }
  },
  applyOutputAsAnchor: async () => {
    const state = get()
    const target = state.editReturnTarget
    if (!target) return
    // Find the latest image-mode output (newest first in the outputs list).
    const latestImage = state.outputs.find(o => o.type === 'image')
    if (!latestImage) {
      console.error('Edit Anything return: no image-mode output yet to apply')
      return
    }
    // The backend resolver in /api/v1/edit-anything will look in the
    // active workspace's outputs/ for a bare filename, so passing the
    // gallery name is enough.
    const outputPath = latestImage.name

    if (target.anchor === 'recast') {
      set(s => ({
        editRecastRefFile: null,
        editRecastRefPath: outputPath,
        editRecastRefUrl: latestImage.url,
        editRecastRefAligned: true,
        editRecastMappings: [
          {
            ...(s.editRecastMappings[0] || DEFAULT_RECAST_MAPPING),
            refFile: null,
            refPath: outputPath,
            refUrl: latestImage.url,
            referenceAlignedToSource: true,
          },
          ...s.editRecastMappings.slice(1),
        ],
      }))
    } else if (target.anchor === 'repaint') {
      set({
        editRepaintFrameFile: null,
        editRepaintFramePath: outputPath,
        editRepaintFrameUrl: latestImage.url,
      })
    } else if (target.anchor === 'start') {
      set({ editAnythingStartAnchor: outputPath })
    } else {
      set({ editAnythingEndAnchor: outputPath })
    }

    // Restore the user's pre-round-trip image-mode reference state and
    // switch back to Edit Anything. setGenerationMode handles the model
    // swap so they land back on their video model with the right LoRAs.
    get().setGenerationMode('avatar')
    set({
      editSubMode: target.anchor === 'recast'
        ? 'recast'
        : target.anchor === 'repaint'
          ? 'restyle'
          : 'edit_anything',
      editReturnTarget: null,
      imageRefs: target.savedImageRefs,
      imageRefType: target.savedImageRefType,
    })
  },
  skipAnchorPhase: () => {
    // Skip = return to Edit Anything without setting the anchor. Empty
    // slot → ltx2.py falls back to the source-extracted frame at
    // generation time (the morph-from-source default).
    const target = get().editReturnTarget
    get().setGenerationMode('avatar')
    set({
      editSubMode: target?.anchor === 'recast'
        ? 'recast'
        : target?.anchor === 'repaint'
          ? 'restyle'
          : 'edit_anything',
      editReturnTarget: null,
      ...(target ? { imageRefs: target.savedImageRefs, imageRefType: target.savedImageRefType } : {}),
    })
  },
  cancelAnchorReturn: () => {
    const target = get().editReturnTarget
    get().setGenerationMode('avatar')
    set({
      editSubMode: target?.anchor === 'recast'
        ? 'recast'
        : target?.anchor === 'repaint'
          ? 'restyle'
          : 'edit_anything',
      editReturnTarget: null,
      ...(target ? { imageRefs: target.savedImageRefs, imageRefType: target.savedImageRefType } : {}),
    })
  },
  editRetakeEngine: 'native' as const,
  editRegenerateAudio: true,
  editSamTarget: '',
  editInvertMask: false,
  editMasksPath: null,
  editMaskPreview: null,
  editDetectedTarget: '',
  continueVideo: null,
  continueVideoPath: '',
  continueVideoUrl: '',
  continueVideoDuration: 0,
  videoSubModeStash: {},
  setContinueVideo: (file, path, url, duration) => set({
    continueVideo: file, continueVideoPath: path, continueVideoUrl: url, continueVideoDuration: duration,
  }),
  clearContinueVideo: () => {
    // Also strip "V" from image_prompt_type — removing the source video
    // means the user is no longer in extend mode, so any leftover "V"
    // flag would cause the backend to demand a video_source we just
    // cleared. startGeneration has a defensive strip at submit time as
    // well, but cleaning state here keeps things consistent for any UI
    // that reads image_prompt_type directly.
    const currentParams = useStore.getState().params
    const ipt = (currentParams.image_prompt_type as string) || ''
    set({
      continueVideo: null, continueVideoPath: '', continueVideoUrl: '', continueVideoDuration: 0,
      params: {
        ...currentParams,
        video_source: undefined,
        image_prompt_type: ipt.replace(/V/g, ''),
      },
    })
  },
  blendClipA: null, blendClipAPath: '', blendClipAUrl: '', blendClipADuration: 0,
  blendClipB: null, blendClipBPath: '', blendClipBUrl: '', blendClipBDuration: 0,
  blendTransitionSec: 5,
  blendStrengthA: 1.0,
  blendStrengthB: 0.7,
  blendMotionPrefixSec: 1.0,
  blendMotionSuffixSec: 1.0,
  blendAnchorStrength: 0.7,
  setBlendClipA: (file, path, url, duration) => set({
    blendClipA: file, blendClipAPath: path, blendClipAUrl: url, blendClipADuration: duration,
  }),
  setBlendClipB: (file, path, url, duration) => set({
    blendClipB: file, blendClipBPath: path, blendClipBUrl: url, blendClipBDuration: duration,
  }),
  clearBlendClipA: () => set({ blendClipA: null, blendClipAPath: '', blendClipAUrl: '', blendClipADuration: 0 }),
  clearBlendClipB: () => set({ blendClipB: null, blendClipBPath: '', blendClipBUrl: '', blendClipBDuration: 0 }),
  setBlendTransitionSec: (sec) => set({ blendTransitionSec: sec }),
  setBlendStrengthA: (v) => set({ blendStrengthA: v }),
  setBlendStrengthB: (v) => set({ blendStrengthB: v }),
  setBlendMotionPrefixSec: (v) => set({ blendMotionPrefixSec: v }),
  setBlendMotionSuffixSec: (v) => set({ blendMotionSuffixSec: v }),
  setBlendAnchorStrength: (v) => set({ blendAnchorStrength: v }),
  blendMode: 'overlap' as const,
  blendOverlapSec: 3,
  setBlendMode: (mode) => set({ blendMode: mode }),
  setBlendOverlapSec: (sec) => set({ blendOverlapSec: sec }),
  outpaintPadding: { top: 0, bottom: 0, left: 0, right: 0 },
  setOutpaintPadding: (padding) => set({ outpaintPadding: padding }),
  outpaintResolutionPreset: 'auto',
  setOutpaintResolutionPreset: (preset) => set({ outpaintResolutionPreset: preset }),
  // 'source' = canvas matches source aspect (no extension by default)
  outpaintAspect: 'source',
  setOutpaintAspect: (a) => set({ outpaintAspect: a }),
  // Default video box: full canvas (no padding). Will be re-fitted by the
  // OutpaintCanvas when the user picks a non-source aspect.
  outpaintVideoBox: { x: 0, y: 0, w: 1, h: 1 },
  setOutpaintVideoBox: (box) => set({ outpaintVideoBox: box }),
  outpaintTrimStart: 0,
  outpaintTrimEnd: 0,
  setOutpaintTrimStart: (t) => set({ outpaintTrimStart: t }),
  setOutpaintTrimEnd: (t) => set({ outpaintTrimEnd: t }),
  outpaintSourcePreservation: 1.0,
  setOutpaintSourcePreservation: (v) => set({ outpaintSourcePreservation: v }),
  outpaintLoraStrength: 1.0,
  setOutpaintLoraStrength: (v) => set({ outpaintLoraStrength: v }),
  outpaintMaskPreserving: true,
  setOutpaintMaskPreserving: (v) => set({ outpaintMaskPreserving: v }),
  outpaintPreserveSourceAudio: true,
  setOutpaintPreserveSourceAudio: (v) => set({ outpaintPreserveSourceAudio: v }),
  outpaintLockSourcePixels: false,  // default OFF — visible rectangle seam outweighs benefit
  setOutpaintLockSourcePixels: (v) => set({ outpaintLockSourcePixels: v }),
  outpaintTrimSmear: true,  // default ON — fixes the 9-frame stutter at window 1→2 boundary
  setOutpaintTrimSmear: (v) => set({ outpaintTrimSmear: v }),
  outpaintWindowSize: 241,  // LTX-2 default (~10s @ 24fps)
  setOutpaintWindowSize: (v) => set({ outpaintWindowSize: v }),
  outpaintWindowOverlap: 9,  // LTX-2 default
  setOutpaintWindowOverlap: (v) => set({ outpaintWindowOverlap: v }),
  setEditVideoPath: (path) => set({ editVideoPath: path }),
  setEditVideo: (file, path, url, duration, resolution) => set({
    editVideoFile: file, editVideoPath: path, editVideoUrl: url,
    editVideoDuration: duration, editVideoResolution: resolution,
    editEndTime: duration,
  }),
  clearEditVideo: () => set({
    editVideoFile: null, editVideoPath: '', editVideoUrl: '',
    editVideoDuration: 0, editVideoResolution: '', editStartTime: 0, editEndTime: 5,
    editMasksPath: null, editMaskPreview: null, editDetectedTarget: '',
  }),
  musicDescription: '',
  setMusicDescription: (s) => set({ musicDescription: s }),
  musicInstrumental: false,
  setMusicInstrumental: (b) => set({ musicInstrumental: b }),
  audioSubMode: 'speech' as import('../types').AudioSubMode,
  selectedModelPerAudioSubMode: {} as Partial<Record<import('../types').AudioSubMode, string>>,
  setAudioSubMode: (subMode) => {
    const { audioSubMode: prevSub, params, models } = get()
    if (subMode === prevSub) return
    // Save current model for the sub-mode we're leaving
    const savedModels = { ...get().selectedModelPerAudioSubMode, [prevSub]: params.model_type }
    // Determine model for target sub-mode
    const audioSubModeDefaults: Record<import('../types').AudioSubMode, string> = {
      speech: 'kugelaudio_0_open',
      // XL SFT LM_4B: the premium CFG variant + strongest LM — the
      // quality default. Turbo variants remain enabled for speed.
      music: 'ace_step_v1_5_xl_sft_lm_4b',
      sfx: 'mmaudio_v2',
      mixer: '',  // Mixer doesn't use a model — it's an ffmpeg-based tool
    }
    const saved = savedModels[subMode]
    const targetModel = (saved && models.some(m => m.model_type === saved))
      ? saved
      : audioSubModeDefaults[subMode]
    set({ audioSubMode: subMode, selectedModelPerAudioSubMode: savedModels })
    if (targetModel && models.some(m => m.model_type === targetModel)) {
      get().selectModel(targetModel)
    }
  },
  selectedModelPerMode: {},
  savedLoraPerMode: {},
  savedParamsPerMode: {},
  savedPromptPerMode: {} as Partial<Record<string, string>>,

  setGenerationMode: (mode) => {
    // Tools is a non-generative post-processing area — it owns no model, so
    // skip the per-mode model/LoRA/params RESTORE machinery entirely. We still
    // SAVE the leaving mode's state (prompt / model / LoRAs / params snapshot)
    // so returning to it restores correctly, leave `params` untouched (no model
    // load, no defaults reset), and persist the *previous* real mode as the
    // landing mode so a reload doesn't drop into Tools with no model loaded.
    if (mode === 'tools') {
      const s = get()
      const prev = s.generationMode
      if (prev === 'tools') { set({ generationMode: 'tools' }); return }
      const { model_type: _mt, prompt: _p, activated_loras: _al, loras_multipliers: _lm, ...paramsSnapshot } = s.params
      const savedModels = { ...s.selectedModelPerMode, [prev]: s.params.model_type }
      const savedParams = {
        ...s.savedParamsPerMode,
        [prev]: { ...paramsSnapshot, filmGrainIntensity: s.filmGrainIntensity, filmGrainSaturation: s.filmGrainSaturation, durationSeconds: s.durationSeconds },
      }
      const savedLoras = {
        ...s.savedLoraPerMode,
        [prev]: { activated_loras: s.params.activated_loras || [], loras_multipliers: s.params.loras_multipliers || '', loraWeights: s.loraWeights, availableLoras: s.availableLoras },
      }
      const savedPrompts = { ...s.savedPromptPerMode, [prev]: s.params.prompt }
      set({
        generationMode: 'tools',
        selectedModelPerMode: savedModels,
        savedParamsPerMode: savedParams,
        savedLoraPerMode: savedLoras,
        savedPromptPerMode: savedPrompts,
      })
      _saveSettings({ generationMode: prev, selectedModelPerMode: savedModels, savedParamsPerMode: savedParams, savedLoraPerMode: savedLoras, savedPromptPerMode: savedPrompts }, s.loraIdByFilename)
      return
    }
    const { families, models, generationMode: prevMode, params, selectedModelPerMode, savedLoraPerMode, savedParamsPerMode, loraWeights, availableLoras, savedPromptPerMode } = get()
    // Save prompt for the mode we're leaving
    const savedPrompts = { ...savedPromptPerMode, [prevMode]: params.prompt }
    // Save current model + LoRA + params state for the mode we're leaving
    const savedModels = { ...selectedModelPerMode, [prevMode]: params.model_type }
    const savedLoras = {
      ...savedLoraPerMode,
      [prevMode]: {
        activated_loras: params.activated_loras || [],
        loras_multipliers: params.loras_multipliers || '',
        loraWeights,
        availableLoras,
      },
    }
    // Save the FULL params snapshot for the leaving mode. Strip the
    // fields that are tracked separately in their own per-mode state
    // structures (model_type → selectedModelPerMode, prompt →
    // savedPromptPerMode, activated_loras / loras_multipliers →
    // savedLoraPerMode) to avoid double-bookkeeping. Everything else
    // — including repeat_generation, negative_prompt, video_prompt_type,
    // video_guide, image_refs, frames_positions, MMAudio_*, etc. — is
    // captured here so it survives a switch-and-return AND doesn't
    // leak into other modes.
    const { model_type: _mt, prompt: _p, activated_loras: _al, loras_multipliers: _lm, ...paramsSnapshot } = params
    const savedParams = {
      ...savedParamsPerMode,
      [prevMode]: {
        ...paramsSnapshot,
        filmGrainIntensity: get().filmGrainIntensity,
        filmGrainSaturation: get().filmGrainSaturation,
        // Save durationSeconds per-mode so audio's 600/1800 (Kugel/Scenema
        // slider max) doesn't leak into video on mode-switch back. Audio
        // mode's loadModelOptions still overrides with the slider.max on
        // model select, so this only matters for video/image/avatar.
        durationSeconds: get().durationSeconds,
      },
    }
    // Restore saved model for target mode, or fall back to default
    const savedModel = savedModels[mode]
    const restoredModel = savedModel && models.some(m => m.model_type === savedModel)
      ? savedModel
      : getDefaultModelForMode(mode, families, models)
    const newModelType = restoredModel || params.model_type
    // Restore saved LoRA state for target mode (if same model)
    const restoredLora = savedLoras[mode]
    const sameModel = restoredLora && savedModel === newModelType
    // Restore the saved params snapshot for the target mode. If the
    // user never visited this mode before, fall back to defaultParams
    // (NOT the previous mode's params — that's what caused the leak).
    const restoredSnapshot = savedParams[mode]
    // Extract film grain from snapshot (top-level store state, not in params)
    const restoredFilmGrain = restoredSnapshot
      ? { filmGrainIntensity: restoredSnapshot.filmGrainIntensity ?? 0, filmGrainSaturation: restoredSnapshot.filmGrainSaturation ?? 0.5 }
      : { filmGrainIntensity: 0, filmGrainSaturation: 0.5 }
    // Restore durationSeconds for the target mode. Non-audio modes (video,
    // avatar, image) fall back to 5s on first visit. Audio mode's
    // durationSeconds gets overridden by loadModelOptions when it sees
    // audio_only && duration_slider, so the snapshot value is mostly
    // ignored there — it's still saved for symmetry.
    const restoredDuration = restoredSnapshot && typeof restoredSnapshot.durationSeconds === 'number'
      ? restoredSnapshot.durationSeconds as number
      : 5
    // Strip filmGrain + durationSeconds keys before applying — they don't belong in params
    const { filmGrainIntensity: _fgi, filmGrainSaturation: _fgs, durationSeconds: _ds, ...restoredParams } = restoredSnapshot || {}
    // Restore saved prompt for target mode (or empty for first visit)
    const restoredPrompt = savedPrompts[mode] ?? ''

    set(_s => ({
      generationMode: mode,
      selectedModelPerMode: savedModels,
      savedLoraPerMode: savedLoras,
      savedParamsPerMode: savedParams,
      savedPromptPerMode: savedPrompts,
      // Default to Auto resolution + aspect in image mode (matches reference image)
      ...(mode === 'image' ? { resolutionPreset: 'auto' as ResolutionPreset, aspectRatio: 'auto' as AspectRatio } : {}),
      ...restoredFilmGrain,
      durationSeconds: restoredDuration,
      // Build params from defaults + restored snapshot. We deliberately
      // do NOT spread `...s.params` here — that's the line that caused
      // every previous-mode field to leak into the new mode. Starting
      // from defaults ensures only the restored snapshot's fields (the
      // user's actual choices in this mode, or nothing on first visit)
      // are present. Then layer model_type / prompt / LoRAs from their
      // separate stores on top, plus the special image_mode logic.
      params: {
        ...defaultParams,
        ...restoredParams,
        model_type: newModelType,
        prompt: restoredPrompt,
        image_mode: mode === 'image' ? 1 : (restoredParams.image_mode ?? 0),
        activated_loras: sameModel ? restoredLora.activated_loras : [],
        loras_multipliers: sameModel ? restoredLora.loras_multipliers : '',
      },
      h3WindowPlan: null,
      loraWeights: sameModel ? restoredLora.loraWeights : {},
      availableLoras: sameModel ? restoredLora.availableLoras : [],
    }))
    if (newModelType && !sfxModelTypes.has(newModelType)) {
      if (!sameModel) {
        get().loadLoras(newModelType)
      }
      get().loadModelOptions(newModelType)
      // Mode switch counts as a model selection too — apply the new
      // model's defaults so numeric primaries (steps, CFG, flow_shift,
      // sample_solver) match what that model expects rather than what
      // the previous mode's model was using. See _applyModelDefaults
      // for the field list and rationale.
      _applyModelDefaults(get, set, newModelType)
    }
    // Persist to localStorage
    _saveSettings({
      generationMode: mode,
      selectedModelPerMode: savedModels,
      savedParamsPerMode: savedParams,
      savedLoraPerMode: savedLoras,
      savedPromptPerMode: savedPrompts,
    }, get().loraIdByFilename)
  },

  params: { ...defaultParams },
  setParam: (key, value) => {
    // Per-sub-mode isolation: remember the outgoing sub-mode BEFORE the
    // param write flips image_mode (see videoSubModeStash).
    const prevImageMode = key === 'image_mode' ? ((get().params.image_mode as number) ?? 0) : null
    const invalidatesH3Plan = [
      'prompt', 'model_type', 'resolution', 'image_start', 'image_end',
      'image_mode', 'image_refs', 'frames_positions', 'video_prompt_type',
      'minimax_h3_camera_coverage',
      'minimax_h3_multi_window',
      'minimax_h3_reference_sequence', 'minimax_h3_references',
      'minimax_h3_sequence_prompt_mode',
      'minimax_h3_sequence_continuity',
      'minimax_h3_sequence_clip_frames',
      'minimax_h3_sequence_memory_override',
    ].includes(String(key))
    set(s => {
      const nextParams = { ...s.params, [key]: value }
      if (key === 'prompt') {
        delete nextParams._h3_original_prompt
        const editedLines = typeof value === 'string'
          ? value.replace(/\r\n?/g, '\n').split('\n').map(line => line.trim()).filter(Boolean)
          : []
        const reviewedLtxPlan = (
          s.modelOptions?.multi_window_sequence_controls === true
          && s.params.ltx_multi_window === true
          && s.params.ltx_window_prompt_mode !== 'manual'
          && Array.isArray(s.params.ltx_window_prompts)
          && editedLines.length === s.params.ltx_window_prompts.length
        )
        if (reviewedLtxPlan) {
          nextParams.ltx_window_prompts = editedLines
        } else {
          delete nextParams._ltx_original_prompt
          delete nextParams.ltx_window_prompts
        }
      }
      if (
        key === 'ltx_multi_window'
        || key === 'ltx_window_prompt_mode'
        || key === 'model_type'
      ) {
        delete nextParams._ltx_original_prompt
        delete nextParams.ltx_window_prompts
      }
      return {
        params: nextParams,
        ...(invalidatesH3Plan ? { h3WindowPlan: null } : {}),
        ...(invalidatesH3Plan ? { promptEnhanceError: null } : {}),
      }
    })
    // Auto-parse speaker names from prompt whenever audio mode has at least
    // one voice slot. Previously gated on audio_prompt_type.includes('B')
    // (multi-voice only), but the user expects single-voice ("Peter: hello")
    // to populate voice slot 1 too. Voice-count gate covers both cases —
    // ttsVoiceCount > 0 means at least one voice clone is active.
    if (key === 'prompt' && typeof value === 'string' && get().generationMode === 'audio' && get().ttsVoiceCount > 0) {
      get()._autoParseSpkeakerNames(value)
    }
    // Handle sub-mode transitions (Frames / Multi-Shot / Extend / Blend)
    if (key === 'image_mode') {
      // Each Studio Video sub-mode is an ISOLATED working set: stash the
      // outgoing sub-mode's full state (prompt, input tiles, settings)
      // and bring back the incoming one. A sub-mode visited for the
      // first time keeps the generic settings but starts with blank
      // inputs — so Extend opens clean while the Frames setup (injected
      // keyframes and all) survives the round-trip untouched.
      const s1 = get()
      if (s1.generationMode === 'video' && typeof value === 'number' && prevImageMode !== null && value !== prevImageMode) {
        const stash = { ...s1.videoSubModeStash, [prevImageMode]: captureVideoSubModeStash(s1) }
        const saved = stash[value]
        if (saved) {
          set({
            videoSubModeStash: stash,
            // Model + LoRA selection stay shared across sub-modes — keep
            // the live values, restore everything else.
            params: {
              ...saved.params,
              image_mode: value,
              model_type: s1.params.model_type,
              activated_loras: s1.params.activated_loras,
              loras_multipliers: s1.params.loras_multipliers,
            },
            startImage: saved.startImage,
            endImage: saved.endImage,
            continueVideo: saved.continueVideo,
            continueVideoPath: saved.continueVideoPath,
            continueVideoUrl: saved.continueVideoUrl,
            continueVideoDuration: saved.continueVideoDuration,
            audioGuideFilename: saved.audioGuideFilename,
            imageRefs: saved.imageRefs,
            imageRefType: saved.imageRefType,
            removeBackgroundRefs: saved.removeBackgroundRefs,
            durationSeconds: saved.durationSeconds,
            slidingWindowSeconds: saved.slidingWindowSeconds,
            slidingWindowOverlap: saved.slidingWindowOverlap,
            clips: saved.clips,
            singlePromptMode: saved.singlePromptMode,
          })
        } else {
          set(s => ({
            videoSubModeStash: stash,
            params: { ...s.params, ...BLANK_VIDEO_INPUT_PARAMS },
            startImage: null,
            endImage: null,
            continueVideo: null,
            continueVideoPath: '',
            continueVideoUrl: '',
            continueVideoDuration: 0,
            audioGuideFilename: null,
            imageRefs: [],
            imageRefType: '',
            removeBackgroundRefs: false,
            // durationSeconds + sliding window intentionally carry over:
            // they're settings, not inputs — they diverge per sub-mode
            // only after the user changes them there.
          }))
        }
      }
      // Multi-clip transitions (after the stash swap so syncClipCount
      // sees the restored duration/params).
      if (value === 2) {
        get().syncClipCount()
      } else {
        set({ clips: [], singlePromptMode: false })
      }
    }
    // Snapshot the changed param into the current mode's IN-MEMORY
    // record so it survives a mode switch + return within this session.
    // Skip keys that are tracked in their own per-mode structures
    // (model_type, prompt, LoRA fields) to avoid double-bookkeeping.
    // Everything else — repeat_generation, negative_prompt,
    // num_inference_steps, video_prompt_type, video_guide, image_refs,
    // frames_positions, MMAudio_*, etc. — gets snapshotted here.
    //
    // Deliberately NOT written to localStorage: a page refresh starts
    // the working state (prompt, seed, LoRA selection, Advanced values)
    // from the model's defaults. v1.2.0 persisted every edit across
    // refreshes and users found the stale text/seeds surprising —
    // in-session mode-switch persistence is the wanted behavior,
    // refresh is a clean slate (see loadModels).
    if (key !== 'model_type' && key !== 'prompt' && key !== 'activated_loras' && key !== 'loras_multipliers') {
      const s = get()
      const mode = s.generationMode
      const { model_type: _mt, prompt: _p, activated_loras: _al, loras_multipliers: _lm, ...paramsSnapshot } = s.params
      const updatedSavedParams = {
        ...s.savedParamsPerMode,
        [mode]: {
          ...paramsSnapshot,
          filmGrainIntensity: s.filmGrainIntensity,
          filmGrainSaturation: s.filmGrainSaturation,
        },
      }
      set({ savedParamsPerMode: updatedSavedParams })
    }
  },
  setParams: (partial) => {
    set(s => ({ params: { ...s.params, ...partial } }))
  },

  settingsOpen: false,
  toggleSettings: () => set(s => ({ settingsOpen: !s.settingsOpen })),
  setSettingsOpen: (open) => set({ settingsOpen: open }),
  sidebarOpen: false,
  toggleSidebar: () => set(s => ({ sidebarOpen: !s.sidebarOpen })),
  setSidebarOpen: (open) => set({ sidebarOpen: open }),

  // Theme — initial value reads from localStorage (with legacy
  // single-theme migration) so it matches what the inline script in
  // index.html applied to <html>. The setters write to the DOM and
  // localStorage via applyThemePrefs, which also installs the OS
  // scheme listener that makes 'auto' live-switch.
  themePrefs: getStoredPrefs(),
  setThemeMode: (mode) => {
    const prefs = { ...get().themePrefs, mode }
    applyThemePrefs(prefs)
    set({ themePrefs: prefs })
  },
  setThemeFamily: (family) => {
    const prefs = { ...get().themePrefs, family }
    applyThemePrefs(prefs)
    set({ themePrefs: prefs })
  },

  // CivitAI LoRA Browser
  // Director Pipeline Dashboard
  retakeDialogOpen: false,
  retakeSourceFile: null,
  openRetakeDialog: (filename) => set({ retakeDialogOpen: true, retakeSourceFile: filename }),
  closeRetakeDialog: () => set({ retakeDialogOpen: false, retakeSourceFile: null }),

  dashboardOpen: false,
  dashboardPipelineList: [],
  dashboardSelectedPipeline: null,
  dashboardLoading: false,
  setDashboardOpen: (open) => {
    set({ dashboardOpen: open })
    if (open) {
      get().loadPipelineList()
      const selected = get().dashboardSelectedPipeline
      if (selected) get().loadSavedPipeline(selected.pipeline_id)
    }
  },
  loadPipelineList: async () => {
    const loadToken = ++_dashboardPipelineListLoadToken
    try {
      const { pipelines } = await api.fetchPipelineList()
      if (loadToken !== _dashboardPipelineListLoadToken) return
      set({ dashboardPipelineList: pipelines })

      // The repair worker belongs to the server, so a browser reload must
      // rediscover active operations and resume UI polling without requiring
      // the Dashboard to be opened first. Keep discovery separate from the
      // selected pipeline so bootstrapping never opens or changes the overlay.
      for (const item of pipelines) {
        if (!item.repair_status || !DIRECTOR_REPAIR_ACTIVE.has(item.repair_status)) continue
        if (_directorRepairPolls.has(item.id) || _directorRepairDiscoveries.has(item.id)) continue

        const discovery = {}
        _directorRepairDiscoveries.set(item.id, discovery)
        void api.fetchSavedPipeline(item.id).then(pipeline => {
          if (_directorRepairDiscoveries.get(item.id) !== discovery) return
          if (_directorRepairPolls.has(item.id)) return

          const repair = pipeline.repair
          if (_repairNeedsPolling(repair)) {
            get().pollPipelineRepair(item.id, repair!.operation_id)
            return
          }

          // The operation may have finished between the list and detail
          // requests. Reflect that terminal state and refresh newly-created
          // media instead of waiting for another Dashboard visit.
          set(s => ({
            dashboardPipelineList: s.dashboardPipelineList.map(entry =>
              entry.id === item.id
                ? { ...entry, repair_status: repair?.status || null }
                : entry),
          }))
          void get().loadOutputs()
        }).catch(e => {
          console.warn(`Failed to reconnect Director repair for ${item.id}:`, e)
        }).finally(() => {
          if (_directorRepairDiscoveries.get(item.id) === discovery) {
            _directorRepairDiscoveries.delete(item.id)
          }
        })
      }
    } catch (e) {
      if (loadToken !== _dashboardPipelineListLoadToken) return
      console.error('Failed to load pipeline list:', e)
    }
  },
  loadSavedPipeline: async (pid) => {
    const loadToken = ++_dashboardPipelineLoadToken
    set({ dashboardLoading: true })
    try {
      const pipeline = await api.fetchSavedPipeline(pid)
      if (loadToken !== _dashboardPipelineLoadToken) return
      set({ dashboardSelectedPipeline: pipeline, dashboardLoading: false })
      if (_repairNeedsPolling(pipeline.repair)) {
        get().pollPipelineRepair(pid, pipeline.repair!.operation_id)
      }
    } catch (e) {
      if (loadToken !== _dashboardPipelineLoadToken) return
      console.error('Failed to load pipeline:', e)
      set({ dashboardLoading: false })
    }
  },
  deletePipeline: async (pid) => {
    // Clear the selection AND drop the pid from the list in the same
    // update: the dashboard's auto-load effect selects pipelineList[0]
    // whenever selection is null, so a stale list would immediately
    // re-fetch the pipeline being deleted (re-mounting its <img>/<video>
    // elements and re-locking the files on Windows).
    _dashboardPipelineLoadToken += 1
    _dashboardPipelineListLoadToken += 1
    set(s => ({
      dashboardSelectedPipeline: null,
      dashboardPipelineList: s.dashboardPipelineList.filter(p => p.id !== pid),
    }))
    await api.deletePipeline(pid)
    await get().loadPipelineList()
    // Pipeline media were gallery items too — refresh the feed.
    get().loadOutputs()
    get().loadWorkspaces()
  },
  tagClip: async (pid, clipIndex, tag) => {
    try {
      await api.tagPipelineClip(pid, clipIndex, tag)
      // Update local state
      set(s => {
        if (!s.dashboardSelectedPipeline || s.dashboardSelectedPipeline.pipeline_id !== pid) return {}
        const clips = [...s.dashboardSelectedPipeline.clips]
        if (clipIndex < clips.length) {
          clips[clipIndex] = { ...clips[clipIndex], tag: tag as 'good' | 'needs_work' | null }
        }
        return { dashboardSelectedPipeline: { ...s.dashboardSelectedPipeline, clips } }
      })
    } catch (e) {
      console.error('Failed to tag clip:', e)
    }
  },
  startPipelineRepair: async (pid: string) => {
    const { repair } = await api.startPipelineRepair(pid)
    set(s => {
      const dashboardPipelineList = s.dashboardPipelineList.map(item =>
        item.id === pid ? { ...item, repair_status: repair.status } : item)
      if (!s.dashboardSelectedPipeline || s.dashboardSelectedPipeline.pipeline_id !== pid) {
        return { dashboardPipelineList }
      }
      return {
        dashboardPipelineList,
        dashboardSelectedPipeline: {
          ...s.dashboardSelectedPipeline,
          repair,
        },
      }
    })
    get().pollPipelineRepair(pid, repair.operation_id)
    return repair
  },
  cancelPipelineRepair: async (pid: string) => {
    const { repair } = await api.cancelPipelineRepair(pid)
    set(s => {
      const dashboardPipelineList = s.dashboardPipelineList.map(item =>
        item.id === pid ? { ...item, repair_status: repair.status } : item)
      if (!s.dashboardSelectedPipeline || s.dashboardSelectedPipeline.pipeline_id !== pid) {
        return { dashboardPipelineList }
      }
      return {
        dashboardPipelineList,
        dashboardSelectedPipeline: {
          ...s.dashboardSelectedPipeline,
          repair,
        },
      }
    })
    get().pollPipelineRepair(pid, repair.operation_id)
    return repair
  },
  pollPipelineRepair: (pid: string, operationId: string) => {
    const existing = _directorRepairPolls.get(pid)
    if (existing?.operationId === operationId) return
    if (existing) _stopDirectorRepairPoll(pid)

    const poll: DirectorRepairPoll = { operationId, timer: null }
    _directorRepairPolls.set(pid, poll)

    const tick = async () => {
      if (_directorRepairPolls.get(pid) !== poll) return
      poll.timer = null
      try {
        const pipeline = await api.fetchSavedPipeline(pid)
        if (_directorRepairPolls.get(pid) !== poll) return

        const repair = pipeline.repair
        set(s => {
          const dashboardPipelineList = s.dashboardPipelineList.map(item =>
            item.id === pid ? { ...item, repair_status: repair?.status || null } : item)
          if (!s.dashboardSelectedPipeline || s.dashboardSelectedPipeline.pipeline_id !== pid) {
            return { dashboardPipelineList }
          }
          return { dashboardPipelineList, dashboardSelectedPipeline: pipeline }
        })

        if (repair?.operation_id !== operationId) {
          _stopDirectorRepairPoll(pid)
          if (_repairNeedsPolling(repair)) {
            get().pollPipelineRepair(pid, repair!.operation_id)
          } else {
            void get().loadPipelineList()
            void get().loadOutputs()
          }
          return
        }
        if (!_repairNeedsPolling(repair)) {
          _stopDirectorRepairPoll(pid)
          void get().loadPipelineList()
          void get().loadOutputs()
          return
        }
      } catch (e) {
        console.warn(`Director repair poll failed for ${pid}; retrying:`, e)
      }

      if (_directorRepairPolls.get(pid) === poll) {
        poll.timer = window.setTimeout(tick, DIRECTOR_REPAIR_POLL_MS)
      }
    }

    void tick()
  },
  rerunClipImage: async (pid: string, clipIndex: number, prompt?: string) => {
    set({ dashboardLoading: true })
    try {
      const result = await api.rerunClipImage(pid, clipIndex, prompt)
      // Refresh the pipeline to get updated state
      const pipeline = await api.fetchSavedPipeline(pid)
      set({ dashboardSelectedPipeline: pipeline, dashboardLoading: false })
      // New files (rerun clip / rejoin video) land in the outputs folder —
      // refresh the gallery so they appear without a browser reload.
      get().loadOutputs()
      return result
    } catch (e) {
      console.error('Re-run image failed:', e)
      set({ dashboardLoading: false })
      throw e
    }
  },
  rerunClipVideo: async (pid: string, clipIndex: number, prompt?: string) => {
    set({ dashboardLoading: true })
    try {
      const result = await api.rerunClipVideo(pid, clipIndex, prompt)
      const pipeline = await api.fetchSavedPipeline(pid)
      set({ dashboardSelectedPipeline: pipeline, dashboardLoading: false })
      // New files (rerun clip / rejoin video) land in the outputs folder —
      // refresh the gallery so they appear without a browser reload.
      get().loadOutputs()
      return result
    } catch (e) {
      console.error('Re-run video failed:', e)
      set({ dashboardLoading: false })
      throw e
    }
  },
  rejoinPipelineClips: async (pid: string) => {
    set({ dashboardLoading: true })
    try {
      const result = await api.rejoinPipeline(pid)
      const pipeline = await api.fetchSavedPipeline(pid)
      set({ dashboardSelectedPipeline: pipeline, dashboardLoading: false })
      // New files (rerun clip / rejoin video) land in the outputs folder —
      // refresh the gallery so they appear without a browser reload.
      get().loadOutputs()
      return result
    } catch (e) {
      console.error('Rejoin failed:', e)
      set({ dashboardLoading: false })
      throw e
    }
  },
  resumePipeline: async (pid: string) => {
    // Kick the crashed pipeline back into running server-side, then close
    // the Dashboard and reconnect the Director view to it so progress shows.
    await api.resumePipeline(pid)
    set({
      dashboardOpen: false,
      pipelineId: pid,
      pipelineStatus: null,
      pipelinePolling: true,
      directorLoading: true,
    })
    get().pollPipelineStatus()
  },

  // ── Recipes (one-click Studio presets) ────────────────────────────
  recipesOpen: false,
  setRecipesOpen: (open) => {
    set({ recipesOpen: open })
    if (open) get().loadRecipes()
  },
  recipes: [],
  recipesLoading: false,
  loadRecipes: async () => {
    set({ recipesLoading: true })
    try {
      const { recipes } = await api.fetchRecipes()
      set({ recipes, recipesLoading: false })
    } catch (e) {
      console.error('Failed to load recipes:', e)
      set({ recipes: [], recipesLoading: false })
    }
  },
  applyRecipe: async (id) => {
    // Applies a recipe like Load Settings applies a saved output: switch
    // model + generation mode, land the tuned params in the active Studio
    // working set, and PREPOPULATE the prompt (a real, editable value — not
    // placeholder text) so the user just tweaks the subject. Seed and repeat
    // reset so a recipe reproduces a look, not a specific frame.
    const recipe = await api.fetchRecipe(id)
    const { models } = get()
    const model = models.find(m => m.model_type === recipe.model_type)
    const mode = model ? getModelMode(recipe.model_type, model.family) : ((recipe.mode as GenerationMode) || 'video')

    const activated = (recipe.loras || []).map(l => l.filename)
    const multipliers = (recipe.loras || []).map(l => String(l.multiplier ?? '1.0')).join(' ')
    const loraWeights: Record<string, number[]> = {}
    for (const l of recipe.loras || []) {
      loraWeights[l.filename] = String(l.multiplier ?? '1.0').split(';').map(Number)
    }

    set(s => ({
      generationMode: mode,
      // NOTE: do NOT close the overlay here. The RecipesOverlay closes
      // itself on success, but keeps itself open when the recipe needs
      // LoRAs you don't have — so it can show the download prompt. Closing
      // here made that prompt dead code (recipe applied, LoRA missing, user
      // generated → cryptic "Loras missing" failure with no guidance).
      params: {
        ...s.params,
        ...(recipe.params as Partial<GenerateParams>),
        model_type: recipe.model_type,
        prompt: recipe.prompt_example || '',
        activated_loras: activated,
        loras_multipliers: multipliers,
        seed: -1,
        repeat_generation: 1,
        // Recipes are look presets — land in the base Studio sub-mode
        // (Frames for video, image-output for image), not Extend/Blend.
        image_mode: mode === 'image' ? 1 : 0,
      },
      loraWeights,
      availableLoras: [],
      selectedModelPerMode: { ...s.selectedModelPerMode, [mode]: recipe.model_type },
      h3WindowPlan: null,
    }))

    if (recipe.model_type) {
      get().loadModelOptions(recipe.model_type)
      // Derive duration from video_length if the recipe carried one.
      const vlen = (recipe.params as Record<string, unknown>)?.video_length
      const fps = model?.fps || 16
      if (typeof vlen === 'number' && vlen > 0) {
        set({ durationSeconds: Math.round((vlen / fps) * 10) / 10 })
      }
      // Await the LoRA list so we can report which recipe LoRAs are missing.
      await get().loadLoras(recipe.model_type)
    }

    const present = new Set(get().availableLoras.map(x => (x || '').replace(/\\/g, '/').split('/').pop() || ''))
    const missing = (recipe.loras || []).filter(l => !present.has(l.filename))
    return { missing }
  },
  saveRecipeFromOutput: async (outputName, name, description, nsfw) => {
    await api.saveRecipeFromOutput({ output_name: outputName, name, description, nsfw })
    if (get().recipesOpen) get().loadRecipes()
  },
  deleteRecipe: async (id) => {
    await api.deleteRecipe(id)
    set(s => ({ recipes: s.recipes.filter(r => r.id !== id) }))
  },
  downloadRecipeLora: async (lora, modelType) => {
    // Best-effort fetch of a recipe's LoRA from its CivitAI source. Portable
    // recipes carry a direct download_url in source_url; if it isn't a
    // CivitAI URL the backend rejects it and the UI falls back to the link.
    if (!lora.source_url) throw new Error('This recipe has no download source for that LoRA — install it manually.')
    const model = get().models.find(m => m.model_type === modelType)
    await api.startCivitAIDownload({
      download_url: lora.source_url,
      filename: lora.filename,
      // architecture (not family) is what the backend's get_lora_dir keys on,
      // so the LoRA lands in the same per-model dir the model loads from.
      target_arch: (model?.architecture as string) || '',
      model_id: 0, version_id: 0, trained_words: [],
      model_name: lora.filename, images: [],
    })
    get().pollCivitAIDownloads()
  },
  loadDirectorFromPipeline: async (pid) => {
    try {
      const pipeline = await api.fetchSavedPipeline(pid)
      set({
        sidebarMode: 'director' as const,
        directorSceneDescription: pipeline.scene_description || '',
        directorClipPlans: pipeline.clips.map(c => ({
          video_prompt: c.video_prompt || '',
          image_prompt: c.image_prompt || '',
        })),
        directorClipImages: pipeline.clips
          .filter(c => c.start_image_filename)
          .map((c, i) => ({
            clipIndex: i,
            prompt: c.image_prompt || '',
            file: null as unknown as File,
            filename: c.start_image_filename!,
          })),
        directorStep: 'review_video',
        directorAutoMode: pipeline.auto_mode,
        directorSeamless: pipeline.seamless,
        directorShotImageGuidance: pipeline.shot_image_guidance || 'auto',
        dashboardOpen: true,
        dashboardSelectedPipeline: pipeline,
      })
    } catch (e) {
      console.error('Failed to load Director pipeline:', e)
    }
  },

  loraBrowserOpen: false,
  loraBrowserArch: null,
  loraBrowserDefaultDir: null,
  setLoraBrowserDefaultDir: (dir) => set({ loraBrowserDefaultDir: dir }),
  setLoraBrowserOpen: (open, arch) => {
    if (open) {
      set({ loraBrowserOpen: true, loraBrowserArch: arch || null, civitSearchResults: [], civitSearchCursor: null, civitSelectedModel: null })
      // Adopt downloads started by URL imports, recipes, or another browser
      // session instead of assuming this store initiated every transfer.
      get().pollCivitAIDownloads()
    } else {
      set({ loraBrowserOpen: false })
      // Refresh LoRA list after closing (may have downloaded new ones)
      const modelType = get().params.model_type
      if (modelType) get().loadLoras(modelType)
    }
  },
  civitSearchResults: [],
  civitSearchCursor: null,
  civitSearchLoading: false,
  civitSearchError: null,
  civitSelectedModel: null,
  civitDownloads: [],

  searchCivitAI: async (params, append = false) => {
    set({ civitSearchLoading: true, civitSearchError: null })
    try {
      const result = await api.searchCivitAI(params as Parameters<typeof api.searchCivitAI>[0])
      if (append) {
        set(s => ({
          civitSearchResults: [...s.civitSearchResults, ...result.items],
          civitSearchCursor: result.metadata?.nextCursor || null,
          civitSearchLoading: false,
        }))
      } else {
        set({
          civitSearchResults: result.items,
          civitSearchCursor: result.metadata?.nextCursor || null,
          civitSearchLoading: false,
          civitSelectedModel: null,
        })
      }
    } catch (e) {
      console.error('CivitAI search failed:', e)
      const msg = e instanceof Error ? e.message : 'CivitAI search failed'
      set({ civitSearchLoading: false, civitSearchError: msg })
    }
  },

  selectCivitAIModel: async (modelId) => {
    try {
      const model = await api.fetchCivitAIModel(modelId)
      set({ civitSelectedModel: model })
    } catch (e) {
      console.error('Failed to fetch model details:', e)
    }
  },

  clearCivitSelection: () => set({ civitSelectedModel: null }),

  startCivitAIDownload: async (params) => {
    try {
      await api.startCivitAIDownload(params as Parameters<typeof api.startCivitAIDownload>[0])
      get().pollCivitAIDownloads()
    } catch (e) {
      console.error('Download failed:', e)
    }
  },

  pollCivitAIDownloads: () => {
    // Mark every invocation, including calls made while the singleton loop is
    // awaiting an older request. The active loop consumes this before exit
    // and takes a new snapshot that was initiated after the caller arrived.
    _civitDownloadPollRequested = true
    if (_civitDownloadPollTask) return

    const controller = new AbortController()
    _civitDownloadPollController = controller
    const poll = async () => {
      let consecutiveErrors = 0
      try {
        while (!controller.signal.aborted) {
          _civitDownloadPollRequested = false
          try {
            const { downloads } = await api.fetchCivitAIDownloads()
            consecutiveErrors = 0
            set({ civitDownloads: downloads })

            // Checkpoint downloads are registered on the server before their
            // terminal record is published. Refresh here, in the singleton
            // poller, so navigating away from ModelDetail cannot skip it and
            // a Content-Disposition filename change cannot break matching.
            const completedCheckpoints = downloads.filter(download =>
              download.status === 'completed'
              && !!download.model_type
              && !_civitRefreshedCheckpointDownloads.has(download.id)
            )
            if (completedCheckpoints.length > 0) {
              try {
                await api.reloadModels()
                await get().loadModels()
                completedCheckpoints.forEach(download => {
                  _civitRefreshedCheckpointDownloads.add(download.id)
                })
              } catch (error) {
                // Keep the IDs unmarked so a later poll retries the refresh.
                console.warn('Checkpoint model refresh failed; will retry:', error)
              }
            }

            // A caller joined while this request was in flight. Its freshness
            // guarantee requires another request, even when this response has
            // no active/recent downloads and would normally end the loop.
            if (_civitDownloadPollRequested) continue

            // Keep taking snapshots while work is active and through the
            // completed row's 30-second display window. This guarantees a
            // caller that joins late still observes the terminal record.
            if (!downloads.some(download => _downloadNeedsPolling(download, Date.now()))) return
            await _waitForDownloadPoll(CIVIT_DOWNLOAD_POLL_MS, controller.signal)
          } catch (error) {
            if (controller.signal.aborted) return
            consecutiveErrors += 1
            if (_civitDownloadPollRequested) continue
            const knownWork = get().civitDownloads.some(download =>
              _downloadNeedsPolling(download, Date.now())
            )
            // Retry transient failures while the browser is open or known
            // work is active. A background adoption probe gets three retries
            // before yielding; a later caller can safely start a fresh loop.
            if (!get().loraBrowserOpen && !knownWork && consecutiveErrors > 3) {
              console.warn('Download polling paused after repeated errors:', error)
              return
            }
            const retryMs = Math.min(10_000, 1000 * (2 ** Math.min(consecutiveErrors - 1, 3)))
            await _waitForDownloadPoll(retryMs, controller.signal)
          }
        }
      } finally {
        if (_civitDownloadPollController === controller) {
          _civitDownloadPollController = null
          _civitDownloadPollTask = null
        }
      }
    }

    _civitDownloadPollTask = poll()
  },

  // Models & families
  families: [],
  models: [],
  modelsLoaded: false,
  enabledModels: _loadEnabledModels() ?? new Set(DEFAULT_ENABLED_MODELS),
  toggleModelEnabled: (modelType) => {
    _markMatureModelsInitialized(get().models, [modelType])
    set(s => {
      const next = new Set(s.enabledModels)
      if (next.has(modelType)) next.delete(modelType)
      else next.add(modelType)
      _saveEnabledModels(next)
      return { enabledModels: next }
    })
  },
  resetEnabledModels: () => {
    _markMatureModelsInitialized(get().models)
    const next = new Set(DEFAULT_ENABLED_MODELS)
    _saveEnabledModels(next)
    set({ enabledModels: next })
  },
  setAllModelsEnabled: (enabled) => {
    _markMatureModelsInitialized(get().models)
    if (enabled) {
      const all = new Set(get().models.map(m => m.model_type))
      _saveEnabledModels(all)
      set({ enabledModels: all })
    } else {
      const empty = new Set<string>()
      _saveEnabledModels(empty)
      set({ enabledModels: empty })
    }
  },
  setModelsEnabled: (modelTypes, enabled) => {
    _markMatureModelsInitialized(get().models, modelTypes)
    set(s => {
      const next = new Set(s.enabledModels)
      for (const mt of modelTypes) {
        if (enabled) next.add(mt)
        else next.delete(mt)
      }
      _saveEnabledModels(next)
      return { enabledModels: next }
    })
  },
  // Open Settings → Performance and ask the Enabled Models section to
  // expand + scroll to the given mode (fired by the ModelSelector hint).
  modelVisibilityFocus: null,
  openModelVisibility: (mode) => set({
    settingsOpen: true,
    settingsTab: 'performance',
    modelVisibilityFocus: mode,
  }),
  clearModelVisibilityFocus: () => set({ modelVisibilityFocus: null }),
  loadModels: async () => {
    try {
      const shouldHydrateVisibility = !_modelVisibilityHydrated
      const shouldHydrateH3WindowOverrides = !_h3WindowOverridesHydrated
      const [data, visibility, h3WindowPreferences] = await Promise.all([
        api.fetchModels(),
        shouldHydrateVisibility
          ? api.fetchModelVisibility().catch(error => {
              console.warn('Failed to load model visibility:', error)
              return null
            })
          : Promise.resolve(null),
        shouldHydrateH3WindowOverrides
          ? api.fetchH3WindowOverrides().catch(error => {
              console.warn('Failed to load H3 window overrides:', error)
              return null
            })
          : Promise.resolve(null),
      ])
      const families = data.families
      // Keep the complete backend capability record. Director publishes a
      // stricter per-workflow contract than Studio; rebuilding model objects
      // field-by-field used to discard that metadata and leave both Director
      // selectors empty even though compatible models were enabled.
      const backendModels: ModelDef[] = data.models.map(m => ({
        ...m,
        guidance_max_phases: m.guidance_max_phases ?? 1,
        fps: m.fps ?? 16,
        is_downloaded: m.is_downloaded ?? false,
        nsfw_only: m.nsfw_only ?? false,
      }))
      // Inject virtual SFX (MMAudio) models alongside backend models
      const models = [...backendModels, ...SFX_VIRTUAL_MODELS]

      if (shouldHydrateH3WindowOverrides && h3WindowPreferences) {
        _h3WindowOverridesHydrated = true
        set({ h3WindowOverrides: h3WindowPreferences.overrides || {} })
      }

      // Pinokio can assign a different web-server port on every launch.
      // Browser localStorage is origin-bound, so hydrate durable visibility
      // once from the server config and keep localStorage only as a
      // migration/cache layer.
      if (shouldHydrateVisibility && visibility) {
        let restoredModels: Set<string>
        if (visibility.configured) {
          restoredModels = new Set(visibility.enabled_models)
          _initializedMatureModels = new Set(
            visibility.initialized_mature_models,
          )
          _modelVisibilityDefaultsVersion = (
            visibility.defaults_version || 1
          )
        } else {
          const legacyModels = _loadEnabledModels()
          restoredModels = legacyModels ?? new Set(DEFAULT_ENABLED_MODELS)
          const legacyDefaultsVersion = parseInt(
            localStorage.getItem(DEFAULTS_VERSION_KEY) || '',
            10,
          )
          _modelVisibilityDefaultsVersion = legacyModels
            ? (legacyDefaultsVersion || 1)
            : DEFAULTS_VERSION
          // An existing browser whitelist is an explicit snapshot. Mark the
          // current Mature entries initialized so migration cannot re-enable
          // one the user deliberately disabled.
          _initializedMatureModels = legacyModels
            ? new Set(
                models
                  .filter(model => model.nsfw_only)
                  .map(model => model.model_type),
              )
            : new Set()
        }
        _modelVisibilityHydrated = true
        set({ enabledModels: restoredModels })
        if (!visibility.configured) _saveEnabledModels(restoredModels)
      }

      // One-time curated-defaults upgrade for existing installs (see
      // DEFAULTS_VERSION). Fresh installs already start from the full
      // DEFAULT_ENABLED_MODELS list; for them this only stamps the
      // version key.
      let migrateMusicDefault = false
      try {
        const storedVer = _modelVisibilityHydrated
          ? _modelVisibilityDefaultsVersion
          : (
              parseInt(
                localStorage.getItem(DEFAULTS_VERSION_KEY) || '1',
                10,
              ) || 1
            )
        if (storedVer < DEFAULTS_VERSION) {
          const additions: string[] = []
          for (let v = storedVer + 1; v <= DEFAULTS_VERSION; v++) {
            additions.push(...(DEFAULTS_ADDED_IN[v] || []))
          }
          const present = additions.filter(id => models.some(m => m.model_type === id))
          if (present.length > 0) {
            set(s => {
              const next = new Set(s.enabledModels)
              present.forEach(id => next.add(id))
              _saveEnabledModels(next)
              return { enabledModels: next }
            })
          }
          migrateMusicDefault = storedVer < 2
          _modelVisibilityDefaultsVersion = DEFAULTS_VERSION
          localStorage.setItem(DEFAULTS_VERSION_KEY, String(DEFAULTS_VERSION))
          _saveEnabledModels(get().enabledModels)
        }
      } catch { /* localStorage blocked — defaults only apply this session */ }

      // Hydrate persisted per-mode settings from localStorage.
      //
      // Deliberately PARTIAL: only the last generation mode and the
      // per-mode model selections survive a page refresh. The working
      // state — prompt text and Advanced settings (seed, steps, LoRA
      // selection, …) — starts fresh from the model's defaults on every
      // load. The per-mode snapshots (savedParamsPerMode /
      // savedLoraPerMode / savedPromptPerMode) still carry edits across
      // MODE SWITCHES within a session, in-memory only. v1.2.0 restored
      // them here on refresh; stale text/seeds/LoRAs re-appearing after
      // a reload felt wrong, so a refresh is a clean slate again.
      const saved = _loadSettings()
      // v2 migration: users whose saved audio model IS the old music
      // default follow it to the new default (see NEW_MUSIC_DEFAULT).
      // (The old-model-params concern the migration used to handle is
      // gone: saved params no longer rehydrate, and the defaults
      // hydration below runs on every boot.)
      if (migrateMusicDefault && saved?.selectedModelPerMode?.audio === OLD_MUSIC_DEFAULT
          && models.some(m => m.model_type === NEW_MUSIC_DEFAULT)) {
        saved.selectedModelPerMode = { ...saved.selectedModelPerMode, audio: NEW_MUSIC_DEFAULT }
      }
      let mode = get().generationMode
      let initialModelType: string

      if (saved) {
        // Restore saved generation mode
        mode = saved.generationMode || mode
        // Validate saved model for this mode still exists
        let savedModel = saved.selectedModelPerMode?.[mode]
        initialModelType = savedModel && models.some(m => m.model_type === savedModel)
          ? savedModel
          : getDefaultModelForMode(mode, families, models)
        const bootedIntoRecast = mode === 'avatar'
          && (initialModelType === 'scail2_14B_recast_fast' || initialModelType === 'scail2_14B')
        const bootedIntoRepaint = mode === 'avatar'
          && initialModelType === 'scail2_14B_fast'

        set(s => ({
          families,
          models,
          modelsLoaded: true,
          generationMode: mode,
          ...(bootedIntoRecast
            ? { editSubMode: 'recast' as const }
            : bootedIntoRepaint
              ? { editSubMode: 'restyle' as const }
              : {}),
          // Seed the VALIDATED boot model into the map (the saved entry
          // may point at a removed model) — _applyModelDefaults' race
          // guard compares against selectedModelPerMode[mode].
          selectedModelPerMode: { ...(saved.selectedModelPerMode || {}), [mode]: initialModelType },
          // Mode-shaping mirrored from setGenerationMode: booting into
          // image mode needs image_mode 1 + Auto resolution. These used
          // to arrive via the restored params snapshot.
          ...(mode === 'image' ? { resolutionPreset: 'auto' as ResolutionPreset, aspectRatio: 'auto' as AspectRatio } : {}),
          params: {
            ...s.params,
            model_type: initialModelType || s.params.model_type,
            ...(mode === 'image' ? { image_mode: 1 } : {}),
          },
        }))
      } else {
        initialModelType = getDefaultModelForMode(mode, families, models)
        set(s => ({
          families,
          models,
          modelsLoaded: true,
          selectedModelPerMode: { [mode]: initialModelType },
          ...(mode === 'image' ? { resolutionPreset: 'auto' as ResolutionPreset, aspectRatio: 'auto' as AspectRatio } : {}),
          params: {
            ...s.params,
            model_type: initialModelType || s.params.model_type,
            ...(mode === 'image' ? { image_mode: 1 } : {}),
          },
        }))
      }

      // Load LoRAs, model options, and tuned defaults for the initial
      // model. The defaults hydration (steps, guidance, LM sampling…)
      // must run on every boot now that saved params don't rehydrate —
      // without it the sliders would show INITIAL_PARAMS' generic values
      // instead of the model's.
      const mt = initialModelType || get().params.model_type
      if (mt && !sfxModelTypes.has(mt)) {
        get().loadLoras(mt)
        get().loadModelOptions(mt)
        _applyModelDefaults(get, set, mt)
      }
      // Refresh the lora_id ↔ filename map from /installed and reconcile
      // any filename renames since save (LoRA version updates land here
      // transparently — saved weights/activations carry over to the new
      // filename without user intervention).
      get().refreshLoraIdMap()

      // Auto-enable each Mature model once, then preserve an explicit
      // disable. The initialized IDs live in the same server-side visibility
      // record, so a changing Pinokio port cannot reset this decision.
      const cfg = get().servicesConfig
      if (cfg?.nsfw_mode && _modelVisibilityHydrated) {
        set(s => {
          const next = _enableUninitializedMatureModels(
            models,
            s.enabledModels,
          )
          if (!next) return s
          _saveEnabledModels(next)
          return { enabledModels: next }
        })
      }
    } catch (e) {
      console.error('Failed to load models:', e)
    }
  },

  resolutionPreset: '720p',
  setResolutionPreset: (preset) => {
    const ratio = get().aspectRatio
    const resolution = resolveResolution(get().modelOptions, preset, ratio)
    set(s => ({
      resolutionPreset: preset,
      params: { ...s.params, resolution },
      h3WindowPlan: null,
    }))
  },

  aspectRatio: '16:9',
  setAspectRatio: (ratio) => {
    const preset = get().resolutionPreset
    const resolution = resolveResolution(get().modelOptions, preset, ratio)
    set(s => ({
      aspectRatio: ratio,
      params: { ...s.params, resolution },
      h3WindowPlan: null,
    }))
  },

  durationSeconds: 5,
  setDurationSeconds: (s) => {
    const options = get().modelOptions
    const fps = options?.fps ?? 16
    const minimum = Math.max(1, (options?.frames_minimum || fps) / fps)
    const nativeMaximum = options?.frames_maximum
      ? options.frames_maximum / fps
      : null
    const h3ReferenceSequence = (
      options?.omni_reference === true
      && get().params.minimax_h3_reference_sequence === true
    )
    const isH3 = String(options?.architecture || '').startsWith('minimax_h3')
    const isLtxSequence = options?.multi_window_sequence_controls === true
    const ltxMultiWindow = (
      isLtxSequence
      && get().params.ltx_multi_window === true
    )
    const ltxWindowDefaults = options?.sliding_window_defaults
    const ltxSinglePassMaximum = isLtxSequence
      ? (ltxWindowDefaults?.window_max ?? Math.round(20 * fps)) / fps
      : null
    const h3FirstLastMultiWindow = (
      isH3
      && options?.omni_reference !== true
      && get().params.minimax_h3_multi_window === true
    )
    const h3SingleNativePass = (
      isH3
      && !h3ReferenceSequence
      && (options?.omni_reference === true || !h3FirstLastMultiWindow)
    )
    const maximum = isH3
      ? (h3ReferenceSequence || h3FirstLastMultiWindow
          ? Number.POSITIVE_INFINITY
          : (nativeMaximum ?? Number.POSITIVE_INFINITY))
      : isLtxSequence
        ? (ltxMultiWindow
            ? Number.POSITIVE_INFINITY
            : (ltxSinglePassMaximum ?? Number.POSITIVE_INFINITY))
      : (options?.sliding_window || nativeMaximum == null
          ? Number.POSITIVE_INFINITY
          : nativeMaximum)
    let seconds = Math.min(maximum, Math.max(minimum, s))
    if (
      options?.sliding_window
      && nativeMaximum
      && seconds <= Math.round(nativeMaximum * 10) / 10
    ) {
      seconds = Math.min(seconds, nativeMaximum)
    }
    let frames = Math.round(seconds * fps)
    if (h3SingleNativePass) {
      frames = normalizeH3NativeFrames(
        frames,
        options?.frames_minimum ?? 124,
        options?.frames_maximum ?? 345,
        options?.frames_steps ?? 17,
      )
      seconds = frames / fps
    }
    set(state => {
      const currentWindowFrames = Math.round(state.slidingWindowSeconds * fps)
      const expandNativeWindow = h3SingleNativePass && frames > currentWindowFrames
      const nextParams = {
        ...state.params,
        video_length: frames,
        ...(expandNativeWindow
          ? {
              sliding_window_size: frames,
              sliding_window_memory_override: true,
              ...(state.modelOptions?.omni_reference === true
                ? { minimax_h3_sequence_memory_override: true }
                : {}),
            }
          : {}),
      }
      delete nextParams.ltx_window_prompts
      return {
        durationSeconds: seconds,
        ...(expandNativeWindow
          ? {
              slidingWindowSeconds: seconds,
              slidingWindowLocked: true,
            }
          : {}),
        params: nextParams,
        h3WindowPlan: null,
        promptEnhanceError: null,
      }
    })
    get().syncClipCount()
  },

  guideVideoFps: null,
  setGuideVideoFps: (fps) => set({ guideVideoFps: fps }),

  slidingWindowSeconds: 5,
  setSlidingWindowSeconds: (s) => {
    const options = get().modelOptions
    const fps = options?.fps ?? 16
    const swDefaults = options?.sliding_window_defaults
    let frames = Math.round(s * fps)
    if (String(options?.architecture || '').startsWith('minimax_h3')) {
      frames = normalizeH3NativeFrames(
        frames,
        options?.frames_minimum ?? 124,
        options?.frames_maximum ?? 345,
        options?.frames_steps ?? 17,
      )
    } else if (swDefaults) {
      const minimum = swDefaults.window_min ?? 1
      const maximum = swDefaults.window_max ?? frames
      const step = Math.max(1, swDefaults.window_step ?? 1)
      frames = minimum + Math.round((frames - minimum) / step) * step
      frames = Math.max(minimum, Math.min(maximum, frames))
    }
    const seconds = frames / fps
    set(state => {
      const nextParams = {
        ...state.params,
        sliding_window_size: frames,
        ...(
          state.modelOptions?.omni_reference === true
          && state.params.minimax_h3_reference_sequence === true
            ? { minimax_h3_sequence_clip_frames: frames }
            : {}
        ),
      }
      delete nextParams.ltx_window_prompts
      return {
        slidingWindowSeconds: seconds,
        params: nextParams,
        h3WindowPlan: null,
        promptEnhanceError: null,
      }
    })
    get().syncClipCount()
  },

  slidingWindowOverlap: 5,
  setSlidingWindowOverlap: (frames) => {
    set(state => {
      const normalized = _normalizeSlidingWindowOverlap(
        frames,
        state.modelOptions?.sliding_window_defaults,
      )
      const nextParams = { ...state.params, sliding_window_overlap: normalized }
      delete nextParams.ltx_window_prompts
      return {
        slidingWindowOverlap: normalized,
        params: nextParams,
        h3WindowPlan: null,
        promptEnhanceError: null,
      }
    })
  },
  slidingWindowLocked: false,
  setSlidingWindowLocked: (locked) => set(state => {
    const isH3 = String(state.modelOptions?.architecture || '').startsWith('minimax_h3')
    return {
      slidingWindowLocked: locked,
      params: isH3 ? {
          ...state.params,
          sliding_window_memory_override: locked,
          ...(state.modelOptions?.omni_reference === true
            ? { minimax_h3_sequence_memory_override: locked }
            : {}),
      } : state.params,
      h3WindowPlan: null,
      promptEnhanceError: null,
    }
  }),
  h3WindowOverrides: {},
  saveH3WindowOverride: (modelType, resolution, frames) => {
    const state = get()
    const minimum = state.modelOptions?.frames_minimum ?? 124
    const maximum = state.modelOptions?.frames_maximum ?? 345
    const step = state.modelOptions?.frames_steps ?? 17
    const normalizedFrames = normalizeH3NativeFrames(
      frames,
      minimum,
      maximum,
      step,
    )
    const key = h3WindowOverrideKey(modelType, resolution)
    const next = { ...state.h3WindowOverrides, [key]: normalizedFrames }
    set({ h3WindowOverrides: next })
    _saveH3WindowOverrides(next)
  },
  clearH3WindowOverride: (modelType, resolution) => {
    const state = get()
    const key = h3WindowOverrideKey(modelType, resolution)
    if (!(key in state.h3WindowOverrides)) return
    const next = { ...state.h3WindowOverrides }
    delete next[key]
    set({ h3WindowOverrides: next })
    _saveH3WindowOverrides(next)
  },

  outputCount: 1,
  setOutputCount: (n) => set(s => ({
    outputCount: n,
    params: { ...s.params, repeat_generation: n },
  })),

  startImage: null,
  endImage: null,
  setStartImage: (f) => set(s => ({
    startImage: f,
    params: f === null ? { ...s.params, image_start: undefined } : s.params,
    h3WindowPlan: null,
  })),
  setEndImage: (f) => set(s => ({
    endImage: f,
    params: f === null ? { ...s.params, image_end: undefined } : s.params,
    h3WindowPlan: null,
  })),

  // Image references
  imageRefs: [],
  imageRefType: '',
  removeBackgroundRefs: false,
  addImageRef: (file) => set(s => ({ imageRefs: [...s.imageRefs, file] })),
  removeImageRef: (index) => set(s => {
    const updated = s.imageRefs.filter((_, i) => i !== index)
    return {
      imageRefs: updated,
      params: updated.length === 0 ? { ...s.params, image_refs: undefined } : s.params,
    }
  }),
  reorderImageRefs: (from, to) => set(s => {
    const refs = [...s.imageRefs]
    const [moved] = refs.splice(from, 1)
    refs.splice(to, 0, moved)
    return { imageRefs: refs }
  }),
  setImageRefType: (type) => set({ imageRefType: type }),
  setRemoveBackgroundRefs: (v) => set({ removeBackgroundRefs: v }),

  // Voice clone postprocessing state — defaults are off / empty so
  // existing generations are unaffected.
  voiceCloneEnabled: false,
  setVoiceCloneEnabled: (v) => set({ voiceCloneEnabled: v }),
  voiceCloneMode: 'single',
  setVoiceCloneMode: (v) => set({ voiceCloneMode: v }),
  voiceCloneRefs: [],
  setVoiceCloneRef: (index, ref) => set(s => {
    const next = [...s.voiceCloneRefs]
    if (ref === null) {
      next.splice(index, 1)
    } else {
      while (next.length <= index) next.push({ filename: '', path: '' })
      next[index] = ref
    }
    return { voiceCloneRefs: next }
  }),

  // ── Tools area (standalone post-processing on an existing clip) ──────
  toolsTool: 'upscale',
  setToolsTool: (t) => set({ toolsTool: t }),
  toolsSourcePath: null,
  toolsSourceName: null,
  toolsSourceUrl: null,
  setToolsSource: (src) => set(src
    ? { toolsSourcePath: src.path, toolsSourceName: src.name, toolsSourceUrl: src.url }
    : { toolsSourcePath: null, toolsSourceName: null, toolsSourceUrl: null }),
  toolsUpscaleMethod: 'flashvsr2',
  setToolsUpscaleMethod: (m) => set({ toolsUpscaleMethod: m }),
  toolsRevoiceMode: 'single',
  setToolsRevoiceMode: (m) => set({ toolsRevoiceMode: m }),
  toolsRevoiceRefs: [null, null],
  setToolsRevoiceRef: (index, ref) => set(s => {
    const next = [...s.toolsRevoiceRefs]
    while (next.length <= index) next.push(null)
    next[index] = ref
    return { toolsRevoiceRefs: next }
  }),
  runTool: async () => {
    const s = get()
    const source = s.toolsSourcePath
    if (!source) return
    const tool = s.toolsTool

    // Revoice needs at least one resolved voice reference.
    const refPaths = s.toolsRevoiceRefs
      .filter((r): r is { filename: string; path: string } => !!r && !!r.path)
      .map(r => r.path)
    if (tool === 'revoice' && refPaths.length === 0) return

    // Placeholder job tile — mirrors the blend/edit submit pattern so the
    // progress shows in the main feed and the gallery refreshes on completion.
    const newJob: GenerationJob = {
      id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
      phase: '', message: tool === 'upscale' ? 'Submitting upscale...' : 'Submitting revoice...',
      outputFiles: [], error: null, oomInfo: null,
    }
    set(st => ({ isGenerating: true, jobs: [newJob, ...st.jobs] }))

    try {
      const result = tool === 'upscale'
        ? await api.submitToolUpscale({ video_path: source, method: s.toolsUpscaleMethod, workspace: s.activeWorkspace })
        : await api.submitToolRevoice({ video_path: source, voice_ref_paths: refPaths, mode: s.toolsRevoiceMode, workspace: s.activeWorkspace })

      set(st => ({
        jobs: st.jobs.map(j => j === newJob ? { ...j, id: result.job_id, status: 'running', message: tool === 'upscale' ? 'Upscaling...' : 'Replacing voice...' } : j),
      }))

      const pollInterval = setInterval(async () => {
        if (!get().jobs.find(j => j.id === result.job_id)) { clearInterval(pollInterval); return }
        try {
          const status = await api.fetchJobStatus(result.job_id)
          set(st => ({
            jobs: st.jobs.map(j => j.id !== result.job_id ? j : {
              ...j, status: status.status, progress: status.progress / 100,
              step: status.step, totalSteps: status.total_steps,
              phase: status.phase, message: status.message,
              outputFiles: status.output_files, error: status.error, oomInfo: status.oom_info ?? null,
            }),
          }))
          if (status.status === 'running') get().refreshOutputs()
          if (status.status === 'completed') {
            clearInterval(pollInterval)
            set(st => {
              const remaining = st.jobs.filter(j => j.id !== result.job_id)
              return { jobs: remaining, isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued') }
            })
            get().loadOutputs()
          } else if (status.status === 'failed' || status.status === 'cancelled') {
            clearInterval(pollInterval)
            set(st => ({ isGenerating: st.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')) }))
          }
        } catch { /* ignore poll errors */ }
      }, 2000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : (tool === 'upscale' ? 'Upscale failed' : 'Revoice failed')
      set(st => ({
        jobs: st.jobs.map(j => j === newJob ? { ...j, id: j.id || `tool-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
        isGenerating: st.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
      }))
      console.error(`Tool ${tool} failed:`, msg)
    }
  },
  quickUpscaleClip: async (name, url) => {
    // Point the Tools state at this clip and run an upscale immediately,
    // reusing runTool()'s submit+poll. The Tools panel reflects this clip
    // afterward (harmless — and convenient if the user opens it).
    set({ toolsTool: 'upscale', toolsSourcePath: name, toolsSourceName: name, toolsSourceUrl: url })
    await get().runTool()
  },
  sendClipToTools: (name, url, tool) => {
    set({ toolsTool: tool, toolsSourcePath: name, toolsSourceName: name, toolsSourceUrl: url })
    get().setGenerationMode('tools')
  },

  // Post-processing defaults (shared for Studio)
  spatialUpsampling: '',
  setSpatialUpsampling: (v) => set({ spatialUpsampling: v }),
  filmGrainIntensity: 0,
  setFilmGrainIntensity: (v) => {
    set({ filmGrainIntensity: v })
    // Persist per mode
    const s = get()
    const mode = s.generationMode
    const updatedSavedParams = {
      ...s.savedParamsPerMode,
      [mode]: {
        num_inference_steps: s.params.num_inference_steps,
        guidance_scale: s.params.guidance_scale,
        resolution: s.params.resolution,
        seed: s.params.seed,
        filmGrainIntensity: v,
        filmGrainSaturation: s.filmGrainSaturation,
      },
    }
    set({ savedParamsPerMode: updatedSavedParams })
  },
  filmGrainSaturation: 0.5,
  setFilmGrainSaturation: (v) => {
    set({ filmGrainSaturation: v })
    const s = get()
    const mode = s.generationMode
    const updatedSavedParams = {
      ...s.savedParamsPerMode,
      [mode]: {
        num_inference_steps: s.params.num_inference_steps,
        guidance_scale: s.params.guidance_scale,
        resolution: s.params.resolution,
        seed: s.params.seed,
        filmGrainIntensity: s.filmGrainIntensity,
        filmGrainSaturation: v,
      },
    }
    set({ savedParamsPerMode: updatedSavedParams })
  },

  // Director-mode post-processing (separate image/video)
  directorImageSpatialUpsampling: '',
  setDirectorImageSpatialUpsampling: (v) => set({ directorImageSpatialUpsampling: v }),
  directorImageFilmGrainIntensity: 0,
  setDirectorImageFilmGrainIntensity: (v) => set({ directorImageFilmGrainIntensity: v }),
  directorImageFilmGrainSaturation: 0.5,
  setDirectorImageFilmGrainSaturation: (v) => set({ directorImageFilmGrainSaturation: v }),
  directorVideoSpatialUpsampling: '',
  setDirectorVideoSpatialUpsampling: (v) => set({ directorVideoSpatialUpsampling: v }),
  directorVideoFilmGrainIntensity: 0,
  setDirectorVideoFilmGrainIntensity: (v) => set({ directorVideoFilmGrainIntensity: v }),
  directorVideoFilmGrainSaturation: 0.5,
  setDirectorVideoFilmGrainSaturation: (v) => set({ directorVideoFilmGrainSaturation: v }),
  directorVideoSelfRefiner: 0,
  setDirectorVideoSelfRefiner: (v) => set({ directorVideoSelfRefiner: v }),
  directorAudioScale: 1.0,
  setDirectorAudioScale: (v) => set({ directorAudioScale: v }),

  audioGuideFilename: null,
  setAudioGuideFilename: (name) => set({ audioGuideFilename: name }),
  audioGuide2Filename: null,
  setAudioGuide2Filename: (name) => set({ audioGuide2Filename: name }),
  ttsSpeakerName1: '',
  ttsSpeakerName2: '',
  ttsSpeakerNamesManual: false,
  setTtsSpeakerName1: (name) => {
    set(s => {
      const voices = [...s.ttsVoices]
      if (voices.length > 0) voices[0] = { ...voices[0], name }
      return { ttsSpeakerName1: name, ttsSpeakerNamesManual: true, ttsVoices: voices }
    })
  },
  setTtsSpeakerName2: (name) => {
    set(s => {
      const voices = [...s.ttsVoices]
      if (voices.length > 1) voices[1] = { ...voices[1], name }
      return { ttsSpeakerName2: name, ttsSpeakerNamesManual: true, ttsVoices: voices }
    })
  },
  _autoParseSpkeakerNames: (text: string, force?: boolean) => {
    // The manual flag prevents auto-parse from clobbering names the user
    // explicitly typed. `force=true` overrides it — used by the enhance
    // button since enhance generates a fresh script whose new names should
    // replace whatever the user had previously set.
    if (!force && get().ttsSpeakerNamesManual) return
    // Match anything before ":" at the start of a line (e.g. "Dr. Mary Jane O'Brien:")
    const matches = text.match(/^(.+?)\s*:/gm)
    if (!matches) return
    const names = [...new Set(matches.map(m => m.replace(/\s*:$/, '').trim()))]
    const voiceCount = get().ttsVoiceCount
    const voices = [...get().ttsVoices]
    // Ensure voices array is big enough
    while (voices.length < voiceCount) {
      voices.push({ name: '', filename: null, path: null })
    }
    for (let i = 0; i < Math.min(names.length, voiceCount); i++) {
      voices[i] = { ...voices[i], name: names[i] }
    }
    set({
      ttsVoices: voices,
      ttsSpeakerName1: names[0] || '',
      ttsSpeakerName2: names[1] || '',
      // Force-call (from enhance) resets the manual flag so subsequent
      // prompt edits can also auto-parse again. Non-force calls preserve
      // the flag (user manually edited a name; keep their state).
      ...(force ? { ttsSpeakerNamesManual: false } : {}),
    })
  },
  // Dynamic multi-speaker (1-6 voices)
  ttsVoiceCount: 0,
  ttsVoices: [],
  setTtsVoiceCount: (count) => {
    const prevCount = get().ttsVoiceCount
    const current = get().ttsVoices
    const voices = [...current]
    while (voices.length < count) {
      voices.push({ name: '', filename: null, path: null })
    }
    // Derive audio_prompt_type from voice count using the model's own selection
    // list. KugelAudio's selection = ["", "A", "AB"] → 0→"", 1→"A", 2+→"AB".
    // Scenema's selection = ["", "A2", "AB2"] → 0→"", 1→"A2", 2+→"AB2".
    // Other (non-Scenema/Kugel) audio-only models keep the legacy ""/A/AB
    // mapping for backward compat.
    const selection = (get().modelOptions?.audio_prompt_type_sources?.selection as string[] | undefined) || ['', 'A', 'AB']
    const audioType = selection[Math.min(count, selection.length - 1)]
    set(s => ({
      ttsVoiceCount: count,
      ttsVoices: voices.slice(0, Math.max(count, voices.length)),
      params: { ...s.params, audio_prompt_type: audioType + ((s.params.audio_prompt_type as string || '').replace(/[^NV]/g, '')) },
    }))
    // If user added voices to an existing prompt (e.g. typed/pasted a
    // dialogue script first, THEN added voice slots), parse the names
    // from the prompt and populate the voice fields. setParam's auto-parse
    // only fires when the prompt CHANGES — without this, growing the slot
    // count after the prompt is set leaves names un-populated. Use
    // force=true so the manual flag (which may have been set by an earlier
    // name edit or by settings restore) doesn't suppress the parse —
    // adding voices is an explicit mode-change action that should re-derive
    // names from the current prompt.
    if (count > prevCount) {
      const prompt = get().params.prompt
      if (typeof prompt === 'string' && prompt.trim()) {
        get()._autoParseSpkeakerNames(prompt, true)
      }
    }
  },
  setTtsVoiceName: (index, name) => {
    set(s => {
      const voices = [...s.ttsVoices]
      if (index < voices.length) voices[index] = { ...voices[index], name }
      return {
        ttsVoices: voices,
        ttsSpeakerNamesManual: true,
        // Keep legacy fields in sync
        ...(index === 0 ? { ttsSpeakerName1: name } : {}),
        ...(index === 1 ? { ttsSpeakerName2: name } : {}),
      }
    })
  },
  setTtsVoiceFile: (index, filename, path) => {
    set(s => {
      const voices = [...s.ttsVoices]
      if (index < voices.length) voices[index] = { ...voices[index], filename, path }
      return {
        ttsVoices: voices,
        // Keep legacy fields in sync
        ...(index === 0 ? { audioGuideFilename: filename } : {}),
        ...(index === 1 ? { audioGuide2Filename: filename } : {}),
      }
    })
  },
  addTtsVoice: () => {
    const count = get().ttsVoiceCount
    // Respect the model's declared max (e.g. Scenema = 2, Kugel = 6).
    // Defaults to 6 if the model_def doesn't specify max_voice_count.
    const maxVoiceCount = ((get().modelOptions as { max_voice_count?: number } | null)?.max_voice_count) ?? 6
    if (count >= maxVoiceCount) return
    get().setTtsVoiceCount(count + 1)
  },
  removeTtsVoice: (index) => {
    set(s => {
      const voices = s.ttsVoices.filter((_, i) => i !== index)
      const newCount = Math.max(0, s.ttsVoiceCount - 1)
      // Same model-aware mapping as setTtsVoiceCount above.
      const selection = (s.modelOptions?.audio_prompt_type_sources?.selection as string[] | undefined) || ['', 'A', 'AB']
      const audioType = selection[Math.min(newCount, selection.length - 1)]
      return {
        ttsVoices: voices,
        ttsVoiceCount: newCount,
        ttsSpeakerName1: voices[0]?.name || '',
        ttsSpeakerName2: voices[1]?.name || '',
        audioGuideFilename: voices[0]?.filename || null,
        audioGuide2Filename: voices[1]?.filename || null,
        params: { ...s.params, audio_prompt_type: audioType + ((s.params.audio_prompt_type as string || '').replace(/[^NV]/g, '')) },
      }
    })
  },

  // Multi-clip state
  clips: [],
  singlePromptMode: false,
  setClipPrompt: (index, prompt) => {
    const clips = [...get().clips]
    if (clips[index]) {
      clips[index] = { ...clips[index], prompt }
      set({ clips })
    }
  },
  setClipStartImage: (index, file) => {
    const clips = [...get().clips]
    if (clips[index]) {
      clips[index] = { ...clips[index], startImage: file }
      set({ clips })
    }
  },
  setSinglePromptMode: (v) => set({ singlePromptMode: v }),
  syncClipCount: () => {
    const { params, durationSeconds, slidingWindowSeconds, slidingWindowOverlap, modelOptions } = get()
    if (params.image_mode !== 2) return
    const fps = modelOptions?.fps ?? 16
    const overlapSeconds = slidingWindowOverlap / fps
    const effectiveWindow = slidingWindowSeconds - overlapSeconds
    const count = effectiveWindow > 0
      ? Math.max(1, Math.ceil((durationSeconds - overlapSeconds) / effectiveWindow))
      : Math.max(1, Math.ceil(durationSeconds / slidingWindowSeconds))
    const current = get().clips
    if (count === current.length) return
    if (count > current.length) {
      const newClips = [...current]
      for (let i = current.length; i < count; i++) {
        newClips.push({ prompt: '', startImage: null, startImagePath: null, endImage: null, endImagePath: null })
      }
      set({ clips: newClips })
    } else {
      set({ clips: current.slice(0, count) })
    }
  },

  jobs: [],
  isGenerating: false,

  startGeneration: async () => {
    // Auto-unload LLM before GPU-heavy generation to free VRAM
    if (get().llmStatus?.loaded) {
      try {
        await api.unloadLlm()
        set({ llmStatus: { loaded: false, model_id: null, device: null, provider: '' } })
      } catch { /* best-effort */ }
    }

    const state = get()

    // Validate: i2v-only models require a start image — Video mode only.
    // Edit sub-modes supply their own source media and validate in their
    // own branches (Recast runs the i2v-only SCAIL-2 against a source
    // video + reference image; this guard silently ate its clicks).
    const isI2vOnly = state.modelOptions?.i2v_class && !state.modelOptions?.t2v_class
    const isOmniReference = state.modelOptions?.omni_reference === true
    const isH3Model = String(state.modelOptions?.architecture || '').startsWith('minimax_h3')
    const isLtxSequenceModel = state.modelOptions?.multi_window_sequence_controls === true
    const hasStartImage = state.startImage || state.params.image_start
    const hasMultiClipImages = state.clips.some(c => c.startImage || c.startImagePath)
    if (state.generationMode === 'video' && isI2vOnly && !isOmniReference && !hasStartImage && !hasMultiClipImages) {
      console.error('This model requires a start image')
      // Could show a toast/notification here in the future
      return
    }
    const omniReferences = state.params.minimax_h3_references ?? []
    if (
      state.generationMode === 'video'
      && isOmniReference
      && !omniReferences.some(reference => reference.type === 'image' || reference.type === 'video')
    ) {
      console.error('MiniMax H3 Omni Reference needs at least one image or video reference')
      return
    }

    // ── Video mode: Blend ──────────────────────────────────────────
    if (state.generationMode === 'video' && (state.params.image_mode as number) === 4) {
      if (!state.blendClipAPath || !state.blendClipBPath) return
      const prompt = (state.params.prompt as string || '').trim()

      const newJob: GenerationJob = {
        id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
        phase: '', message: 'Submitting blend...', outputFiles: [], error: null, oomInfo: null,
      }
      set(s => ({ isGenerating: true, jobs: [newJob, ...s.jobs] }))

      try {
        const result = await api.submitBlend({
          clip_a_path: state.blendClipAPath,
          clip_b_path: state.blendClipBPath,
          prompt: prompt || 'smooth natural transition between the two clips',
          model_type: state.params.model_type as string,
          blend_mode: state.blendMode,
          overlap_sec: state.blendOverlapSec,
          // Blend-specific tuning knobs (exposed in BlendControls sliders)
          motion_prefix_sec: state.blendMotionPrefixSec,
          motion_suffix_sec: state.blendMotionSuffixSec,
          input_video_strength: state.blendAnchorStrength,
          seed: (state.params.seed as number) ?? -1,
          activated_loras: (state.params.activated_loras as string[]) || [],
          loras_multipliers: (state.params.loras_multipliers as string) || '',
          workspace: state.activeWorkspace,
          // Pass the full Studio params so the backend can inherit the user's
          // progressive_pipeline / num_inference_steps / guidance_scale /
          // negative_prompt settings, matching what a manual SE generation
          // would have used. Blend-specific fields (image_start/end, video_length,
          // resolution, image_prompt_type) are overridden server-side.
          base_params: state.params as unknown as Record<string, unknown>,
        })

        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: result.job_id, status: 'running', message: 'Blending...' } : j),
        }))

        const pollInterval = setInterval(async () => {
          if (!get().jobs.find(j => j.id === result.job_id)) { clearInterval(pollInterval); return }
          try {
            const status = await api.fetchJobStatus(result.job_id)
            set(s => ({
              jobs: s.jobs.map(j => j.id !== result.job_id ? j : {
                ...j, status: status.status, progress: status.progress / 100,
                step: status.step, totalSteps: status.total_steps,
                phase: status.phase, message: status.message,
                outputFiles: status.output_files, error: status.error, oomInfo: status.oom_info ?? null,
              }),
            }))
            if (status.status === 'running') get().refreshOutputs()
            if (status.status === 'completed') {
              clearInterval(pollInterval)
              set(s => {
                const remaining = s.jobs.filter(j => j.id !== result.job_id)
                return {
                  jobs: remaining,
                  isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
                }
              })
              get().loadOutputs()
            } else if (status.status === 'failed' || status.status === 'cancelled') {
              clearInterval(pollInterval)
              // Keep the failed/cancelled job in the queue so its placeholder
              // stays visible with the error message — user dismisses via X.
              set(s => ({
                isGenerating: s.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')),
              }))
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Blend failed'
        // Submit itself failed (pre-queue). Convert the placeholder to a
        // failed state in place so the user sees what went wrong instead of
        // the tile silently disappearing.
        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
          isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
        }))
        console.error('Blend failed:', msg)
      }
      return
    }

    // ── Edit mode: Outpaint ────────────────────────────────────────
    if (state.generationMode === 'avatar' && state.editSubMode === 'outpaint') {
      if (!state.editVideoPath) return
      const prompt = (state.params.prompt as string || '').trim()

      // Resolve source pixel dimensions from the loaded video metadata.
      // We need them to convert the canvas-relative video box into absolute
      // pad_top/bottom/left/right pixel values that the server expects.
      const srcRes = state.editVideoResolution || ''
      const [srcWStr, srcHStr] = srcRes.split('x')
      const srcW = parseInt(srcWStr) || 0
      const srcH = parseInt(srcHStr) || 0
      if (srcW <= 0 || srcH <= 0) {
        console.error('Outpaint: source dimensions unknown')
        return
      }

      // Resolve canvas dimensions in source-pixel-space from the chosen aspect.
      // Canvas is grown so the source fits inside without cropping; pure
      // letterbox math.
      const aspect = state.outpaintAspect
      let canvasW = srcW, canvasH = srcH
      if (aspect !== 'source') {
        const [aw, ah] = aspect.split(':').map(Number)
        const target = aw / ah
        const srcRatio = srcW / srcH
        if (srcRatio > target) {
          canvasW = srcW
          canvasH = Math.round(srcW / target)
        } else {
          canvasH = srcH
          canvasW = Math.round(srcH * target)
        }
      }

      // The video box is canvas-relative (0–1). Convert to pixel pads.
      const box = state.outpaintVideoBox
      const videoX = Math.round(box.x * canvasW)
      const videoY = Math.round(box.y * canvasH)
      const videoW = Math.round(box.w * canvasW)
      const videoH = Math.round(box.h * canvasH)
      const padTop = Math.max(0, videoY)
      const padLeft = Math.max(0, videoX)
      const padBottom = Math.max(0, canvasH - videoY - videoH)
      const padRight = Math.max(0, canvasW - videoX - videoW)
      const totalPad = padTop + padBottom + padLeft + padRight
      if (totalPad === 0) return

      // Mirror the computed pads to outpaintPadding so metadata sidecars
      // and any older read paths still see the values.
      set({ outpaintPadding: { top: padTop, bottom: padBottom, left: padLeft, right: padRight } })

      // Optional film-strip trim: only send if user picked a non-trivial range.
      const trimStart = state.outpaintTrimStart || 0
      const trimEnd = state.outpaintTrimEnd || 0
      const sendTrim = trimEnd > trimStart && trimEnd > 0.05

      const newJob: GenerationJob = {
        id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
        phase: '', message: 'Submitting outpaint...', outputFiles: [], error: null, oomInfo: null,
      }
      set(s => ({ isGenerating: true, jobs: [newJob, ...s.jobs] }))

      // Sliding window size: the Advanced Settings slider stores seconds.
      // Convert to frames using the loaded model's fps so the same value
      // round-trips between video and outpaint modes. Falls back to 25
      // (LTX-2 22B's native rate) if modelOptions hasn't loaded yet.
      const fps = (state.modelOptions?.fps as number) || 25
      const windowFrames = Math.max(1, Math.round(state.slidingWindowSeconds * fps))
      const overlapFrames = state.slidingWindowOverlap || 9

      try {
        const result = await api.submitOutpaint({
          video_path: state.editVideoPath,
          prompt: prompt || 'extend the scene naturally',
          model_type: state.params.model_type as string,
          pad_top: padTop,
          pad_bottom: padBottom,
          pad_left: padLeft,
          pad_right: padRight,
          outpaint_aspect: state.outpaintAspect,
          resolution_preset: state.outpaintResolutionPreset,
          source_preservation: 1.0,
          outpaint_lora_strength: 1.0,
          mask_preserving_outpaint: state.outpaintMaskPreserving,
          preserve_source_audio: state.outpaintPreserveSourceAudio,
          lock_source_pixels: false,
          trim_window_smear: state.outpaintTrimSmear,
          sliding_window_size: windowFrames,
          sliding_window_overlap: overlapFrames,
          ...(sendTrim ? { start_time: trimStart, end_time: trimEnd } : {}),
          num_inference_steps: (state.params.num_inference_steps as number) ?? undefined,
          guidance_scale: (state.params.guidance_scale as number) ?? undefined,
          negative_prompt: (state.params.negative_prompt as string) || undefined,
          seed: (state.params.seed as number) ?? -1,
          activated_loras: (state.params.activated_loras as string[]) || [],
          loras_multipliers: (state.params.loras_multipliers as string) || '',
          workspace: state.activeWorkspace,
        })

        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: result.job_id, status: 'running', message: 'Outpainting...' } : j),
        }))

        const pollInterval = setInterval(async () => {
          if (!get().jobs.find(j => j.id === result.job_id)) { clearInterval(pollInterval); return }
          try {
            const status = await api.fetchJobStatus(result.job_id)
            set(s => ({
              jobs: s.jobs.map(j => j.id !== result.job_id ? j : {
                ...j, status: status.status, progress: status.progress / 100,
                step: status.step, totalSteps: status.total_steps,
                phase: status.phase, message: status.message,
                outputFiles: status.output_files, error: status.error, oomInfo: status.oom_info ?? null,
              }),
            }))
            if (status.status === 'running') get().refreshOutputs()
            if (status.status === 'completed') {
              clearInterval(pollInterval)
              set(s => {
                const remaining = s.jobs.filter(j => j.id !== result.job_id)
                return {
                  jobs: remaining,
                  isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
                }
              })
              get().loadOutputs()
            } else if (status.status === 'failed' || status.status === 'cancelled') {
              clearInterval(pollInterval)
              // Keep the failed/cancelled job in the queue so its placeholder
              // stays visible with the error message — user dismisses via X.
              set(s => ({
                isGenerating: s.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')),
              }))
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Outpaint failed'
        // Submit itself failed (pre-queue). Convert the placeholder to a
        // failed state in place so the user sees what went wrong instead of
        // the tile silently disappearing.
        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
          isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
        }))
        console.error('Outpaint failed:', msg)
      }
      return
    }

    // ── Edit mode: Recast (SCAIL-2 Replace) ─────────────────────
    // Standalone branch: the prompt is OPTIONAL here (the server has a
    // sensible default), unlike the shared edit block below which
    // hard-requires one.
    // Repaint is the easy front door to the proven Studio Video/Frames
    // SCAIL-2 Animate path: an edited first frame defines the finished look
    // while the source video supplies motion and camera movement.
    if (state.generationMode === 'avatar' && state.editSubMode === 'restyle') {
      if (!state.editVideoPath || !state.editRepaintFramePath) return
      const repaintMappings = state.editRepaintMappings.slice(0, 5)
      if (repaintMappings.some(mapping => !mapping.source.trim() || !mapping.target.trim())) return
      const promptText = ((state.params.prompt as string) || '').trim()
      const newJob: GenerationJob = {
        id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
        phase: '', message: 'Submitting repaint...', outputFiles: [], error: null, oomInfo: null,
      }
      set(s => ({ isGenerating: true, jobs: [newJob, ...s.jobs] }))

      try {
        const repaintModel = (state.params.model_type as string) || ''
        const repaintIsScail2 = repaintModel === 'scail2_14B_fast' || repaintModel === 'scail2_14B'
        const result = await api.submitRepaint({
          video_path: state.editVideoPath,
          target_frame_path: state.editRepaintFramePath,
          region_mappings: repaintMappings.map(mapping => ({
            id: mapping.id,
            source: mapping.source.trim(),
            target: mapping.target.trim(),
          })),
          ...(promptText ? { prompt: promptText } : {}),
          resolution_profile: state.editRepaintResolutionProfile,
          ...(repaintIsScail2 ? {
            model_type: repaintModel,
            num_inference_steps: (state.params.num_inference_steps as number) ?? undefined,
            ...(repaintModel === 'scail2_14B' ? {
              guidance_scale: (state.params.guidance_scale as number) ?? undefined,
            } : {}),
          } : {}),
          start_time: state.editStartTime,
          end_time: state.editEndTime,
          seed: (state.params.seed as number) ?? -1,
          negative_prompt: (state.params.negative_prompt as string) || '',
          activated_loras: (state.params.activated_loras as string[]) || [],
          loras_multipliers: (state.params.loras_multipliers as string) || '',
          workspace: state.activeWorkspace,
        })

        set(s => ({
          jobs: s.jobs.map(j => j === newJob
            ? { ...j, id: result.job_id, status: 'running', message: 'Queued...' }
            : j),
        }))

        const pollInterval = setInterval(async () => {
          if (!get().jobs.find(j => j.id === result.job_id)) {
            clearInterval(pollInterval)
            return
          }
          try {
            const status = await api.fetchJobStatus(result.job_id)
            set(s => ({
              jobs: s.jobs.map(j => j.id !== result.job_id ? j : {
                ...j,
                status: status.status,
                progress: status.progress / 100,
                step: status.step,
                totalSteps: status.total_steps,
                phase: status.phase,
                message: status.message,
                outputFiles: status.output_files,
                error: status.error,
                oomInfo: status.oom_info ?? null,
              }),
            }))
            if (status.status === 'running') get().refreshOutputs()
            if (status.status === 'completed') {
              clearInterval(pollInterval)
              set(s => {
                const remaining = s.jobs.filter(j => j.id !== result.job_id)
                return {
                  jobs: remaining,
                  isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
                }
              })
              get().loadOutputs()
            } else if (status.status === 'failed' || status.status === 'cancelled') {
              clearInterval(pollInterval)
              set(s => ({
                isGenerating: s.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')),
              }))
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Repaint failed'
        set(s => ({
          jobs: s.jobs.map(j => j === newJob
            ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg }
            : j),
          isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
        }))
        console.error('Repaint failed:', msg)
      }
      return
    }

    if (state.generationMode === 'avatar' && state.editSubMode === 'recast') {
      const recastMappings = state.editRecastMappings.slice(0, 5)
      if (
        !state.editVideoPath
        || recastMappings.length === 0
        || recastMappings.some(mapping => !mapping.target.trim() || !mapping.refPath)
      ) return
      const promptText = ((state.params.prompt as string) || '').trim()

      const newJob: GenerationJob = {
        id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
        phase: '', message: 'Submitting recast...', outputFiles: [], error: null, oomInfo: null,
      }
      set(s => ({ isGenerating: true, jobs: [newJob, ...s.jobs] }))

      try {
        // Honor the selector's Recast SCAIL-2 choice (dedicated Fast vs
        // native base). Guard on
        // architecture so a stale LTX model_type can never reach the
        // recast endpoint — the server then falls back to Fast.
        const recastModel = (state.params.model_type as string) || ''
        const recastIsScail2 = state.models.find(m => m.model_type === recastModel)?.architecture === 'scail2_14B'
        const result = await api.submitRecast({
          video_path: state.editVideoPath,
          // Legacy fields remain populated for old sidecars/API clients, while
          // the explicit cards provide deterministic target/color assignment.
          ref_image_path: recastMappings[0].refPath,
          target: recastMappings[0].target || 'person',
          person_count: recastMappings.length,
          reference_aligned_to_source: recastMappings[0].referenceAlignedToSource,
          character_mappings: recastMappings.map(mapping => ({
            id: mapping.id,
            target: mapping.target.trim(),
            ref_image_path: mapping.refPath,
            additional_ref_image_paths: mapping.additionalRefs
              .map(reference => reference.path)
              .filter(Boolean),
            reference_aligned_to_source: mapping.referenceAlignedToSource,
          })),
          // Simplified Recast recipe: identity preparation and native
          // bystander preservation are automatic; prompt rewriting and the
          // seam-prone post-composite remain off. The backend still accepts
          // all legacy fields for saved/API callers.
          isolate_reference: true,
          auto_face_detail: true,
          enhance_prompt: false,
          protect_bystanders: false,
          preserve_bystanders: true,
          use_relighting: state.editRecastUseRelighting,
          resolution_profile: state.editRecastResolutionProfile,
          ...(promptText ? { prompt: promptText } : {}),
          ...(recastIsScail2 ? {
            model_type: recastModel,
            num_inference_steps: (state.params.num_inference_steps as number) ?? undefined,
            ...(recastModel === 'scail2_14B' ? {
              guidance_scale: (state.params.guidance_scale as number) ?? undefined,
            } : {}),
          } : {}),
          start_time: state.editStartTime,
          end_time: state.editEndTime,
          seed: (state.params.seed as number) ?? -1,
          negative_prompt: (state.params.negative_prompt as string) || '',
          activated_loras: (state.params.activated_loras as string[]) || [],
          loras_multipliers: (state.params.loras_multipliers as string) || '',
          workspace: state.activeWorkspace,
        })

        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: result.job_id, status: 'running', message: 'Queued...' } : j),
        }))

        const pollInterval = setInterval(async () => {
          if (!get().jobs.find(j => j.id === result.job_id)) { clearInterval(pollInterval); return }
          try {
            const status = await api.fetchJobStatus(result.job_id)
            set(s => ({
              jobs: s.jobs.map(j => j.id !== result.job_id ? j : {
                ...j, status: status.status, progress: status.progress / 100,
                step: status.step, totalSteps: status.total_steps,
                phase: status.phase, message: status.message,
                outputFiles: status.output_files, error: status.error, oomInfo: status.oom_info ?? null,
              }),
            }))
            if (status.status === 'running') get().refreshOutputs()
            if (status.status === 'completed') {
              clearInterval(pollInterval)
              set(s => {
                const remaining = s.jobs.filter(j => j.id !== result.job_id)
                return {
                  jobs: remaining,
                  isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
                }
              })
              get().loadOutputs()
            } else if (status.status === 'failed' || status.status === 'cancelled') {
              clearInterval(pollInterval)
              set(s => ({
                isGenerating: s.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')),
              }))
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Recast failed'
        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
          isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
        }))
        console.error('Recast failed:', msg)
      }
      return
    }

    // ── Edit mode: Retake / Inpaint / Edit Anything ─────────────
    if (state.generationMode === 'avatar' && (state.editSubMode === 'retake' || state.editSubMode === 'inpaint' || state.editSubMode === 'edit_anything')) {
      if (!state.editVideoPath) return
      const prompt = (state.params.prompt as string || '').trim()
      if (!prompt) return

      const newJob: GenerationJob = {
        id: '', status: 'queued', progress: 0, step: 0, totalSteps: 0,
        phase: '', message: 'Submitting...', outputFiles: [], error: null, oomInfo: null,
      }
      set(s => ({ isGenerating: true, jobs: [newJob, ...s.jobs] }))

      try {
        let result: { job_id: string }
        if (state.editSubMode === 'edit_anything') {
          result = await api.submitEditAnything({
            video_path: state.editVideoPath,
            prompt,
            model_type: state.params.model_type as string,
            start_time: state.editStartTime,
            end_time: state.editEndTime,
            lora_strength: state.editAnythingLoraStrength,
            retake_strength: state.editRetakeStrength,
            seed: (state.params.seed as number) ?? -1,
            // Edit Anything LoRA card: start with CFG=1 on distilled; raise
            // only if the edit is too weak. We route the user's global CFG
            // slider through so they can experiment.
            guidance_scale: (state.params.guidance_scale as number) ?? 1.0,
            num_inference_steps: (state.params.num_inference_steps as number) ?? 8,
            negative_prompt: (state.params.negative_prompt as string) || '',
            activated_loras: (state.params.activated_loras as string[]) || [],
            loras_multipliers: (state.params.loras_multipliers as string) || '',
            workspace: state.activeWorkspace,
            // Optional boundary anchors. Empty values mean "use source
            // frames" (today's auto-extract behavior); ltx2.py treats
            // missing/null/empty path as "fall back to source".
            ...(state.editAnythingStartAnchor ? { start_anchor_path: state.editAnythingStartAnchor } : {}),
            ...(state.editAnythingEndAnchor ? { end_anchor_path: state.editAnythingEndAnchor } : {}),
          })
        } else if (state.editSubMode === 'inpaint') {
          result = await api.submitInpaint({
            video_path: state.editVideoPath,
            description: prompt,
            sam_target: state.editSamTarget || undefined,
            invert_mask: state.editInvertMask || undefined,
            start_time: state.editStartTime,
            end_time: state.editEndTime,
            model_type: state.params.model_type as string,
            seed: (state.params.seed as number) ?? -1,
            // Inpaint needs CFG > 1.0 to make the prompt actually influence
            // the masked region. The edit-specific editPromptStrength slider
            // (default 3.5) drives this; the global params.guidance_scale is
            // fine for normal generation but would silently default to 1.0
            // and silently break inpaint.
            guidance_scale: state.editPromptStrength,
            retake_strength: state.editRetakeStrength,
            num_inference_steps: (state.params.num_inference_steps as number) ?? 8,
            negative_prompt: (state.params.negative_prompt as string) || '',
            resolution: (state.params.resolution as string) || '',
            activated_loras: (state.params.activated_loras as string[]) || [],
            loras_multipliers: (state.params.loras_multipliers as string) || '',
            masks_path: state.editMasksPath || undefined,
            workspace: state.activeWorkspace,
          })
        } else {
          result = await api.submitRetake({
            video_path: state.editVideoPath,
            start_time: state.editStartTime,
            end_time: state.editEndTime,
            prompt,
            model_type: state.params.model_type as string,
            retake_strength: state.editRetakeStrength,
            retake_engine: state.editRetakeEngine,
            regenerate_audio: state.editRegenerateAudio,
            seed: (state.params.seed as number) ?? -1,
            // Retake also benefits from CFG > 1.0 when the user provides a
            // prompt that should drive the regenerated region (e.g. new
            // outfit, different style). Previously stuck at 1.0 via
            // params.guidance_scale fallback — same silent bug as inpaint.
            guidance_scale: state.editPromptStrength,
            num_inference_steps: (state.params.num_inference_steps as number) ?? 8,
            negative_prompt: (state.params.negative_prompt as string) || '',
            resolution: (state.params.resolution as string) || '',
            activated_loras: (state.params.activated_loras as string[]) || [],
            loras_multipliers: (state.params.loras_multipliers as string) || '',
            workspace: state.activeWorkspace,
          })
        }

        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: result.job_id, status: 'running', message: 'Queued...' } : j),
        }))

        // Standard job polling (same as regular generation)
        const pollInterval = setInterval(async () => {
          if (!get().jobs.find(j => j.id === result.job_id)) { clearInterval(pollInterval); return }
          try {
            const status = await api.fetchJobStatus(result.job_id)
            set(s => ({
              jobs: s.jobs.map(j => j.id !== result.job_id ? j : {
                ...j, status: status.status, progress: status.progress / 100,
                step: status.step, totalSteps: status.total_steps,
                phase: status.phase, message: status.message,
                outputFiles: status.output_files, error: status.error, oomInfo: status.oom_info ?? null,
              }),
            }))
            if (status.status === 'running') get().refreshOutputs()
            if (status.status === 'completed') {
              clearInterval(pollInterval)
              set(s => {
                const remaining = s.jobs.filter(j => j.id !== result.job_id)
                return {
                  jobs: remaining,
                  isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
                }
              })
              get().loadOutputs()
            } else if (status.status === 'failed' || status.status === 'cancelled') {
              clearInterval(pollInterval)
              // Keep the failed/cancelled job in the queue so its placeholder
              // stays visible with the error message — user dismisses via X.
              set(s => ({
                isGenerating: s.jobs.some(j => j.id !== result.job_id && (j.status === 'running' || j.status === 'queued')),
              }))
            }
          } catch { /* ignore poll errors */ }
        }, 2000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Generation failed'
        // Submit itself failed (pre-queue). Convert the placeholder to a
        // failed state in place so the user sees what went wrong instead of
        // the tile silently disappearing.
        set(s => ({
          jobs: s.jobs.map(j => j === newJob ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
          isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
        }))
        console.error('Edit generation failed:', msg)
      }
      return  // Don't fall through to normal generation
    }

    const params: Record<string, unknown> = { ...state.params, generation_mode: state.generationMode, workspace: state.activeWorkspace }
    let effectiveH3SequenceClipFrames: number | null = null
    let h3ManualSequencePrompts: string[] | null = null
    let h3ManualFirstLastPrompts: string[] | null = null

    if (
      state.generationMode === 'video'
      && state.modelOptions?.infer_audio_prompt_from_guide === true
      && params.audio_guide
      && (!params.video_guide || !String(params.video_prompt_type || '').includes('V'))
    ) {
      const audioPromptType = String(params.audio_prompt_type || '')
      if (![...'AK2'].some(letter => audioPromptType.includes(letter))) {
        // The visible soundtrack tile and its hidden mode must travel as one
        // contract. This also heals Load Settings from an affected sidecar.
        params.audio_prompt_type = `A${audioPromptType}`
      }
    }

    // H3 video-to-audio freezes the Control Video's pictures, so any
    // remembered V2V mask/edit controls are irrelevant. Normalize the request
    // copy here as a durable safety net for loaded sidecars and older saved UI
    // state; the user's friendly mode selection remains available in Studio.
    if (
      state.modelOptions?.video_to_video_inpaint === true
      && String(params.audio_prompt_type || '').includes('2')
    ) {
      params.video_prompt_type = 'GV'
      delete params.video_mask
      params.denoising_strength = 1.0
      params.masking_strength = 1.0
    }

    if (state.generationMode === 'video') {
      const fps = state.modelOptions?.fps ?? 16
      const supportsSlidingWindows = state.modelOptions?.sliding_window === true
      const minimumFrames = state.modelOptions?.frames_minimum ?? 1
      const maximumFrames = state.modelOptions?.frames_maximum ?? null
      const h3ReferenceSequenceRequested = (
        isOmniReference
        && params.minimax_h3_reference_sequence === true
      )
      const h3FirstLastMultiWindowRequested = (
        isH3Model
        && !isOmniReference
        && params.minimax_h3_multi_window === true
      )
      const ltxMultiWindowRequested = (
        isLtxSequenceModel
        && params.ltx_multi_window === true
      )
      const h3DirectOmniPass = (
        isOmniReference
        && !h3ReferenceSequenceRequested
      )
      effectiveH3SequenceClipFrames = maximumFrames
      if (h3ReferenceSequenceRequested && maximumFrames != null) {
        const sequenceBudget = effectiveH3OmniSequenceFrames({
          policy: state.modelOptions?.omni_sequence_memory_policy,
          resolution: String(params.resolution || ''),
          totalVramGb: state.systemStats?.gpu.vram_total_gb ?? 0,
          minimumFrames,
          maximumFrames,
          frameStep: state.modelOptions?.frames_steps ?? 17,
          selectedFrames: Math.round(state.slidingWindowSeconds * fps),
          manualOverride: state.slidingWindowLocked,
        })
        effectiveH3SequenceClipFrames = sequenceBudget.frames
        params.minimax_h3_sequence_clip_frames = effectiveH3SequenceClipFrames
        params.minimax_h3_sequence_memory_override = state.slidingWindowLocked
      } else {
        delete params.minimax_h3_sequence_clip_frames
        delete params.minimax_h3_sequence_memory_override
      }
      let requestedFrames = Math.max(
        minimumFrames,
        Math.round(state.durationSeconds * fps),
      )
      if (h3DirectOmniPass && maximumFrames != null) {
        // Ordinary Omni generation is one native pass. Duration is the
        // user's requested pass length; Window Length is only the VRAM-aware
        // default. Never silently shorten a visible Duration merely because
        // the saved/automatic window state is smaller.
        requestedFrames = Math.min(maximumFrames, requestedFrames)
      } else if (
        isH3Model
        && !isOmniReference
        && !h3FirstLastMultiWindowRequested
      ) {
        requestedFrames = Math.min(
          requestedFrames,
          Math.max(minimumFrames, Math.round(state.slidingWindowSeconds * fps)),
        )
      } else if (isLtxSequenceModel && !ltxMultiWindowRequested) {
        requestedFrames = Math.min(
          requestedFrames,
          Math.max(minimumFrames, Math.round(state.slidingWindowSeconds * fps)),
        )
      } else if (!supportsSlidingWindows && maximumFrames != null) {
        requestedFrames = Math.min(maximumFrames, requestedFrames)
      } else if (
        supportsSlidingWindows
        && maximumFrames != null
        && requestedFrames <= maximumFrames + 1
      ) {
        requestedFrames = Math.min(maximumFrames, requestedFrames)
      }
      if (
        isH3Model
        && maximumFrames != null
        && requestedFrames <= maximumFrames + 1
      ) {
        // Uploaded audio/video and old sidecars describe ordinary seconds.
        // Convert values such as 5.0s = 120 frames to H3's first legal clip
        // (124), and do this after all single-pass clamps so an old 5.0s
        // window preference cannot reintroduce the invalid value.
        requestedFrames = normalizeH3ClipFrames(
          requestedFrames,
          minimumFrames,
          maximumFrames,
          state.modelOptions?.frames_steps ?? 17,
        )
      }
      params.video_length = requestedFrames

      if (supportsSlidingWindows) {
        const swDefaults = state.modelOptions?.sliding_window_defaults
        let windowFrames = h3DirectOmniPass
          ? requestedFrames
          : Math.round(state.slidingWindowSeconds * fps)
        if (swDefaults) {
          const windowMinimum = swDefaults.window_min ?? 1
          const windowMaximum = swDefaults.window_max ?? windowFrames
          const windowStep = Math.max(1, swDefaults.window_step ?? 1)
          windowFrames = windowMinimum
            + Math.round((windowFrames - windowMinimum) / windowStep) * windowStep
          windowFrames = Math.max(
            windowMinimum,
            Math.min(windowMaximum, windowFrames),
          )
        }
        params.sliding_window_size = windowFrames
        params.sliding_window_overlap = _normalizeSlidingWindowOverlap(
          state.slidingWindowOverlap,
          swDefaults,
        )
        params.sliding_window_discard_last_frames = swDefaults?.discard_last_frames ?? 0
        if (isH3Model) {
          const nativeRecommendation = h3DirectOmniPass
            ? recommendedH3PassProfile(
                state.modelOptions?.omni_sequence_memory_policy,
                String(params.resolution || ''),
                state.systemStats?.gpu.vram_total_gb ?? 0,
              )
            : null
          const directOmniDurationOverride = h3DirectOmniPass && (
            state.slidingWindowLocked
            || nativeRecommendation?.supported === false
            || (
              nativeRecommendation?.frames != null
              && requestedFrames > nativeRecommendation.frames
            )
          )
          // Raising the visible one-pass Omni Duration above Auto's
          // recommendation is itself an intentional override. Derive this
          // again at submit time so model switches, loaded sidecars, or a
          // cached UI state cannot lose the user's selection.
          params.sliding_window_memory_override = (
            state.slidingWindowLocked || directOmniDurationOverride
          )
        } else if (state.modelOptions?.sliding_window_memory_policy?.manual_override) {
          params.sliding_window_memory_override = state.slidingWindowLocked
        } else {
          delete params.sliding_window_memory_override
        }
      } else {
        delete params.sliding_window_size
        delete params.sliding_window_overlap
        delete params.sliding_window_discard_last_frames
        delete params.sliding_window_memory_override
      }

      if (
        h3FirstLastMultiWindowRequested
        && params.minimax_h3_window_storyboard === false
        && requestedFrames > Number(params.sliding_window_size || 0)
      ) {
        h3ManualFirstLastPrompts = String(params.prompt || '')
          .replace(/\r\n?/g, '\n')
          .split('\n')
          .map(line => line.trim())
          .filter(Boolean)
        const expectedPromptCount = h3SlidingWindowCount({
          totalFrames: requestedFrames,
          windowFrames: Number(params.sliding_window_size || requestedFrames),
          overlapFrames: Number(params.sliding_window_overlap || 0),
          discardFrames: Number(params.sliding_window_discard_last_frames || 0),
        })
        if (h3ManualFirstLastPrompts.length !== expectedPromptCount) {
          set({
            promptEnhanceError: `Manual First / Last sequence needs exactly ${expectedPromptCount} non-empty prompt ${expectedPromptCount === 1 ? 'line' : 'lines'} (window 1 through window ${expectedPromptCount}); found ${h3ManualFirstLastPrompts.length}.`,
          })
          return
        }
        params.h3_window_prompts = h3ManualFirstLastPrompts
      }

      if (
        ltxMultiWindowRequested
        && params.ltx_window_prompt_mode === 'manual'
        && requestedFrames > Number(params.sliding_window_size || 0)
      ) {
        const ltxManualPrompts = String(params.prompt || '')
          .replace(/\r\n?/g, '\n')
          .split('\n')
          .map(line => line.trim())
          .filter(Boolean)
        const expectedPromptCount = h3SlidingWindowCount({
          totalFrames: requestedFrames,
          windowFrames: Number(params.sliding_window_size || requestedFrames),
          overlapFrames: Number(params.sliding_window_overlap || 0),
          discardFrames: Number(params.sliding_window_discard_last_frames || 0),
        })
        if (ltxManualPrompts.length !== expectedPromptCount) {
          set({
            promptEnhanceError: `Manual LTX sequence needs exactly ${expectedPromptCount} non-empty prompt ${expectedPromptCount === 1 ? 'line' : 'lines'} (window 1 through window ${expectedPromptCount}); found ${ltxManualPrompts.length}.`,
          })
          return
        }
        params.ltx_window_prompts = ltxManualPrompts
      }

      if (
        h3ReferenceSequenceRequested
        && params.minimax_h3_sequence_prompt_mode === 'manual'
        && effectiveH3SequenceClipFrames != null
      ) {
        h3ManualSequencePrompts = String(params.prompt || '')
          .replace(/\r\n?/g, '\n')
          .split('\n')
          .map(line => line.trim())
          .filter(Boolean)
        const nativeContinuation = params.minimax_h3_sequence_continuity !== false
        const expectedPromptCount = h3OmniSequenceWindowCount({
          totalFrames: requestedFrames,
          windowFrames: effectiveH3SequenceClipFrames,
          overlapFrames: Number(params.sliding_window_overlap || 0),
          nativeContinuation,
        })
        if (h3ManualSequencePrompts.length !== expectedPromptCount) {
          const unit = nativeContinuation ? 'window' : 'clip'
          set({
            promptEnhanceError: `Manual Omni sequence needs exactly ${expectedPromptCount} non-empty prompt ${expectedPromptCount === 1 ? 'line' : 'lines'} (${unit} 1 through ${unit} ${expectedPromptCount}); found ${h3ManualSequencePrompts.length}.`,
          })
          return
        }
        params.h3_window_prompts = h3ManualSequencePrompts
      }
    }

    if (isOmniReference) {
      // Ref2VA has its own ordered media manifest. Do not let a saved Frames,
      // Multi-Shot, Extend, or Blend state silently enter those pipelines.
      params.image_mode = 0
      params.image_prompt_type = ''
      delete params.image_start
      delete params.image_end
      delete params.video_source
    } else {
      // Keep Omni references in the model's in-memory working set, but do not
      // leak them into unrelated model requests or their saved sidecars.
      delete params.minimax_h3_references
      delete params.minimax_h3_reference_detail
      delete params.minimax_h3_reference_sequence
      delete params.minimax_h3_sequence_continuity
      delete params.minimax_h3_sequence_clip_frames
      delete params.minimax_h3_sequence_memory_override
    }
    if (!isH3Model) {
      delete params.minimax_h3_multi_window
      delete params.minimax_h3_sequence_prompt_mode
    }
    if (!isLtxSequenceModel) {
      delete params.ltx_multi_window
      delete params.ltx_window_prompt_mode
      delete params.ltx_window_prompts
      delete params._ltx_original_prompt
    }
    if (!state.modelOptions?.minimax_h3_text_encoder_choices?.length) {
      delete params.minimax_h3_text_encoder
    }
    if (state.modelOptions?.ltx25_video_vae_choices?.length) {
      const validLtx25VideoVae = state.modelOptions.ltx25_video_vae_choices.some(
        choice => choice.value === params.ltx25_video_vae
      )
      if (!validLtx25VideoVae) {
        params.ltx25_video_vae = (
          state.modelOptions.ltx25_video_vae_default
          || state.modelOptions.ltx25_video_vae_choices[0].value
        )
      }
    } else {
      delete params.ltx25_video_vae
    }
    if (
      state.modelOptions?.sol_attention
      && state.modelOptions.sol_attention_status?.supported
    ) {
      params.override_attention = (
        params.override_attention === 'sol' ? 'sol' : ''
      )
    } else {
      delete params.override_attention
    }
    if (state.modelOptions?.first_block_cache) {
      const allowedThresholds = (
        state.modelOptions.skip_steps_multiplier_choices || []
      ).map(choice => choice[1])
      const requestedThreshold = Number(
        params.skip_steps_multiplier
        ?? state.modelOptions.default_skip_steps_multiplier
        ?? 0.08
      )
      params.skip_steps_multiplier = allowedThresholds.includes(requestedThreshold)
        ? requestedThreshold
        : (allowedThresholds[0] ?? 0.08)
      params.skip_steps_start_step_perc = Math.max(
        0,
        Math.min(
          100,
          Number(
            params.skip_steps_start_step_perc
            ?? state.modelOptions.default_skip_steps_start_step_perc
            ?? 25
          ),
        ),
      )
      if (params.skip_steps_cache_type !== 'first_block') {
        params.skip_steps_cache_type = ''
      }
    } else {
      delete params.skip_steps_cache_type
      delete params.skip_steps_multiplier
      delete params.skip_steps_start_step_perc
    }

    // STG (Spatio-Temporal Guidance) wiring. The backend only runs STG when
    // perturbation_switch === 2 (skip-self-attention) — stg_scale alone is
    // inert. Derive the switch from the slider so an untouched slider keeps
    // the exact request shape from before this feature existed, and strip
    // all perturbation params for models without the capability so a stale
    // value can't leak across a model switch.
    if (state.modelOptions?.perturbation) {
      const stg = params.stg_scale as number | undefined
      if (stg !== undefined) {
        params.perturbation_switch = stg > 0 ? 2 : 0
      }
    } else {
      delete params.stg_scale
      delete params.perturbation_switch
      delete params.perturbation_layers
      delete params.perturbation_start_perc
      delete params.perturbation_end_perc
    }
    // Reference pipeline is a per-model capability — strip a stale toggle
    // value if the user switched to a model that doesn't support it.
    if (!(state.modelOptions as Record<string, unknown> | null)?.reference_pipeline) {
      delete params.reference_pipeline
    }

    // Tag avatar/edit-mode generations with their sub-mode so the gallery's
    // Edits filter and the loadSettingsFromOutput restore path can identify
    // them. Dedicated edit endpoints tag their jobs on the server; this is
    // retained for compatible generic edit submissions.
    if (state.generationMode === 'avatar' && state.editSubMode) {
      params.edit_sub_mode = state.editSubMode
    }

    // Default I2V / video-source strength. Distilled LTX-2 pipelines produce
    // noticeably better motion when the input anchor is at 0.7 instead of
    // tight-locked 1.0 — matches ComfyUI's reference distilled workflows
    // (stage 1 / single-stage both use 0.7). Dev and other families keep 1.0.
    // User can override via the slider; this only fires when the param isn't
    // already set.
    const _defaultIVS = (() => {
      const mt = (params.model_type as string) || ''
      return mt.includes('distilled') ? 0.7 : 1.0
    })()

    // force_fps="control" models (SCAIL-2 class) generate at the control
    // video's frame rate, but durationSeconds→video_length math uses the
    // model's nominal fps (16). Against a 25fps guide that under-counts
    // frames by a third: a "10s" request would cover only 6.4s of the
    // source performance. When the guide's real fps is known (probed at
    // upload), recompute the frame count at the rate the output will
    // actually play at.
    if (
      state.generationMode === 'video' &&
      params.video_guide &&
      params.force_fps === 'control' &&
      state.guideVideoFps && state.guideVideoFps > 0
    ) {
      // Cap at 30fps to match the server's follow-rate cap — a 60fps
      // guide would double the frame count (and sliding windows) for
      // no visible gain.
      const fpsUsed = Math.min(state.guideVideoFps, 30)
      params.video_length = Math.max(5, Math.round(state.durationSeconds * fpsUsed))
    }
    // Always tell the server what duration the user actually asked for.
    // For control-fps models the server recomputes video_length from
    // this at the guide's REAL frame rate — the durable fix for stale
    // restores (Load Settings from old sidecars carries frame counts
    // computed under the wrong fps) and for sessions where the guide's
    // fps never got probed. Underscore keys ride through harmlessly.
    if (state.generationMode === 'video' && params.video_guide) {
      ;(params as Record<string, unknown>)._duration_seconds = state.durationSeconds
    }

    // Smart multi-line prompt handling for video Frames mode:
    // When there's no sliding window (single window), send all lines as ONE prompt
    // with newlines preserved (LTX uses newlines as temporal markers within the clip).
    // When there IS sliding window, each line becomes a window prompt (mode 1).
    if (state.generationMode === 'video' && (isOmniReference || state.params.image_mode !== 2)) {
      const prompt = (params.prompt as string) || ''
      const h3WindowPromptRoutingEnabled = !isH3Model || (
        isOmniReference
          ? params.minimax_h3_reference_sequence === true
          : params.minimax_h3_multi_window === true
      )
      const ltxWindowPromptRoutingEnabled = (
        !isLtxSequenceModel
        || params.ltx_multi_window === true
      )
      const hasSlidingWindow = state.modelOptions?.sliding_window === true
        && h3WindowPromptRoutingEnabled
        && ltxWindowPromptRoutingEnabled
        && state.durationSeconds > state.slidingWindowSeconds
      if (
        hasSlidingWindow
        && (
          state.modelOptions?.sliding_window_auto_prompt_pacing === true
          || (
            isLtxSequenceModel
            && params.ltx_window_prompt_mode !== 'manual'
          )
        )
      ) {
        // Auto planners receive one complete story idea. The backend then
        // compiles exact H3 Context-IR or LTX prose for each native pass.
        params.multi_prompts_gen_type = 2
      } else if (hasSlidingWindow && prompt.includes('\n')) {
        // Sliding window: each line = one window prompt (rolling generation)
        params.multi_prompts_gen_type = 1
      } else if (!hasSlidingWindow && prompt.includes('\n')) {
        // No sliding window — send entire prompt as one (multi_prompts_gen_type=2 preserves newlines)
        params.multi_prompts_gen_type = 2
      }
    }

    // Post-processing settings
    if (state.spatialUpsampling) params.spatial_upsampling = state.spatialUpsampling
    if (state.filmGrainIntensity > 0) {
      params.film_grain_intensity = state.filmGrainIntensity
      params.film_grain_saturation = state.filmGrainSaturation
    }
    // Voice clone (SeedVC) — only send if the user explicitly enabled
    // it AND provided at least one reference. Backend defaults all three
    // params to falsy if absent (postprocessing step is a no-op).
    if (state.voiceCloneEnabled && state.voiceCloneRefs.length > 0) {
      const validRefs = state.voiceCloneRefs.filter(r => r && r.path)
      if (validRefs.length > 0) {
        params.voice_clone_enabled = true
        params.voice_clone_mode = state.voiceCloneMode
        // Pass server-side paths (already uploaded via /api/v1/upload-audio).
        params.voice_clone_refs = validRefs.map(r => r.path)
      }
    }

    // Image mode: force single frame + image output format
    // Backend uses image_mode > 0 to determine output as image (.jpg) vs video (.mp4)
    if (state.generationMode === 'image') {
      params.video_length = 1
      params.image_mode = 1
      // WanGP expects control input in image_guide (not video_guide) for image mode
      if (params.video_guide && !params.image_guide) {
        params.image_guide = params.video_guide
      }
    }

    // Audio mode: branch by sub-mode (Speech/Music vs SFX)
    if (state.generationMode === 'audio') {
      // Record the active sub-tab in the request so it lands in the
      // .meta.json sidecar — Load Settings uses it to restore Speech /
      // Music / SFX, not just the Audio tab. Underscore keys ride
      // through generation untouched, same as _tts_*. Music also saves
      // its song-writer inputs (UI-only, not consumed by generation).
      params._audio_sub_mode = state.audioSubMode
      if (state.audioSubMode === 'music') {
        params._music_description = state.musicDescription || ''
        params._music_instrumental = !!state.musicInstrumental
      }
      if (state.audioSubMode === 'sfx') {
        // SFX mode: use MMAudio to generate sound effects
        // MMAudio runs as post-processing on a video model, so use a video model as carrier
        const sfxModel = params.model_type as string
        const isSfxVirtual = sfxModel.startsWith('mmaudio_')
        if (isSfxVirtual) {
          // Swap virtual MMAudio model for a real video model; backend uses MMAudio params
          params.model_type = 'ltx2_22B_distilled_1_1'
          // Keep the virtual id so Load Settings can restore the SFX tab's
          // model selection (the sidecar otherwise records only the carrier).
          params._sfx_virtual_model = sfxModel
        }
        params.MMAudio_setting = 1
        // Always set MMAudio variant explicitly so backend doesn't fall back to server config
        params._mmaudio_variant = sfxModel === 'mmaudio_nsfw' ? 'nsfw' : 'v2'
        // Copy MMAudio prompt into main prompt field (for API validation & metadata)
        if (!params.prompt && params.MMAudio_prompt) {
          params.prompt = params.MMAudio_prompt
        }
        params.sfx_mode = true
        params.duration_seconds = state.durationSeconds
        // Generate a minimal video if no video_guide uploaded (1 frame), then run MMAudio
        if (!params.video_guide) {
          params.video_length = 17  // Minimum viable video for MMAudio (~1s)
          params.num_inference_steps = 4
        } else {
          params.video_length = 0  // No video gen needed — just run MMAudio on uploaded video
        }
        params.image_mode = 0
        // Clear video-specific params
        delete params.sliding_window_size
        delete params.sliding_window_overlap
        delete params.sliding_window_discard_last_frames
      } else {
        // Speech/Music TTS mode
        params.video_length = 0
        params.image_mode = 0
        params.multi_prompts_gen_type = 2  // Preserve full text as one prompt (don't split by newlines)
        // Save original prompt + speaker names before swap (for load settings)
        params._tts_original_prompt = params.prompt
        params._tts_speaker_name1 = state.ttsSpeakerName1 || ''
        params._tts_speaker_name2 = state.ttsSpeakerName2 || ''
        // Save all voice names for metadata
        for (let i = 0; i < state.ttsVoices.length; i++) {
          (params as Record<string, unknown>)[`_tts_speaker_name${i + 1}`] = state.ttsVoices[i]?.name || ''
        }
        params._tts_voice_count = state.ttsVoiceCount
        // Swap character names → Speaker N: for TTS multi-voice mode
        const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        let text = params.prompt as string
        for (let i = 0; i < state.ttsVoices.length; i++) {
          const name = state.ttsVoices[i]?.name
          if (name) {
            text = text.replace(new RegExp(escapeRegex(name) + '\\s*:', 'gi'), `Speaker ${i + 1}:`)
          }
        }
        params.prompt = text
        // Set audio_guide paths for each voice (audio_guide, audio_guide2, audio_guide3, etc.)
        for (let i = 0; i < state.ttsVoices.length; i++) {
          const voice = state.ttsVoices[i]
          if (voice?.path) {
            const key = i === 0 ? 'audio_guide' : `audio_guide${i + 1}`
            params[key as keyof typeof params] = voice.path as never
          }
        }
        // TTS duration (max duration for the model to generate)
        if (state.modelOptions?.audio_only) {
          // Prefer the slider's `default` (some TTS models — e.g. DramaBox —
          // set default=0 to mean "auto-derive duration from prompt"); fall
          // back to `max` then 600.
          const ds = state.modelOptions.duration_slider
          const sliderDefault = ds?.default ?? ds?.max ?? 600
          params.duration_seconds = state.durationSeconds < 30 ? sliderDefault : state.durationSeconds
        }
        // Let the TTS model use its own defaults for steps/guidance if ours are video defaults
        if ((params.num_inference_steps as number) > 0 && state.modelOptions?.default_num_inference_steps == null) {
          params.num_inference_steps = 0
        }
        // Clear video-specific params
        delete params.sliding_window_size
        delete params.sliding_window_overlap
        delete params.sliding_window_discard_last_frames
      }
    }

    // Defensive cleanup: strip stale "V" (Source Video / extend) flag from
    // image_prompt_type when we're NOT entering the extend/continue path.
    //
    // The leak: when the user does a video extend (image_mode=3 +
    // continueVideoPath) and submits, the continue-mode branch below sets
    // params.image_prompt_type = "V". That mutation is on the local params
    // copy and shouldn't persist, BUT load-settings (loadSettingsFromOutput
    // at line 5284) DOES restore image_prompt_type from sidecar metadata
    // into state.params.image_prompt_type. So after extending a video and
    // then switching back to Frames mode via ModeToggle (which only flips
    // image_mode 3 -> 0, leaving image_prompt_type untouched), the next
    // generation carries forward image_prompt_type="V" from state.
    //
    // The single-clip and end-image handlers below only APPEND flags
    // (e.g. "S" + "V" -> "SV"), they never strip stale ones. So the "V"
    // survives, the backend (wgp.py:941-943) sees it and demands
    // video_source — but the user is in Frames mode with no source video.
    //
    // Symptom user reported: "I did a video extend and it worked. then I
    // switched to normal video mode and it keeps telling me to load a
    // source video. even after I refresh the page and try a new generation."
    //
    // Fix: strip "V" up-front unless we're going to re-add it in the
    // continue/extend branch below. The continue branch (image_mode === 3
    // + continueVideoPath) re-sets image_prompt_type = "V" wholesale, so
    // stripping here is safe — that branch puts it back.
    const willEnterContinueBranch = state.generationMode === 'video'
      && (params.image_mode === 3 || state.params.image_mode === 3)
      && !!state.continueVideoPath
    if (!willEnterContinueBranch) {
      const ipt = (params.image_prompt_type as string) || ''
      if (ipt.includes('V')) {
        params.image_prompt_type = ipt.replace(/V/g, '')
      }
      // Same stale-flag defense for the "T" temporal-alignment flag the
      // continue branch adds to video_prompt_type (see below). Without a
      // source video it's a backend no-op (alignment shift = 0 frames),
      // but stripping keeps restored-from-sidecar state from carrying it
      // into unrelated generations. Only the TRAILING "T" is that flag — an
      // internal "T" is the depth_temporal control letter (PTVG/TVG/TEVG),
      // and a global strip turned "Motion + Temporal Depth" (PTVG) into plain
      // "Transfer Human Motion" (PVG) at submit time, so use /T$/.
      const vptClean = (params.video_prompt_type as string) || ''
      if (vptClean.endsWith('T')) {
        params.video_prompt_type = vptClean.replace(/T$/, '')
      }
    }

    // Multi-clip path
    if (!isOmniReference && state.params.image_mode === 2) {
      const clips = state.clips
      const imagePaths: string[] = []
      const endImagePaths: string[] = []
      let hasAnyEndImage = false

      for (const clip of clips) {
        if (clip.startImage) {
          try {
            const result = await api.uploadImage(clip.startImage)
            imagePaths.push(result.path)
          } catch (e) {
            console.error('Failed to upload clip image:', e)
            imagePaths.push('')
          }
        } else if (clip.startImagePath) {
          imagePaths.push(clip.startImagePath)
        } else {
          imagePaths.push('')
        }

        // Upload end images (seamless mode)
        if (clip.endImage) {
          try {
            const result = await api.uploadImage(clip.endImage)
            endImagePaths.push(result.path)
            hasAnyEndImage = true
          } catch (e) {
            console.error('Failed to upload clip end image:', e)
            endImagePaths.push('')
          }
        } else if (clip.endImagePath) {
          endImagePaths.push(clip.endImagePath)
          hasAnyEndImage = true
        } else {
          endImagePaths.push('')
        }
      }

      let promptLines: string[]
      if (state.singlePromptMode) {
        const p: string = clips[0]?.prompt || (params.prompt as string) || ''
        promptLines = clips.map(() => p)
      } else {
        promptLines = clips.map(c => c.prompt || '')
      }

      params.prompt = promptLines.join('\n')
      params.image_start = imagePaths
      if (hasAnyEndImage) {
        params.image_end = endImagePaths
      }
      params.multi_prompts_gen_type = 3
      params.image_mode = 0
      params.image_prompt_type = hasAnyEndImage ? 'SE' : 'S'
      if (params.input_video_strength == null) params.input_video_strength = _defaultIVS
    }
    // Single I2V path: Upload images if present (new File upload takes priority)
    // Skip in image mode — startImage is for video I2V, not image generation
    else if (!isOmniReference && state.startImage && state.generationMode !== 'image') {
      try {
        const result = await api.uploadImage(state.startImage)
        params.image_start = result.path
        params.image_mode = 0
        const ipt = (params.image_prompt_type as string) || ''
        if (!ipt.includes('S')) params.image_prompt_type = 'S' + ipt
        if (params.input_video_strength == null) params.input_video_strength = _defaultIVS
      } catch (e) {
        console.error('Failed to upload start image:', e)
      }
    } else if (!isOmniReference && params.image_start && state.generationMode !== 'image') {
      // Re-roll case: image_start is already an absolute path from sidecar metadata
      params.image_mode = 0
      const ipt = (params.image_prompt_type as string) || ''
      if (!ipt.includes('S')) params.image_prompt_type = 'S' + ipt
      if (params.input_video_strength == null) params.input_video_strength = _defaultIVS
    }
    if (!isOmniReference && state.endImage) {
      try {
        const result = await api.uploadImage(state.endImage)
        params.image_end = result.path
        const ipt = (params.image_prompt_type as string) || ''
        if (!ipt.includes('E')) params.image_prompt_type = ipt + 'E'
      } catch (e) {
        console.error('Failed to upload end image:', e)
      }
    } else if (params.image_end) {
      const ipt = (params.image_prompt_type as string) || ''
      if (!ipt.includes('E')) params.image_prompt_type = ipt + 'E'
    }

    // Continue mode: set video_source and image_prompt_type="V"
    if (!isOmniReference && state.generationMode === 'video' && params.image_mode === 3 && state.continueVideoPath) {
      params.video_source = state.continueVideoPath
      params.image_prompt_type = 'V'
      params.image_mode = 0
      if (params.input_video_strength == null) params.input_video_strength = _defaultIVS
      // Temporal alignment: the UI scopes EVERYTHING to the new content —
      // durationSeconds is the extend length, and ControlVideoSection's
      // injected-frame positions are computed against that timeline. The
      // backend, however, defaults to interpreting frames_positions (and
      // control video / control audio alignment) against the FULL timeline
      // including the source clip (wgp.py: reset_control_aligment = "T" in
      // video_prompt_type; alignment_shift = source frames only when "T").
      // Without "T", a frame injected at "end of the new 20s" of a 10s clip
      // lands at the 20s mark of the 30s output — 10s early; on longer
      // sources the position can fall entirely INSIDE the source span and
      // visibly never happen. "T" = upstream's "Aligned to the beginning of
      // the First Window of the new Video Sample", which matches the UI.
      // Append the alignment flag as a TRAILING "T". Guard on endsWith, not
      // includes: a control value with an internal "T" is depth_temporal
      // (PTVG/TVG/TEVG), and an includes() guard would skip the append for
      // those — silently dropping temporal alignment on an extend that uses a
      // Temporal-Depth control video. endsWith adds the flag while leaving the
      // process letter intact; the display/persist/submit strips remove only
      // this trailing "T" again.
      const vptExtend = (params.video_prompt_type as string) || ''
      if (!vptExtend.endsWith('T')) {
        params.video_prompt_type = vptExtend + 'T'
      }
      // Duration is the amount of NEW content requested by the user. The
      // backend adds the source-tail overlap only to the model's first pass;
      // it is conditioning context and must not be subtracted here. If that
      // context pushes the request beyond one safe H3 pass, native sliding
      // windows are the correct behavior and preserve the requested length.
    }

    // Safety net: Studio Video mode ALWAYS produces video. The sub-mode
    // branches above translate image_mode 2/3 (Multi-Shot/Extend) to 0 + other
    // flags, but a plain T2V gen (no start image) hits none of them — so a
    // stale non-zero image_mode (e.g. an I2V clip's settings loaded via the
    // pencil, or Extend mode left without a source video) would leak through
    // and the backend (is_image = image_mode > 0) would emit a single PNG
    // instead of a video. Force video output here, after the sub-mode branches
    // have already read image_mode.
    if (state.generationMode === 'video') {
      params.image_mode = 0
    }

    // Image references (from ImageRefSection)
    if (state.imageRefType && state.imageRefs.length > 0) {
      const refPaths: string[] = []
      for (const file of state.imageRefs) {
        try {
          const result = await api.uploadImage(file)
          refPaths.push(result.path)
        } catch (e) {
          console.error('Failed to upload reference image:', e)
        }
      }
      if (refPaths.length > 0) {
        params.image_refs = refPaths
        params.remove_background_images_ref = state.removeBackgroundRefs ? 1 : 0
        // Merge image ref letter codes into video_prompt_type
        let vpt = (params.video_prompt_type as string) || ''
        for (const letter of state.imageRefType) {
          if (!vpt.includes(letter)) vpt += letter
        }
        params.video_prompt_type = vpt
      }
    } else if (params.image_refs && (params.image_refs as string[]).length > 0) {
      // Re-roll case: image_refs already populated from sidecar metadata
      params.remove_background_images_ref = params.remove_background_images_ref ?? 0
    } else {
      // No reference images attached for this submission. Strip any
      // image-ref letter codes from video_prompt_type that may have
      // persisted from an earlier task — without this, a user who
      // generates with refs once and then clears them gets stuck with
      // "I" (or other ref-letter codes) baked into the saved per-mode
      // params snapshot, which the backend rejects with "You must
      // provide at least one Reference Image". The backend has a
      // safety net that catches this too, but cleaning at the source
      // keeps the snapshot itself sensible.
      const vpt = (params.video_prompt_type as string) || ''
      if (vpt) {
        // Default ref letters used by Maestro when image refs are
        // present. If imageRefType is configured we trust that;
        // otherwise fall back to the conservative "I" — the most common
        // and the one we've actually observed leaking.
        const refLetters = state.imageRefType || 'I'
        let cleaned = vpt
        for (const letter of refLetters) {
          cleaned = cleaned.split(letter).join('')
        }
        if (cleaned !== vpt) {
          params.video_prompt_type = cleaned
        }
      }
      // Make sure no stale image_refs path list rides along either.
      if (params.image_refs !== undefined && (!params.image_refs || (params.image_refs as string[]).length === 0)) {
        delete params.image_refs
      }
    }

    // Voice reference (ID-LoRA) — upload if present, add to params
    if (state.directorVoiceRef) {
      let vrPath = state.directorVoiceRefPath
      if (!vrPath) {
        try {
          const uploaded = await api.uploadAudio(state.directorVoiceRef)
          vrPath = uploaded.path
          set({ directorVoiceRefPath: vrPath })
        } catch { /* skip */ }
      }
      if (vrPath) {
        params.voice_reference = vrPath
        params.identity_guidance_scale = state.directorIdentityGuidanceScale
      }
    }

    const h3WindowStoryboardActive = (
      state.generationMode === 'video'
      && state.modelOptions?.sliding_window_auto_prompt_pacing === true
      && params.minimax_h3_multi_window === true
      && params.minimax_h3_window_storyboard !== false
      && state.params.image_mode !== 2
      && Number(params.video_length || 0) > Number(params.sliding_window_size || 0)
    )
    const h3ReferenceSequenceActive = (
      state.generationMode === 'video'
      && isOmniReference
      && params.minimax_h3_reference_sequence === true
      && Number(params.video_length || 0) > Number(
        effectiveH3SequenceClipFrames
        || state.modelOptions?.frames_maximum
        || 0,
      )
    )
    const h3ManualReferenceSequence = (
      state.generationMode === 'video'
      && isOmniReference
      && params.minimax_h3_reference_sequence === true
      && params.minimax_h3_sequence_prompt_mode === 'manual'
    )
    const h3ManualFirstLastSequence = (
      state.generationMode === 'video'
      && isH3Model
      && !isOmniReference
      && params.minimax_h3_multi_window === true
      && params.minimax_h3_window_storyboard === false
      && Number(params.video_length || 0) > Number(params.sliding_window_size || 0)
    )
    const ltxWindowSequenceActive = (
      state.generationMode === 'video'
      && isLtxSequenceModel
      && params.ltx_multi_window === true
      && Number(params.video_length || 0) > Number(params.sliding_window_size || 0)
    )
    const ltxAutoPlanActive = (
      ltxWindowSequenceActive
      && params.ltx_window_prompt_mode !== 'manual'
    )
    const h3PlanActive = h3WindowStoryboardActive || (
      h3ReferenceSequenceActive && !h3ManualReferenceSequence
    )
    if (h3ManualFirstLastSequence) {
      params.minimax_h3_window_storyboard = false
      params.h3_window_prompts = h3ManualFirstLastPrompts ?? []
      delete params.h3_window_plan_signature
      delete params.h3_window_plan
    } else if (h3ManualReferenceSequence) {
      delete params.minimax_h3_window_storyboard
      params.h3_window_prompts = h3ManualSequencePrompts ?? []
      delete params.h3_window_plan_signature
      delete params.h3_window_plan
    } else if (state.modelOptions?.sliding_window_auto_prompt_pacing === true) {
      params.minimax_h3_window_storyboard = h3WindowStoryboardActive
      if (h3WindowStoryboardActive && state.h3WindowPlan) {
        params.h3_window_prompts = state.h3WindowPlan.windows.map(window => window.prompt)
        params.h3_window_plan_signature = state.h3WindowPlan.signature
        params.h3_window_plan = state.h3WindowPlan
      } else {
        delete params.h3_window_prompts
        delete params.h3_window_plan_signature
        delete params.h3_window_plan
      }
    } else if (h3ReferenceSequenceActive) {
      delete params.minimax_h3_window_storyboard
      if (state.h3WindowPlan?.plan_kind === 'reference_sequence') {
        params.h3_window_prompts = state.h3WindowPlan.windows.map(window => window.prompt)
        params.h3_window_plan_signature = state.h3WindowPlan.signature
        params.h3_window_plan = state.h3WindowPlan
      } else {
        delete params.h3_window_prompts
        delete params.h3_window_plan_signature
        delete params.h3_window_plan
      }
    } else {
      delete params.minimax_h3_window_storyboard
      delete params.h3_window_prompts
      delete params.h3_window_plan_signature
      delete params.h3_window_plan
    }

    const newJob: GenerationJob = {
      id: '',
      status: 'queued',
      progress: 0,
      step: 0,
      totalSteps: 0,
      phase: '',
      message: h3PlanActive
        ? `Planning H3 ${h3ReferenceSequenceActive ? 'reference sequence' : 'windows'}...`
        : ltxAutoPlanActive
          ? 'Planning LTX windows...'
        : h3ManualReferenceSequence
          ? 'Preparing H3 manual sequence...'
          : 'Submitting...',
      outputFiles: [],
      error: null,
      oomInfo: null,
    }

    set(s => ({
      isGenerating: true,
      jobs: [newJob, ...s.jobs],
    }))

    try {
      const { job_id, h3_window_plan, ltx_window_plan } = await api.submitGeneration(params)

      if (h3_window_plan) {
        const planFps = state.modelOptions?.fps ?? 24
        const effectiveWindowFrames = h3_window_plan.effective_window_frames
          || h3_window_plan.window_frames
        if (h3_window_plan.plan_kind === 'reference_sequence') {
          set(s => ({
            h3WindowPlan: h3_window_plan,
            slidingWindowSeconds: effectiveWindowFrames / planFps,
            params: {
              ...s.params,
              minimax_h3_sequence_clip_frames: effectiveWindowFrames,
            },
          }))
        } else {
          set(s => ({
            h3WindowPlan: h3_window_plan,
            slidingWindowSeconds: effectiveWindowFrames / planFps,
            params: { ...s.params, sliding_window_size: effectiveWindowFrames },
          }))
        }
      }
      if (ltx_window_plan) {
        const isManualPlan = ltx_window_plan.planned_by === 'manual'
        set(s => ({
          params: {
            ...s.params,
            prompt: ltx_window_plan.window_prompts.join('\n'),
            ltx_window_prompts: ltx_window_plan.window_prompts,
            _ltx_original_prompt: isManualPlan
              ? undefined
              : ltx_window_plan.source_prompt,
          },
        }))
      }

      // Update the job with its server-assigned ID
      set(s => ({
        jobs: s.jobs.map(j => j === newJob ? {
          ...j,
          id: job_id,
          status: 'running',
          message: 'Queued...',
          h3WindowPlan: h3_window_plan ?? null,
        } : j),
      }))

      // Poll for status on this specific job
      const pollInterval = setInterval(async () => {
        // Check if this job was removed (stopped)
        if (!get().jobs.find(j => j.id === job_id)) {
          clearInterval(pollInterval)
          return
        }

        try {
          const status = await api.fetchJobStatus(job_id)

          set(s => ({
            jobs: s.jobs.map(j => j.id !== job_id ? j : {
              ...j,
              status: status.status,
              progress: status.progress / 100,
              step: status.step,
              totalSteps: status.total_steps,
              phase: status.phase,
              message: status.message,
              outputFiles: status.output_files,
              error: status.error,
              oomInfo: status.oom_info ?? null,
            }),
          }))

          // Refresh gallery during generation to show sliding window progress
          if (status.status === 'running') {
            get().refreshOutputs()
          }

          if (status.status === 'completed') {
            clearInterval(pollInterval)
            // Completed job — remove the placeholder, real output now in gallery
            set(s => {
              const remaining = s.jobs.filter(j => j.id !== job_id)
              return {
                jobs: remaining,
                isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
              }
            })
            get().loadOutputs()
          } else if (status.status === 'failed' || status.status === 'cancelled') {
            clearInterval(pollInterval)
            // Keep the failed/cancelled job in the queue so its placeholder
            // card stays visible with the error message. User dismisses via
            // the X button on the tile.
            set(s => ({
              isGenerating: s.jobs.some(j => j.id !== job_id && (j.status === 'running' || j.status === 'queued')),
            }))
          }
        } catch (e) {
          console.error('Status poll error:', e)
        }
      }, 2000)

    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Generation failed'
      // Submit itself failed (pre-queue). Convert the placeholder to a failed
      // state in place so the user sees what happened, rather than making the
      // tile disappear and leaving them to wonder.
      set(s => ({
        jobs: s.jobs.map(j => j === newJob ? { ...j, id: j.id || `submit-fail-${Date.now()}`, status: 'failed', message: msg, error: msg } : j),
        isGenerating: s.jobs.some(j => j !== newJob && (j.status === 'running' || j.status === 'queued')),
      }))
    }
  },

  stopGeneration: (jobId) => {
    if (jobId) {
      // Cancel specific job on backend, then remove from UI
      api.cancelJob(jobId).catch(e => console.error('Cancel failed:', e))
      set(s => {
        const remaining = s.jobs.filter(j => j.id !== jobId)
        return { jobs: remaining, isGenerating: remaining.length > 0 }
      })
    } else {
      // Cancel all jobs
      const jobs = get().jobs
      jobs.forEach(j => {
        if (j.id) api.cancelJob(j.id).catch(() => {})
      })
      set({ jobs: [], isGenerating: false })
    }
  },

  // UI-only removal of a job tile (e.g. dismissing a failed/cancelled
  // placeholder). No backend call — the job is already terminal.
  dismissJob: (jobId) => {
    set(s => {
      const remaining = s.jobs.filter(j => j.id !== jobId)
      return {
        jobs: remaining,
        isGenerating: remaining.some(j => j.status === 'running' || j.status === 'queued'),
      }
    })
  },

  reconnectJobs: async () => {
    // On page load, check backend for any active jobs and restore them
    try {
      const data = await api.fetchActiveJobs()
      if (data.jobs.length > 0) {
        const existingIds = new Set(get().jobs.map(j => j.id))
        const newJobs: GenerationJob[] = data.jobs
          .filter(j => !existingIds.has(j.job_id))
          .map(j => ({
            id: j.job_id,
            status: j.status as GenerationJob['status'],
            progress: j.progress / 100,
            step: j.step,
            totalSteps: j.total_steps,
            phase: j.phase,
            message: j.message,
            outputFiles: j.output_files,
            error: j.error,
            oomInfo: (j as { oom_info?: import('../types').OomInfo | null }).oom_info ?? null,
            h3WindowPlan: j.h3_window_plan ?? null,
          }))
        if (newJobs.length > 0) {
          set(s => ({
            jobs: [...s.jobs, ...newJobs],
            isGenerating: true,
          }))
          // Start polling for each reconnected job
          newJobs.forEach(job => {
            const pollInterval = setInterval(async () => {
              try {
                const status = await api.fetchJobStatus(job.id)
                set(s => ({
                  jobs: s.jobs.map(j => j.id !== job.id ? j : {
                    ...j,
                    status: status.status,
                    progress: status.progress / 100,
                    step: status.step,
                    totalSteps: status.total_steps,
                    phase: status.phase,
                    message: status.message,
                    outputFiles: status.output_files,
                    error: status.error,
                    oomInfo: status.oom_info ?? null,
                  }),
                }))
                if (status.status === 'completed' || status.status === 'failed' || status.status === 'cancelled') {
                  clearInterval(pollInterval)
                  set(s => {
                    const remaining = s.jobs.filter(j => j.id !== job.id)
                    return { jobs: remaining, isGenerating: remaining.length > 0 }
                  })
                  get().loadOutputs()
                }
              } catch {
                // Job may have been cleaned up
                clearInterval(pollInterval)
                set(s => {
                  const remaining = s.jobs.filter(j => j.id !== job.id)
                  return { jobs: remaining, isGenerating: remaining.length > 0 }
                })
              }
            }, 2000)
          })
          console.log(`[Queue] Reconnected to ${newJobs.length} active job(s)`)
        }
      }
    } catch {
      // Backend might not have the endpoint yet, silently ignore
    }
  },

  // LoRA state
  availableLoras: [],
  lorasLoading: false,
  loraWeights: {},
  loraIdByFilename: {},
  filenameByLoraId: {},

  /**
   * Refresh the lora_id ↔ filename maps from /api/v1/loras/installed.
   * Called once at boot (from loadModels) and again whenever LoRAs may
   * have been added/removed (after CivitAI download, scan, etc.).
   *
   * Side effect: runs reconciliation against the persisted savedLoraPerMode.
   * If a saved filename no longer exists on disk but the snapshot lora_id
   * resolves to a different filename in the fresh map, the rename is
   * applied transparently — that's the LoRA-version-update flow.
   */
  refreshLoraIdMap: async () => {
    try {
      const { loras } = await api.fetchInstalledLoras()
      const byFilename: Record<string, string> = {}
      const byLoraId: Record<string, string> = {}
      for (const l of loras) {
        if (!l.lora_id || !l.filename) continue
        byFilename[l.filename] = l.lora_id
        // If two files share a lora_id (rare — user kept v1 + v2 side by
        // side), the last one wins. Reconciliation will prefer whichever
        // matches the saved filename.
        byLoraId[l.lora_id] = l.filename
      }
      // Reconcile: rewrite stale filenames in savedLoraPerMode using the
      // snapshot loaded from localStorage (lora_id → filename-at-save-time).
      const s = get()
      const snapshot = s._loraFilenameSnapshotAtLoad || {}
      const reconciled: typeof s.savedLoraPerMode = {}
      let changed = false
      for (const [mode, blob] of Object.entries(s.savedLoraPerMode)) {
        if (!blob) continue
        const renameFilename = (fname: string): string | null => {
          if (byFilename[fname]) return fname  // still on disk, no change
          // Stale: look up its lora_id in snapshot, then current filename in fresh map.
          // Walk snapshot backwards (lora_id → fname) to find the lora_id this filename had.
          let foundId: string | null = null
          for (const [id, snapFname] of Object.entries(snapshot)) {
            if (snapFname === fname) { foundId = id; break }
          }
          if (foundId && byLoraId[foundId]) {
            changed = true
            return byLoraId[foundId]  // renamed
          }
          // LoRA was deleted entirely.
          changed = true
          return null
        }
        const newActivated = (blob.activated_loras || [])
          .map(renameFilename)
          .filter((x): x is string => x !== null)
        const newWeights: Record<string, number[]> = {}
        for (const [fname, w] of Object.entries(blob.loraWeights || {})) {
          const renamed = renameFilename(fname)
          if (renamed) newWeights[renamed] = w
        }
        const newAvailable = (blob.availableLoras || [])
          .map(renameFilename)
          .filter((x): x is string => x !== null)
        reconciled[mode as GenerationMode] = {
          ...blob,
          activated_loras: newActivated,
          loraWeights: newWeights,
          availableLoras: newAvailable,
        }
      }
      if (changed) {
        // Also rewrite the in-memory runtime state if its keys are stale
        const renameRuntimeFilename = (fname: string): string | null => {
          if (byFilename[fname]) return fname
          let foundId: string | null = null
          for (const [id, snapFname] of Object.entries(snapshot)) {
            if (snapFname === fname) { foundId = id; break }
          }
          if (foundId && byLoraId[foundId]) return byLoraId[foundId]
          return null
        }
        const curActivated = (s.params.activated_loras || [])
          .map(renameRuntimeFilename)
          .filter((x): x is string => x !== null)
        const curWeights: Record<string, number[]> = {}
        for (const [fname, w] of Object.entries(s.loraWeights || {})) {
          const renamed = renameRuntimeFilename(fname)
          if (renamed) curWeights[renamed] = w
        }
        set(state => ({
          loraIdByFilename: byFilename,
          filenameByLoraId: byLoraId,
          savedLoraPerMode: reconciled,
          params: { ...state.params, activated_loras: curActivated },
          loraWeights: curWeights,
        }))
        // Persist the reconciled state so next boot doesn't need to redo it.
        const ns = get()
        _saveSettings({
          generationMode: ns.generationMode,
          selectedModelPerMode: ns.selectedModelPerMode,
          savedParamsPerMode: ns.savedParamsPerMode,
          savedLoraPerMode: ns.savedLoraPerMode,
          savedPromptPerMode: ns.savedPromptPerMode,
        }, byFilename)
      } else {
        set({ loraIdByFilename: byFilename, filenameByLoraId: byLoraId })
      }
      // Fire-and-forget: kick off an update check, debounced server-side
      // by a 24h staleness window. If the manifest is fresh, the backend
      // returns immediately without hitting CivitAI; if stale, it walks
      // the library and refreshes badges in the background. The user's
      // current LoraSelector instance will pick up new badges on its
      // next /details fetch (mode change or refresh).
      api.checkLoraUpdates(false).catch(() => {
        // Network failures here are non-fatal — the manual "Check" button
        // in the LoraSelector remains available for retries.
      })
    } catch {
      // Non-fatal. Persistence will keep using filename-keyed legacy shape
      // until the map populates on a subsequent attempt.
    }
  },

  loadLoras: async (modelType) => {
    set({ lorasLoading: true })
    try {
      const data = await api.fetchLoras(modelType)
      set({ availableLoras: data.loras, lorasLoading: false })
    } catch {
      set({ availableLoras: [], lorasLoading: false })
    }
  },

  toggleLora: (filename) => {
    const { params, loraWeights, modelOptions, generationMode, editSubMode } = get()
    const current = [...params.activated_loras]
    const idx = current.indexOf(filename)
    const newWeights = { ...loraWeights }
    // SCAIL-2 Recast is intentionally a single-phase pipeline even though
    // the shared Wan model family advertises support for up to three phases.
    const recastSinglePhase = generationMode === 'avatar' && editSubMode === 'recast'
    const phases = recastSinglePhase ? 1 : Math.max(1, modelOptions?.guidance_max_phases ?? 1)
    const managedTurboFilenames = new Set(
      modelOptions?.minimax_h3_turbo?.presets?.map(preset => preset.filename)
      || (modelOptions?.minimax_h3_turbo?.filename
        ? [modelOptions.minimax_h3_turbo.filename]
        : []),
    )
    const removedTurboPreset = idx >= 0 && managedTurboFilenames.has(filename)

    if (idx >= 0) {
      current.splice(idx, 1)
      delete newWeights[filename]
    } else {
      current.push(filename)
      newWeights[filename] = Array(phases).fill(1.0)
    }

    // Serialize multipliers
    const multipliers = current.map(name => {
      const w = newWeights[name] || [1.0]
      return Array.from(
        { length: phases },
        (_, i) => w[i] ?? w[w.length - 1] ?? 1.0,
      ).map(v => v.toFixed(2)).join(';')
    }).join(' ')

    set(s => ({
      loraWeights: newWeights,
      params: {
        ...s.params,
        activated_loras: current,
        loras_multipliers: multipliers,
        ...(removedTurboPreset ? { minimax_h3_turbo_mode: false } : {}),
      },
    }))
    // Persist LoRA state
    const s = get()
    const mode = s.generationMode
    const updatedLoraPerMode = {
      ...s.savedLoraPerMode,
      [mode]: { activated_loras: current, loras_multipliers: multipliers, loraWeights: newWeights, availableLoras: s.availableLoras },
    }
    const updatedParamsPerMode = removedTurboPreset
      ? {
          ...s.savedParamsPerMode,
          [mode]: {
            ...(s.savedParamsPerMode[mode] || {}),
            minimax_h3_turbo_mode: false,
          },
        }
      : s.savedParamsPerMode
    set({
      savedLoraPerMode: updatedLoraPerMode,
      savedParamsPerMode: updatedParamsPerMode,
    })
    _saveSettings({ generationMode: mode, selectedModelPerMode: s.selectedModelPerMode, savedParamsPerMode: updatedParamsPerMode, savedLoraPerMode: updatedLoraPerMode, savedPromptPerMode: s.savedPromptPerMode }, s.loraIdByFilename)
  },

  ensureTransitionLoraForBlend: async () => {
    const state = get()
    const modelType = state.params.model_type as string
    // Only applies to LTX-2 family models — the LoRA is trained for LTX-2.3
    if (!modelType || !modelType.startsWith('ltx2')) return

    const HF_URL = 'https://huggingface.co/valiantcat/LTX-2.3-Transition-LORA'
    const matchesTransitionLora = (name: string) => /transition/i.test(name)

    try {
      // Step 1: check if already installed
      let { loras } = await api.fetchLoras(modelType)
      let transitionFilename = loras.find(matchesTransitionLora)

      // Step 2: if not installed, trigger HF download
      if (!transitionFilename) {
        console.log('[Blend] Transition LoRA not found locally — downloading from HuggingFace')
        let result: { filename: string } | null = null
        try {
          result = await api.importHuggingFaceLora(HF_URL)
        } catch (e) {
          console.error('[Blend] Transition LoRA download request failed:', e)
          return
        }
        // Poll the LoRA list until the new file appears (download runs in
        // a backend thread). Cap at ~3 min total.
        const expectedFilename = result?.filename
        for (let i = 0; i < 90; i++) {
          await new Promise(r => setTimeout(r, 2000))
          const refreshed = await api.fetchLoras(modelType)
          loras = refreshed.loras
          const found = expectedFilename
            ? loras.find(l => l === expectedFilename || matchesTransitionLora(l))
            : loras.find(matchesTransitionLora)
          if (found) { transitionFilename = found; break }
        }
        if (!transitionFilename) {
          console.warn('[Blend] Transition LoRA download did not complete in time — skipping auto-activation')
          return
        }
        console.log(`[Blend] Transition LoRA ready: ${transitionFilename}`)
        // Refresh the in-store available LoRA list so the UI shows the new file
        try { await get().loadLoras(modelType) } catch { /* non-fatal */ }
      }

      // Step 3: ensure it's in activated_loras (but don't toggle-off if it
      // happens to already be there)
      const activated = (get().params.activated_loras as string[]) || []
      if (!activated.includes(transitionFilename)) {
        get().toggleLora(transitionFilename)
        console.log(`[Blend] Auto-activated transition LoRA: ${transitionFilename}`)
      }
    } catch (e) {
      console.error('[Blend] ensureTransitionLoraForBlend failed:', e)
    }
  },

  ensureEditAnythingLora: async () => {
    const state = get()
    const modelType = state.params.model_type as string
    if (!modelType || !modelType.startsWith('ltx2')) return

    const HF_URL = 'https://huggingface.co/Alissonerdx/LTX-LoRAs'
    // Must match EDIT_ANYTHING_LORA_FILENAME in app/launch.py. The endpoint
    // will activate this server-side regardless of the client's LoRA list,
    // so we only need to ensure the file is present on disk before the
    // user hits Generate.
    const EDIT_ANYTHING_FILENAME =
      'ltx23_edit_anything_global_rank128_v1_9000steps_adamw.safetensors'
    const matchesEditAnything = (name: string) =>
      name === EDIT_ANYTHING_FILENAME ||
      /edit_anything.*9000steps/i.test(name)

    try {
      const { loras } = await api.fetchLoras(modelType)
      const already = loras.find(matchesEditAnything)
      if (already) return

      console.log('[EditAnything] LoRA not found locally — downloading from HuggingFace')
      try {
        await api.importHuggingFaceLora(HF_URL, undefined, EDIT_ANYTHING_FILENAME)
      } catch (e) {
        console.error('[EditAnything] LoRA download request failed:', e)
        return
      }
      // Poll every 2s until the file appears (up to ~3 min)
      for (let i = 0; i < 90; i++) {
        await new Promise(r => setTimeout(r, 2000))
        const refreshed = await api.fetchLoras(modelType)
        if (refreshed.loras.find(matchesEditAnything)) {
          console.log(`[EditAnything] LoRA ready: ${EDIT_ANYTHING_FILENAME}`)
          try { await get().loadLoras(modelType) } catch { /* non-fatal */ }
          return
        }
      }
      console.warn('[EditAnything] LoRA download did not complete in time')
    } catch (e) {
      console.error('[EditAnything] ensureEditAnythingLora failed:', e)
    }
  },

  setLoraWeight: (filename, phaseIndex, value) => {
    const { params, loraWeights, modelOptions, generationMode, editSubMode } = get()
    const newWeights = { ...loraWeights }
    if (!newWeights[filename]) return
    const recastSinglePhase = generationMode === 'avatar' && editSubMode === 'recast'
    const phases = recastSinglePhase ? 1 : Math.max(1, modelOptions?.guidance_max_phases ?? 1)
    if (phaseIndex < 0 || phaseIndex >= phases) return
    const currentWeights = newWeights[filename]
    newWeights[filename] = Array.from(
      { length: phases },
      (_, i) => currentWeights[i] ?? currentWeights[currentWeights.length - 1] ?? 1.0,
    )
    newWeights[filename][phaseIndex] = value

    // Reserialize
    const multipliers = params.activated_loras.map(name => {
      const w = newWeights[name] || [1.0]
      return Array.from(
        { length: phases },
        (_, i) => w[i] ?? w[w.length - 1] ?? 1.0,
      ).map(v => v.toFixed(2)).join(';')
    }).join(' ')

    set(s => ({
      loraWeights: newWeights,
      params: { ...s.params, loras_multipliers: multipliers },
    }))
    // Persist LoRA state
    const s = get()
    const mode = s.generationMode
    const updatedLoraPerMode = {
      ...s.savedLoraPerMode,
      [mode]: { activated_loras: s.params.activated_loras, loras_multipliers: multipliers, loraWeights: newWeights, availableLoras: s.availableLoras },
    }
    set({ savedLoraPerMode: updatedLoraPerMode })
    _saveSettings({ generationMode: mode, selectedModelPerMode: s.selectedModelPerMode, savedParamsPerMode: s.savedParamsPerMode, savedLoraPerMode: updatedLoraPerMode, savedPromptPerMode: s.savedPromptPerMode }, s.loraIdByFilename)
  },

  // Presets
  presets: [],
  presetsLoading: false,

  loadPresets: async () => {
    set({ presetsLoading: true })
    try {
      const { presets } = await api.fetchPresets()
      set({ presets })
    } catch (e) {
      console.error('Failed to load presets:', e)
    } finally {
      set({ presetsLoading: false })
    }
  },

  savePreset: async (name) => {
    const { params, loraWeights, generationMode } = get()
    try {
      const preset = await api.createPreset({
        name,
        mode: generationMode,
        model_type: params.model_type,
        prompt: '',
        activated_loras: params.activated_loras,
        loras_multipliers: params.loras_multipliers,
        lora_weights: loraWeights,
        params: {
          num_inference_steps: params.num_inference_steps,
          guidance_scale: params.guidance_scale,
          resolution: params.resolution,
          seed: params.seed,
          negative_prompt: params.negative_prompt,
          flow_shift: params.flow_shift,
          self_refiner_setting: params.self_refiner_setting,
          stage2_steps: params.stage2_steps,
        },
      })
      set(s => ({ presets: [...s.presets, preset] }))
    } catch (e) {
      console.error('Failed to save preset:', e)
    }
  },

  loadPreset: (preset) => {
    const newParams: Partial<GenerateParams> = {
      activated_loras: preset.activated_loras,
      loras_multipliers: preset.loras_multipliers,
      ...(preset.params as Partial<GenerateParams>),
    }
    set(s => ({
      params: { ...s.params, ...newParams },
      loraWeights: preset.lora_weights || {},
    }))
  },

  deletePreset: async (id) => {
    try {
      await api.deletePreset(id)
      set(s => ({ presets: s.presets.filter(p => p.id !== id) }))
    } catch (e) {
      console.error('Failed to delete preset:', e)
    }
  },

  // Model options
  modelOptions: null,
  modelOptionsLoading: false,

  loadModelOptions: async (modelType) => {
    const seq = ++_modelOptionsSeq
    set({ modelOptionsLoading: true })
    try {
      const options = await api.fetchModelOptions(modelType)
      // Staleness guard: a newer loadModelOptions call was issued while this
      // fetch was in flight (rapid model switching, or a settings restore
      // that jumped models). Applying a superseded response would clobber
      // params (default steps/guidance) and modelOptions with the WRONG
      // model's values — last requested wins.
      if (seq !== _modelOptionsSeq) return
      const activeState = get()
      const { durationSeconds, slidingWindowSeconds } = activeState
      const fps = options.fps || 16
      // Set overlap from model defaults
      const swDefaults = (options as unknown as Record<string, unknown>).sliding_window_defaults as Record<string, number> | undefined
      const overlapDefault = swDefaults?.overlap_default ?? 5
      const discardDefault = swDefaults?.discard_last_frames ?? 0
      const minimumDuration = Math.max(1, (options.frames_minimum || fps) / fps)
      const nativeMaximumDuration = options.frames_maximum
        ? options.frames_maximum / fps
        : null
      const h3ReferenceSequence = (
        options.omni_reference === true
        && activeState.params.minimax_h3_reference_sequence === true
      )
      const isH3 = String(options.architecture || '').startsWith('minimax_h3')
      const maximumDuration = options.omni_reference === true
        ? (nativeMaximumDuration && !h3ReferenceSequence
            ? nativeMaximumDuration
            : Number.POSITIVE_INFINITY)
        : (!options.sliding_window && nativeMaximumDuration
            ? nativeMaximumDuration
            : Number.POSITIVE_INFINITY)
      let nextDurationSeconds = Math.min(
        maximumDuration,
        Math.max(minimumDuration, durationSeconds),
      )
      if (
        options.sliding_window
        && nativeMaximumDuration
        && nextDurationSeconds <= Math.round(nativeMaximumDuration * 10) / 10
      ) {
        // H3's native ceiling is 14.375s but the UI displays one decimal.
        // Treat displayed 14.4s as that same one-window endpoint instead of
        // scheduling a second minimum-size pass for one rounded frame.
        nextDurationSeconds = Math.min(
          nextDurationSeconds,
          nativeMaximumDuration,
        )
      }
      let nextWindowFrames = Math.round(slidingWindowSeconds * fps)
      if (options.sliding_window && swDefaults?.window_default != null) {
        nextWindowFrames = swDefaults.window_default
      }
      if (options.sliding_window && swDefaults) {
        nextWindowFrames = Math.max(
          swDefaults.window_min ?? 1,
          Math.min(swDefaults.window_max ?? nextWindowFrames, nextWindowFrames),
        )
      } else if (!options.sliding_window) {
        nextWindowFrames = h3ReferenceSequence && options.frames_maximum
          ? options.frames_maximum
          : Math.round(nextDurationSeconds * fps)
      }
      let nextWindowSeconds = nextWindowFrames / fps
      let nextWindowLocked = false
      const paramUpdates: Record<string, unknown> = {
        guidance_phases: options.guidance_max_phases,
        video_length: Math.round(nextDurationSeconds * fps),
        sliding_window_size: nextWindowFrames,
        sliding_window_overlap: overlapDefault,
        sliding_window_discard_last_frames: discardDefault,
      }
      let nextResolutionPreset = activeState.resolutionPreset
      let nextAspectRatio = activeState.aspectRatio
      const modelPresetOrder = options.resolution_preset_order || []
      if (modelPresetOrder.length > 0) {
        if (!modelPresetOrder.includes(nextResolutionPreset)) {
          // A model-specific list can contain an expensive experimental tier
          // at the end. Select its ordinary 720p tier when the previous
          // model's preset is unavailable instead of silently jumping to the
          // largest canvas.
          nextResolutionPreset = modelPresetOrder.includes('720p')
            ? '720p'
            : modelPresetOrder[0]
        }
        if (nextAspectRatio === 'auto' && !options.supports_auto_aspect) {
          nextAspectRatio = '16:9'
        }
        paramUpdates.resolution = resolveResolution(
          options,
          nextResolutionPreset,
          nextAspectRatio,
        )
      } else if (
        nextAspectRatio === 'auto'
        && activeState.generationMode !== 'image'
        && !options.supports_auto_aspect
      ) {
        nextAspectRatio = '16:9'
        paramUpdates.resolution = resolveResolution(
          options,
          nextResolutionPreset,
          nextAspectRatio,
        )
      }
      if (isH3) {
        const selectedResolution = String(
          paramUpdates.resolution || activeState.params.resolution || '',
        )
        const overrideKey = h3WindowOverrideKey(modelType, selectedResolution)
        const savedOverride = activeState.h3WindowOverrides[overrideKey]
        const memoryPolicy = options.omni_reference === true
          ? options.omni_sequence_memory_policy
          : options.sliding_window_memory_policy
        const recommendation = h3ReferenceSequence
          ? recommendedH3OmniSequenceProfile(
              memoryPolicy,
              selectedResolution,
              activeState.systemStats?.gpu.vram_total_gb ?? 0,
              options.frames_minimum ?? 124,
              options.frames_maximum ?? 345,
              options.frames_steps ?? 17,
            )
          : recommendedH3PassProfile(
              memoryPolicy,
              selectedResolution,
              activeState.systemStats?.gpu.vram_total_gb ?? 0,
            )
        const selectedFrames = savedOverride ?? recommendation?.frames
        if (selectedFrames != null) {
          nextWindowFrames = normalizeH3NativeFrames(
            selectedFrames,
            options.frames_minimum ?? 124,
            options.frames_maximum ?? 345,
            options.frames_steps ?? 17,
          )
          nextWindowSeconds = nextWindowFrames / fps
          paramUpdates.sliding_window_size = nextWindowFrames
        }
        nextWindowLocked = savedOverride != null
        paramUpdates.sliding_window_memory_override = nextWindowLocked
        if (options.omni_reference === true) {
          paramUpdates.minimax_h3_sequence_memory_override = nextWindowLocked
          if (h3ReferenceSequence) {
            paramUpdates.minimax_h3_sequence_clip_frames = nextWindowFrames
          }
        }
        const multiWindowEnabled = options.omni_reference === true
          ? h3ReferenceSequence
          : activeState.params.minimax_h3_multi_window === true
        if (!multiWindowEnabled) {
          nextDurationSeconds = Math.min(nextDurationSeconds, nextWindowSeconds)
          paramUpdates.video_length = Math.round(nextDurationSeconds * fps)
        }
      }
      // Apply model defaults for inference steps and guidance scale
      if (options.default_num_inference_steps != null) {
        paramUpdates.num_inference_steps = options.default_num_inference_steps
      }
      if (options.default_guidance_scale != null) {
        paramUpdates.guidance_scale = options.default_guidance_scale
      }
      if (options.minimax_h3_text_encoder_choices?.length) {
        const currentEncoder = get().params.minimax_h3_text_encoder
        const valid = options.minimax_h3_text_encoder_choices.some(
          choice => choice.value === currentEncoder
        )
        if (!valid) {
          paramUpdates.minimax_h3_text_encoder = (
            options.minimax_h3_text_encoder_default
            || options.minimax_h3_text_encoder_choices[0].value
          )
        }
      }
      if (options.ltx25_video_vae_choices?.length) {
        const currentVideoVae = get().params.ltx25_video_vae
        const valid = options.ltx25_video_vae_choices.some(
          choice => choice.value === currentVideoVae
        )
        if (!valid) {
          paramUpdates.ltx25_video_vae = (
            options.ltx25_video_vae_default
            || options.ltx25_video_vae_choices[0].value
          )
        }
      }
      if (options.minimax_h3_turbo) {
        const turboPresets = options.minimax_h3_turbo.presets?.length
          ? options.minimax_h3_turbo.presets
          : [{
              id: options.minimax_h3_turbo.preset_id,
              filename: options.minimax_h3_turbo.filename,
              steps: options.minimax_h3_turbo.steps,
            }]
        const requestedPresetId = get().params.minimax_h3_turbo_preset
        const selectedPreset = (
          turboPresets.find(preset => preset.id === requestedPresetId)
          || turboPresets.find(preset => preset.id === options.minimax_h3_turbo?.preset_id)
          || turboPresets[0]
        )
        paramUpdates.minimax_h3_turbo_preset = selectedPreset.id
        // A restored Turbo preset always displays the same step count the
        // backend will enforce. This also closes a race where model defaults
        // (20 steps) arrive after the user checks Turbo (6 steps).
        if (get().params.minimax_h3_turbo_mode === true) {
          paramUpdates.num_inference_steps = selectedPreset.steps
        }
      } else {
        // Model switches preserve most Studio params. Never carry the Full-H3
        // Turbo flag invisibly into a Pruned H3 or unrelated model.
        paramUpdates.minimax_h3_turbo_mode = false
        paramUpdates.minimax_h3_turbo_preset = undefined
      }
      // TTS default duration. Prefer the model's declared `default` (DramaBox
      // uses 0 = auto-derive from prompt); fall back to `max` (legacy behavior
      // for older TTS models that didn't declare a default), then 600.
      const ttsDefaults: Record<string, unknown> = {}
      if (options.audio_only && options.duration_slider) {
        const ds = options.duration_slider
        ttsDefaults.durationSeconds = ds.default ?? ds.max ?? 600
      }
      // Clamp current voice count to the new model's max_voice_count (e.g.
      // user had 5 voices on Kugel, switches to Scenema which caps at 2 —
      // trim slots 3-5 so the UI doesn't show ghost voices that the backend
      // would silently ignore).
      const newMaxVoiceCount = ((options as { max_voice_count?: number }).max_voice_count) ?? 6
      const currentVoiceCount = get().ttsVoiceCount
      if (currentVoiceCount > newMaxVoiceCount) {
        const trimmedVoices = get().ttsVoices.slice(0, newMaxVoiceCount)
        ttsDefaults.ttsVoiceCount = newMaxVoiceCount
        ttsDefaults.ttsVoices = trimmedVoices
        // Re-derive audio_prompt_type from the clamped count using the new
        // model's selection list.
        const selection = (options.audio_prompt_type_sources?.selection as string[] | undefined) || ['', 'A', 'AB']
        const audioType = selection[Math.min(newMaxVoiceCount, selection.length - 1)]
        paramUpdates.audio_prompt_type = audioType
      }
      set(s => ({
        ...ttsDefaults,
        modelOptions: options,
        modelOptionsLoading: false,
        durationSeconds: (
          typeof ttsDefaults.durationSeconds === 'number'
            ? ttsDefaults.durationSeconds
            : nextDurationSeconds
        ),
        slidingWindowSeconds: nextWindowSeconds,
        slidingWindowOverlap: overlapDefault,
        slidingWindowLocked: nextWindowLocked,
        resolutionPreset: nextResolutionPreset,
        aspectRatio: nextAspectRatio,
        params: {
          ...s.params,
          ...paramUpdates,
        },
      }))
    } catch {
      // Same staleness rule as the success path — a superseded request's
      // failure must not null out the newer request's options.
      if (seq === _modelOptionsSeq) {
        set({ modelOptions: null, modelOptionsLoading: false })
      }
    }
  },

  // System config
  systemConfig: null,
  systemConfigLoading: false,
  loadSystemConfig: async () => {
    set({ systemConfigLoading: true })
    try {
      const config = await api.fetchSystemConfig()
      set({ systemConfig: config, systemConfigLoading: false })
    } catch (e) {
      console.error('Failed to load system config:', e)
      set({ systemConfigLoading: false })
    }
  },
  updateSystemConfig: async (partial) => {
    try {
      await api.updateSystemConfig(partial)
      set(s => ({
        systemConfig: s.systemConfig ? { ...s.systemConfig, ...partial } : null,
      }))
    } catch (e) {
      console.error('Failed to update system config:', e)
      get().loadSystemConfig()
    }
  },

  // Hardware detect — see type definition above. Initial value null;
  // populated when AutoPerformanceCard mounts (Settings → System).
  // Refreshed when the user clicks Re-detect on the auto card.
  systemDetect: null,
  loadSystemDetect: async () => {
    try {
      const detect = await api.fetchSystemDetect()
      set({ systemDetect: detect })
    } catch (e) {
      console.error('Failed to load system detect:', e)
    }
  },

  // Live hardware telemetry (HardwareStatusBar). Polled ~2s from the
  // component while mounted. Swallows a single failed tick (e.g. backend
  // restarting) instead of spamming the console at 2s cadence.
  systemStats: null,
  loadSystemStats: async () => {
    try {
      const stats = await api.fetchSystemStats()
      set({ systemStats: stats })
    } catch {
      /* transient poll failure — ignore this tick */
    }
  },

  // Settings tab
  settingsTab: 'performance' as SettingsTab,
  setSettingsTab: (tab) => set({ settingsTab: tab }),

  // Services config
  servicesConfig: null,
  servicesConfigLoading: false,
  loadServicesConfig: async () => {
    set({ servicesConfigLoading: true })
    try {
      const config = await api.fetchServicesConfig()
      set({ servicesConfig: config, servicesConfigLoading: false })
      if (
        config.nsfw_mode
        && _modelVisibilityHydrated
        && get().models.length > 0
      ) {
        set(s => {
          const next = _enableUninitializedMatureModels(
            s.models,
            s.enabledModels,
          )
          if (!next) return s
          _saveEnabledModels(next)
          return { enabledModels: next }
        })
      }
    } catch (e) {
      console.error('Failed to load services config:', e)
      set({ servicesConfigLoading: false })
    }
  },
  updateServicesConfig: async (partial) => {
    try {
      await api.updateServicesConfig(partial)
      get().loadServicesConfig()
      // Newly-discovered Mature models appear once when Mature Mode is
      // enabled. Previously initialized models retain the user's whitelist.
      if (partial.nsfw_mode === true && _modelVisibilityHydrated) {
        set(s => {
          const next = _enableUninitializedMatureModels(
            s.models,
            s.enabledModels,
          )
          if (!next) return s
          _saveEnabledModels(next)
          return { enabledModels: next }
        })
      }
    } catch (e) {
      console.error('Failed to update services config:', e)
      get().loadServicesConfig()
    }
  },

  // LLM state
  llmStatus: null,
  llmLoading: false,
  llmModels: [],
  loadLlmStatus: async () => {
    try {
      const status = await api.fetchLlmStatus()
      set({ llmStatus: status })
    } catch (e) {
      console.error('Failed to load LLM status:', e)
    }
  },
  loadLlmModels: async () => {
    try {
      const data = await api.fetchLlmModels()
      set({ llmModels: data.models })
    } catch (e) {
      console.error('Failed to load LLM models:', e)
    }
  },
  loadLlm: async () => {
    set({ llmLoading: true })
    try {
      const result = await api.loadLlm()
      set({ llmStatus: { loaded: result.loaded, model_id: result.model_id, device: result.device, provider: result.provider || '' }, llmLoading: false })
    } catch (e) {
      console.error('Failed to load LLM:', e)
      set({ llmLoading: false })
    }
  },
  unloadLlm: async () => {
    try {
      await api.unloadLlm()
      set({ llmStatus: { loaded: false, model_id: null, device: null, provider: '' } })
    } catch (e) {
      console.error('Failed to unload LLM:', e)
    }
  },

  // Prompt enhancement
  isEnhancing: false,
  promptEnhanceError: null,
  h3WindowPlan: null,
  updateH3WindowPrompt: (index, prompt) => set(s => {
    if (!s.h3WindowPlan || index < 0 || index >= s.h3WindowPlan.windows.length) return {}
    const windows = s.h3WindowPlan.windows.map((window, windowIndex) => (
      windowIndex === index ? { ...window, prompt } : window
    ))
    return {
      h3WindowPlan: {
        ...s.h3WindowPlan,
        windows,
        window_prompts: windows.map(window => window.prompt),
      },
    }
  }),
  clearH3WindowPlan: () => set({ h3WindowPlan: null }),
  enhancePrompt: async (ttsMode?: string) => {
    const state = get()
    const { params, generationMode, startImage, endImage, imageRefs } = state
    if (!params.prompt.trim()) return
    if (
      state.modelOptions?.omni_reference === true
      && params.minimax_h3_reference_sequence === true
      && params.minimax_h3_sequence_prompt_mode === 'manual'
    ) {
      set({
        promptEnhanceError: 'Manual Omni sequence mode uses each prompt line exactly as written. Switch Window prompts to Auto plan to use the LLM planner.',
      })
      return
    }
    if (
      String(state.modelOptions?.architecture || '').startsWith('minimax_h3')
      && state.modelOptions?.omni_reference !== true
      && params.minimax_h3_multi_window === true
      && params.minimax_h3_window_storyboard === false
    ) {
      set({
        promptEnhanceError: 'Manual H3 multi-window mode uses each prompt line as written. Switch Window prompts to Auto plan to use the LLM planner.',
      })
      return
    }
    if (
      state.modelOptions?.multi_window_sequence_controls === true
      && params.ltx_multi_window === true
      && params.ltx_window_prompt_mode === 'manual'
    ) {
      set({
        promptEnhanceError: 'Manual LTX multi-window mode uses each prompt line exactly as written. Switch Window prompts to Auto plan to use the LLM planner.',
      })
      return
    }
    set({ isEnhancing: true, promptEnhanceError: null })
    try {
      // Collect images relevant to the CURRENT mode only
      const imagePaths: string[] = []
      let referenceContext: string | undefined
      const isOmniReference = state.modelOptions?.omni_reference === true
      const isH3FirstLast = (
        state.modelOptions?.architecture?.startsWith('minimax_h3') === true
        && !isOmniReference
      )
      const isLtxSequence = state.modelOptions?.multi_window_sequence_controls === true
      const injectedPositions = String(params.frames_positions || '').split(/[\s,]+/).filter(Boolean)
      const injectedKeyframes = (
        isH3FirstLast
        && String(params.video_prompt_type || '').includes('KFI')
        && Array.isArray(params.image_refs)
      ) ? params.image_refs
          .map((path, index) => ({ path, position: injectedPositions[index] || '' }))
          .filter(item => !!item.path && !!item.position)
        : []

      if (isOmniReference) {
        let pictureIndex = 0
        let videoIndex = 0
        let audioIndex = 0
        const labelLines: string[] = []
        for (const reference of params.minimax_h3_references ?? []) {
          const note = (reference.role || reference.filename || 'reference').trim()
          if (reference.type === 'audio') {
            const intent = reference.audio_intent ?? 'voice'
            if (intent === 'drive') {
              labelLines.push(`Exact target soundtrack: ${note}; intent=AUDIO REUSE / PERFORMANCE DRIVER; retention=fully_preserved; preserve its waveform and audible timeline exactly and synchronize visible action and lip movement to it; this is target conditioning rather than a numbered Omni audio reference`)
            } else if (intent === 'style') {
              labelLines.push(`<Audio ${++audioIndex}>: ${note}; intent=AUDIO REFERENCE; retention=weak_reference; borrow only rhythm/style/texture and do not copy the source signal or words`)
            } else {
              labelLines.push(`<Audio ${++audioIndex}>: ${note}; intent=VOICE REFERENCE; retention=reference; use timbre/emotion/delivery for new scripted dialogue without copying source words, timing, or waveform`)
            }
          } else if (reference.type === 'image') {
            labelLines.push(`<Picture ${++pictureIndex}>: visual identity/appearance reference for ${note}; retention=reference for identity only; do not reproduce its background, framing, composition, or pose`)
            if (reference.path) imagePaths.push(reference.path)
          } else {
            const nextVideoIndex = videoIndex + 1
            if ((reference.has_audio || reference.audio_path) && reference.include_audio !== false) {
              labelLines.push(`<Audio ${++audioIndex}>: soundtrack paired with <Video ${nextVideoIndex}>; intent=AUDIO REUSE / PERFORMANCE DRIVER; retention=partially_copy; preserve its audible timeline and synchronize action to it`)
            }
            videoIndex = nextVideoIndex
            labelLines.push(`<Video ${videoIndex}>: motion/camera/scene/timing reference for ${note}`)
          }
        }
        referenceContext = labelLines.join('\n')
      } else if (generationMode === 'image') {
        // Image mode: send reference images only
        for (const ref of imageRefs) {
          try {
            const uploaded = await api.uploadImage(ref)
            imagePaths.push(uploaded.path)
          } catch { /* best effort */ }
        }
      } else {
        // Video/Avatar mode normally sends the start image. H3 First / Last
        // additionally presents its end and injected frames in the same order
        // the runtime's Qwen conditioner will number them.
        let h3HasStartAttachment = false
        let h3HasEndAttachment = false
        if (startImage) {
          try {
            const uploaded = await api.uploadImage(startImage)
            imagePaths.push(uploaded.path)
            h3HasStartAttachment = true
          } catch { /* best effort */ }
        } else if (params.image_start && typeof params.image_start === 'string') {
          imagePaths.push(params.image_start as string)
          h3HasStartAttachment = true
        }
        if (isH3FirstLast) {
          if (endImage) {
            try {
              const uploaded = await api.uploadImage(endImage)
              imagePaths.push(uploaded.path)
              h3HasEndAttachment = true
            } catch { /* best effort */ }
          } else if (params.image_end && typeof params.image_end === 'string') {
            imagePaths.push(params.image_end)
            h3HasEndAttachment = true
          }
          for (const keyframe of injectedKeyframes) {
            // Reusing the same file at two positions still creates two Qwen
            // picture slots, so preserve duplicates and their ordering.
            imagePaths.push(keyframe.path)
          }

          let pictureIndex = 0
          const alignmentLines: string[] = []
          const h3Fps = state.modelOptions?.fps ?? 24
          const h3Duration = Number(params.video_length || 0) / h3Fps
          if (h3HasStartAttachment) {
            alignmentLines.push(`For the target video, at 0.00 seconds into the target video, <Picture ${++pictureIndex}> (from [Shot 1]) is fully referenced.`)
          }
          if (h3HasEndAttachment) {
            alignmentLines.push(`At ${h3Duration.toFixed(2)} seconds, <Picture ${++pictureIndex}> is the required final-frame destination.`)
          }
          for (const keyframe of injectedKeyframes) {
            const match = /^W1:(\d{1,3})$/i.exec(keyframe.position)
            let localSeconds: number | null = null
            if (match) localSeconds = h3Duration * Math.min(100, Number(match[1])) / 100
            else if (/^\d+$/.test(keyframe.position)) localSeconds = Math.max(0, Number(keyframe.position) - 1) / h3Fps
            else if (/^l$/i.test(keyframe.position)) localSeconds = h3Duration
            const timing = localSeconds == null
              ? `at timeline position ${keyframe.position}`
              : `at ${localSeconds.toFixed(2)} seconds into the target video`
            alignmentLines.push(`${timing}, <Picture ${++pictureIndex}> is fully referenced as an exact injected frame; reach it naturally and continue from it.`)
          }
          referenceContext = alignmentLines.join('\n') || undefined
        }
      }
      // Include duration/window info for video models
      const fps = state.modelOptions?.fps ?? 16
      const swDefaults = (state.modelOptions as Record<string, unknown> | null)?.sliding_window_defaults as Record<string, number> | undefined
      const discardFrames = swDefaults?.discard_last_frames ?? 0
      const overlapSec = state.slidingWindowOverlap / fps
      const discardSec = discardFrames / fps
      const stride = state.slidingWindowSeconds - discardSec - overlapSec
      const supportsSlidingWindows = state.modelOptions?.sliding_window === true
      const windowCount = supportsSlidingWindows
        && (!isH3FirstLast || params.minimax_h3_multi_window === true)
        && (!isLtxSequence || params.ltx_multi_window === true)
        && stride > 0
        && state.durationSeconds > state.slidingWindowSeconds
        ? 1 + Math.ceil((state.durationSeconds - state.slidingWindowSeconds + discardSec) / stride)
        : 1
      const totalFrames = Math.max(1, Math.round(state.durationSeconds * fps))
      const h3NativeMaximumFrames = state.modelOptions?.frames_maximum ?? null
      const h3SequenceBudget = (
        isOmniReference
        && params.minimax_h3_reference_sequence === true
        && h3NativeMaximumFrames != null
      ) ? effectiveH3OmniSequenceFrames({
          policy: state.modelOptions?.omni_sequence_memory_policy,
          resolution: String(params.resolution || ''),
          totalVramGb: state.systemStats?.gpu.vram_total_gb ?? 0,
          minimumFrames: state.modelOptions?.frames_minimum ?? 124,
          maximumFrames: h3NativeMaximumFrames,
          frameStep: state.modelOptions?.frames_steps ?? 17,
          selectedFrames: Math.round(state.slidingWindowSeconds * fps),
          manualOverride: state.slidingWindowLocked,
        }) : null
      const h3SequenceClipFrames = h3SequenceBudget?.frames
        ?? h3NativeMaximumFrames
      const shouldPlanH3Sequence = (
        generationMode === 'video'
        && isOmniReference
        && params.minimax_h3_reference_sequence === true
        && params.minimax_h3_sequence_prompt_mode !== 'manual'
        && h3SequenceClipFrames != null
        && totalFrames > h3SequenceClipFrames
      )
      const h3PlanningSource = (
        typeof params._h3_original_prompt === 'string'
        && params._h3_original_prompt.trim()
      ) || params.prompt

      if (shouldPlanH3Sequence) {
        const plan = await api.planH3Sequence({
          prompt: h3PlanningSource,
          model_type: params.model_type,
          resolution: params.resolution,
          total_frames: totalFrames,
          references: params.minimax_h3_references ?? [],
          sequence_clip_frames: h3SequenceClipFrames,
          sequence_memory_override: state.slidingWindowLocked,
          overlap_frames: state.slidingWindowOverlap,
          sequence_continuity: params.minimax_h3_sequence_continuity !== false,
          camera_coverage: params.minimax_h3_camera_coverage || 'auto',
        })
        const effectiveClipFrames = plan.effective_window_frames
          || plan.window_frames
        set(s => ({
          h3WindowPlan: plan,
          slidingWindowSeconds: effectiveClipFrames / fps,
          params: {
            ...s.params,
            prompt: plan.source_prompt || h3PlanningSource,
            _h3_original_prompt: undefined,
            minimax_h3_sequence_clip_frames: effectiveClipFrames,
            minimax_h3_sequence_memory_override: state.slidingWindowLocked,
          },
          isEnhancing: false,
        }))
        return
      }

      const shouldPlanH3Windows = (
        generationMode === 'video'
        && state.modelOptions?.sliding_window_auto_prompt_pacing === true
        && params.minimax_h3_multi_window === true
        && params.minimax_h3_window_storyboard !== false
        && params.image_mode !== 2
        && windowCount > 1
      )
      if (shouldPlanH3Windows) {
        // The ordinary H3 enhancer writes one complete Context-IR timeline.
        // Multi-window H3 instead needs a structured storyboard whose prompts
        // contain only their own local actions. Endpoint and injected images
        // were collected above in the runtime's stable presentation order.
        const plan = await api.planH3Windows({
          prompt: h3PlanningSource,
          model_type: params.model_type,
          resolution: params.resolution,
          total_frames: totalFrames,
          window_frames: Math.max(1, Math.round(state.slidingWindowSeconds * fps)),
          overlap_frames: state.slidingWindowOverlap,
          discard_frames: discardFrames,
          sliding_window_memory_override: state.slidingWindowLocked,
          has_start_image: !!(startImage || params.image_start),
          has_end_image: !!(endImage || params.image_end),
          image_paths: imagePaths.length > 0 ? imagePaths : undefined,
          injected_keyframes: injectedKeyframes.length > 0 ? injectedKeyframes : undefined,
          camera_coverage: params.minimax_h3_camera_coverage || 'auto',
        })
        const effectiveWindowFrames = plan.effective_window_frames || plan.window_frames
        set(s => ({
          h3WindowPlan: plan,
          slidingWindowSeconds: effectiveWindowFrames / fps,
          // Clicking Enhance on a multi-window H3 First/Last job is an
          // explicit request to plan the idea across those windows. Turn the
          // planner back on even when an old saved setting left legacy mode
          // disabled; otherwise the ordinary H3 enhancer flattens every
          // window into one globally timed screenplay.
          params: {
            ...s.params,
            prompt: plan.source_prompt || h3PlanningSource,
            _h3_original_prompt: undefined,
            sliding_window_size: effectiveWindowFrames,
            minimax_h3_multi_window: true,
            minimax_h3_window_storyboard: true,
          },
          isEnhancing: false,
        }))
        return
      }

      // TTS dialogue needs more tokens for longer conversations
      const maxTokens = (generationMode === 'audio' && ttsMode) ? 2048 : undefined
      const ltxEnhanceSource = (
        generationMode === 'video'
        && isLtxSequence
        && params.ltx_multi_window === true
        && params.ltx_window_prompt_mode !== 'manual'
        && typeof params._ltx_original_prompt === 'string'
        && params._ltx_original_prompt.trim()
      ) ? params._ltx_original_prompt : params.prompt

      const result = await api.llmEnhancePrompt({
        prompt: ltxEnhanceSource,
        mode: generationMode,
        model_type: params.model_type,
        max_new_tokens: maxTokens,
        image_paths: imagePaths.length > 0 ? imagePaths : undefined,
        duration_seconds: (generationMode === 'video' || generationMode === 'avatar') ? state.durationSeconds : undefined,
        window_count: (generationMode === 'video' || generationMode === 'avatar') ? windowCount : undefined,
        window_size_seconds: (generationMode === 'video' || generationMode === 'avatar') ? state.slidingWindowSeconds : undefined,
        activated_loras: params.activated_loras.length > 0 ? params.activated_loras : undefined,
        tts_enhance_mode: ttsMode || undefined,
        tts_voice_count: state.ttsVoiceCount || undefined,
        reference_context: referenceContext,
      })
      const preserveH3Source = (
        generationMode === 'video'
        && state.modelOptions?.architecture?.startsWith('minimax_h3') === true
      )
        ? ((typeof params._h3_original_prompt === 'string'
            && params._h3_original_prompt.trim()) || params.prompt)
        : undefined
      const enhancedLtxLines = result.enhanced
        .replace(/\r\n?/g, '\n')
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean)
      const preserveLtxPlan = (
        generationMode === 'video'
        && isLtxSequence
        && params.ltx_multi_window === true
        && params.ltx_window_prompt_mode !== 'manual'
        && windowCount > 1
        && enhancedLtxLines.length === windowCount
      )
      const ltxSourcePrompt = ltxEnhanceSource
      set(s => ({
        params: {
          ...s.params,
          prompt: preserveLtxPlan ? enhancedLtxLines.join('\n') : result.enhanced,
          ...(preserveH3Source ? { _h3_original_prompt: preserveH3Source } : {}),
          ...(preserveLtxPlan ? {
            _ltx_original_prompt: ltxSourcePrompt,
            ltx_window_prompts: enhancedLtxLines,
          } : {}),
        },
        isEnhancing: false,
      }))
      // Auto-parse speaker names from the enhanced text whenever there are
      // voice slots to fill. Previously gated to dialogue mode only; the user
      // expects monologue enhance ("Peter: Hello world.") to also populate
      // voice slot 1 with "Peter". `force=true` overrides the manual flag
      // — enhance creates a fresh script, so previous user-edited names are
      // no longer relevant.
      if (ttsMode && get().ttsVoiceCount > 0) {
        get()._autoParseSpkeakerNames(result.enhanced, true)
      }
    } catch (e) {
      console.error('Failed to enhance prompt:', e)
      const message = e instanceof Error ? e.message : 'Prompt enhancement failed'
      set({
        isEnhancing: false,
        promptEnhanceError: `Prompt enhancement failed: ${message}`,
      })
    }
  },

  // Director (Music Video Director)
  sidebarMode: 'studio' as const,
  directorStep: 'upload',
  directorAudioFile: null,
  directorAudioPath: null,
  directorAnalysis: null,
  directorPlannedClips: [],
  directorEnergyBias: 0,
  directorClipPlans: [],
  directorSceneDescription: '',
  directorLoading: false,
  directorLoadingMessage: null,
  directorError: null,
  directorReferenceImage: null,
  directorReferenceImagePath: null,
  directorCharacterRefs: [],
  directorCharacterRefPaths: [],
  directorCharacterRefLabels: [],
  directorLocationRefs: [],
  directorLocationRefPaths: [],
  directorLocationRefLabels: [],
  directorVoiceRef: null,
  directorVoiceRefPath: null,
  directorIdentityGuidanceScale: 3.0,
  setDirectorVoiceRef: (file) => {
    if (file) {
      set({ directorVoiceRef: file, directorVoiceRefPath: null })
    } else {
      set({ directorVoiceRef: null, directorVoiceRefPath: null })
    }
  },
  setDirectorIdentityGuidanceScale: (v) => set({ directorIdentityGuidanceScale: v }),
  directorClipImages: [],
  directorSetClipImage: (clipIndex, file) => set(s => {
    const remaining = s.directorClipImages.filter(
      image => image.clipIndex !== clipIndex,
    )
    if (!file) return { directorClipImages: remaining }
    const image: DirectorClipImage = {
      clipIndex,
      prompt: s.directorClipPlans[clipIndex]?.image_prompt || '',
      file,
      filename: file.name,
    }
    return {
      directorClipImages: [...remaining, image].sort(
        (left, right) => left.clipIndex - right.clipIndex,
      ),
    }
  }),
  directorImageGenProgress: null,
  directorSpeakers: [],
  directorSpeakerMappings: [],
  // Defaults per user preference (2026-06): Auto ON (hands-off pipeline is
  // the common flow), Seamless OFF (separate per-clip generations are easier
  // to retake/review than one rolling-window render).
  directorAutoMode: true,
  directorSeamless: false,
  directorShotImageGuidance: 'auto' as DirectorShotImageGuidance,
  directorLlmLog: [],
  directorSkill: null,
  directorMusicSource: null,
  directorMusicModel: 'ace_step_v1_5_xl_sft_lm_4b',
  directorSongDescription: '',
  directorSongInstrumental: false,
  directorSongStyle: '',
  directorSongLyrics: '',
  directorSongDuration: 120,
  directorTrackGenerating: false,
  setDirectorMusicSource: (s) => set({ directorMusicSource: s }),
  setDirectorMusicModel: (modelType) => set({
    directorMusicModel: modelType,
    // The two model families use different caption contracts. Never retain a
    // hidden song plan written for the previously selected generator.
    directorSongStyle: '',
    directorSongLyrics: '',
  }),
  setDirectorSongDescription: (v) => set({ directorSongDescription: v }),
  setDirectorSongInstrumental: (v) => set({ directorSongInstrumental: v }),
  setDirectorSongStyle: (v) => set({ directorSongStyle: v }),
  setDirectorSongLyrics: (v) => set({ directorSongLyrics: v }),
  setDirectorSongDuration: (v) => set({ directorSongDuration: v }),
  directorResolution: '720p' as ResolutionPreset,
  directorAspectRatio: '16:9' as AspectRatio,
  directorVideoInferenceStepsByModel: {},
  directorVideoMaxShotFramesByModel: {},
  directorH3TurboModeByModel: {},
  directorH3TurboPresetByModel: {},
  directorH3SolModeByModel: {},
  directorH3FirstBlockCacheByModel: {},
  directorH3FirstBlockCacheMultiplierByModel: {},
  directorH3FirstBlockCacheWarmupByModel: {},
  shortFilmCharacters: [],
  shortFilmPath: null,
  shortFilmTargetDuration: 30,
  shortFilmNarrative: false,
  llmStreamText: '',
  llmStreamDone: true,
  pipelineId: null,
  pipelineStatus: null,
  pipelinePolling: false,
  setDirectorAutoMode: (v) => set({ directorAutoMode: v }),
  setDirectorSeamless: (v) => set({ directorSeamless: v }),
  setDirectorShotImageGuidance: (v) => set({
    directorShotImageGuidance: v,
    // Selecting "None" must not leave generated images from an earlier
    // choice silently attached to manual video jobs. Users can add fresh
    // per-scene uploads from the review screen after making this choice.
    ...(v === 'prompt_only' ? {
      directorClipImages: [],
      directorImageGenProgress: null,
    } : {}),
  }),
  directorAppendLlmLog: (stage, text) => set(s => {
    const t = (text || '').trim()
    if (!t) return {}
    const last = s.directorLlmLog[s.directorLlmLog.length - 1]
    // Skip exact repeats (the poll can fire the done-transition more than
    // once for the same stream when stages restart back-to-back).
    if (last && last.stage === stage && last.text === t) return {}
    return { directorLlmLog: [...s.directorLlmLog, { stage, text: t }] }
  }),
  setDirectorSkill: (skill) => {
    set({ directorSkill: skill })
    const state = get()
    const selectedVideoModel = state.selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const selectedVideoDefinition = state.models.find(
      model => model.model_type === selectedVideoModel,
    )
    if (directorModelUsesFixedMediaStrength(
      selectedVideoModel,
      selectedVideoDefinition?.architecture,
    )) {
      if (state.params.input_video_strength !== 1.0) {
        state.setParam('input_video_strength', 1.0)
      }
      return
    }
    // Music director default for image-to-video reference strength is
    // 0.7 (loosens the lock to the start frame so motion can develop
    // naturally) rather than 1.0 (rigid frame). Only initialize when
    // the param is unset OR still at the global 1.0 default — preserves
    // any value the user has already adjusted in this session.
    //
    // Goes through setParam (not a direct `params` write) so the value
    // propagates into savedParamsPerMode.video — that's what the
    // Director pipeline reads when building video_params for the
    // submission. Without this routing the slider would show 0.7 but
    // the pipeline would still send 1.0.
    if (skill === 'music_video') {
      const current = get().params.input_video_strength
      if (current == null || current === 1.0) {
        get().setParam('input_video_strength', 0.7)
      }
    }
  },
  setDirectorResolution: (preset) => set({ directorResolution: preset }),
  setDirectorAspectRatio: (ratio) => set({ directorAspectRatio: ratio }),
  setDirectorVideoInferenceSteps: (modelType, steps) => set(s => {
    const next = { ...s.directorVideoInferenceStepsByModel }
    if (steps == null || !Number.isFinite(steps)) {
      delete next[modelType]
    } else {
      next[modelType] = Math.max(1, Math.min(50, Math.round(steps)))
    }
    return { directorVideoInferenceStepsByModel: next }
  }),
  setDirectorVideoMaxShotFrames: (modelType, frames) => set(s => {
    const next = { ...s.directorVideoMaxShotFramesByModel }
    if (frames == null || !Number.isFinite(frames) || frames <= 0) {
      delete next[modelType]
    } else {
      next[modelType] = Math.round(frames)
    }
    return { directorVideoMaxShotFramesByModel: next }
  }),
  setDirectorH3TurboMode: (modelType, enabled) => set(s => ({
    directorH3TurboModeByModel: {
      ...s.directorH3TurboModeByModel,
      [modelType]: enabled,
    },
  })),
  setDirectorH3TurboPreset: (modelType, presetId) => set(s => ({
    directorH3TurboPresetByModel: {
      ...s.directorH3TurboPresetByModel,
      [modelType]: presetId,
    },
  })),
  setDirectorH3SolMode: (modelType, enabled) => set(s => ({
    directorH3SolModeByModel: {
      ...s.directorH3SolModeByModel,
      [modelType]: enabled,
    },
  })),
  setDirectorH3FirstBlockCache: (modelType, enabled) => set(s => ({
    directorH3FirstBlockCacheByModel: {
      ...s.directorH3FirstBlockCacheByModel,
      [modelType]: enabled,
    },
  })),
  setDirectorH3FirstBlockCacheMultiplier: (modelType, value) => set(s => ({
    directorH3FirstBlockCacheMultiplierByModel: {
      ...s.directorH3FirstBlockCacheMultiplierByModel,
      [modelType]: value,
    },
  })),
  setDirectorH3FirstBlockCacheWarmup: (modelType, value) => set(s => ({
    directorH3FirstBlockCacheWarmupByModel: {
      ...s.directorH3FirstBlockCacheWarmupByModel,
      [modelType]: Math.max(0, Math.min(75, Math.round(value / 5) * 5)),
    },
  })),

  selectDirectorImageModel: (modelType) => {
    if (get().selectedModelPerMode.image === modelType) return
    set(s => ({
      selectedModelPerMode: { ...s.selectedModelPerMode, image: modelType },
      // Model-specific prompts and rendered starts must never survive a
      // pre-planning model change. Keep the uploaded/analyzed source intact.
      directorClipPlans: [],
      directorClipImages: [],
      directorImageGenProgress: null,
      directorError: null,
    }))
    const s = get()
    _saveSettings({
      generationMode: s.generationMode,
      selectedModelPerMode: s.selectedModelPerMode,
      savedParamsPerMode: s.savedParamsPerMode,
      savedLoraPerMode: s.savedLoraPerMode,
      savedPromptPerMode: s.savedPromptPerMode,
    }, s.loraIdByFilename)
  },

  selectDirectorVideoModel: (modelType) => {
    const previousModel = get().selectedModelPerMode.video
    if (previousModel === modelType) return
    set(s => ({
      selectedModelPerMode: { ...s.selectedModelPerMode, video: modelType },
      // The video model determines both prompt rules and the legal frame
      // lattice. Preserve source media and analysis, but invalidate anything
      // derived downstream from those choices.
      directorClipPlans: [],
      directorClipImages: [],
      directorImageGenProgress: null,
      directorError: null,
    }))
    const selectedVideoDefinition = get().models.find(
      model => model.model_type === modelType,
    )
    if (directorModelUsesFixedMediaStrength(
      modelType,
      selectedVideoDefinition?.architecture,
    ) && get().params.input_video_strength !== 1.0) {
      get().setParam('input_video_strength', 1.0)
    }
    get().loadModelOptions(modelType)
    const current = get()
    if (
      current.directorAnalysis
      && (current.directorStep === 'structure' || current.directorStep === 'style')
    ) {
      // Rebuild clip lengths against the newly selected model without
      // re-uploading or re-transcribing the user's audio.
      void current.directorSetEnergyBias(current.directorEnergyBias)
    }
    const s = get()
    _saveSettings({
      generationMode: s.generationMode,
      selectedModelPerMode: s.selectedModelPerMode,
      savedParamsPerMode: s.savedParamsPerMode,
      savedLoraPerMode: s.savedLoraPerMode,
    }, s.loraIdByFilename)
  },

  directorSetLora: (mode, activated_loras, loras_multipliers, loraWeights, availableLoras) => {
    const s = get()
    const updatedLoraPerMode = {
      ...s.savedLoraPerMode,
      [mode]: { activated_loras, loras_multipliers, loraWeights, availableLoras },
    }
    set({ savedLoraPerMode: updatedLoraPerMode })
    _saveSettings({
      generationMode: s.generationMode,
      selectedModelPerMode: s.selectedModelPerMode,
      savedParamsPerMode: s.savedParamsPerMode,
      savedLoraPerMode: updatedLoraPerMode,
    }, s.loraIdByFilename)
  },

  directorSetSpeakerMapping: (speakerId, name, role) => {
    set(s => ({
      directorSpeakerMappings: s.directorSpeakerMappings.map(m =>
        m.speakerId === speakerId ? { ...m, name, role } : m
      ),
    }))
  },

  directorInsertSpeakerMention: (speakerId) => {
    set(s => ({
      directorSceneDescription: s.directorSceneDescription
        ? `${s.directorSceneDescription} @${speakerId}`
        : `@${speakerId}`,
    }))
  },

  setSidebarMode: (mode) => {
    if (mode === 'director') {
      const { sidebarMode, directorAudioFile } = get()
      if (sidebarMode !== 'director') {
        if (!directorAudioFile) {
          set({ sidebarMode: 'director', directorStep: 'upload', directorError: null })
        } else {
          set({ sidebarMode: 'director' })
        }
      }
    } else {
      set({ sidebarMode: 'studio' })
    }
  },

  directorUploadAndAnalyze: async (file) => {
    set({
      directorLoading: true,
      directorLoadingMessage: 'Uploading audio...',
      directorError: null,
      directorAudioFile: file,
      directorStep: 'analyze',
    })
    try {
      const uploaded = await api.uploadAudio(file)
      await get().directorAnalyzeAndPlan(uploaded.path, { transcribe: true })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Upload failed'
      console.error('Director upload failed:', e)
      set({ directorLoading: false, directorLoadingMessage: null, directorError: msg, directorStep: 'upload' })
    }
  },

  // Shared analyze → section-classify → plan-structure chain. Works for an
  // UPLOADED track or a GENERATED one — both converge here with an audio path
  // on disk and land on the 'structure' step, so everything downstream is
  // identical regardless of where the audio came from.
  directorAnalyzeAndPlan: async (audioPath, opts) => {
    const transcribe = opts?.transcribe !== false
    set({
      directorAudioPath: audioPath,
      directorLoading: true,
      directorLoadingMessage: 'Analyzing audio...',
      directorError: null,
      directorStep: 'analyze',
    })
    // Poll the backend's audio-analyze status during the long synchronous
    // /audio/analyze call so the UI can show "Loading transcription model
    // (first use downloads ~300MB)..." vs "Transcribing audio..." instead of
    // a single "Analyzing audio..." for the entire first-run wait. Cleared on
    // success or failure in the finally block.
    let analyzePoll: ReturnType<typeof setInterval> | null = null
    const startAnalyzePolling = () => {
      analyzePoll = setInterval(async () => {
        try {
          const status = await api.fetchAudioAnalyzeStatus()
          if (!status.step) return  // No analyze in flight or just cleared
          set({ directorLoadingMessage: `${status.detail}...` })
        } catch { /* polling errors are non-fatal */ }
      }, 1000)
    }
    const stopAnalyzePolling = () => {
      if (analyzePoll !== null) {
        clearInterval(analyzePoll)
        analyzePoll = null
      }
    }
    try {
      startAnalyzePolling()
      let analysis = await api.analyzeAudio({
        audio_path: audioPath,
        transcribe,
        extract_vocals: transcribe,
        lyrics_hint: opts?.lyricsHint || undefined,
      })
      stopAnalyzePolling()

      // Try LLM-based section classification (falls back to heuristic)
      if (analysis.lyrics && analysis.lyrics.length > 0) {
        try {
          set({ directorLoadingMessage: 'Identifying sections (LLM)...' })
          const classified = await api.classifySections({ analysis })
          analysis = {
            ...analysis,
            sections: classified.sections,
            song_structure: classified.song_structure || null,
          }
        } catch {
          // LLM not available — keep heuristic labels
        }
      }

      set({ directorAnalysis: analysis })

      // Extract unique speakers from diarized lyrics
      const speakers: string[] = []
      if (analysis.lyrics) {
        const seen = new Set<string>()
        for (const seg of analysis.lyrics) {
          if (seg.speaker && !seen.has(seg.speaker)) {
            seen.add(seg.speaker)
            speakers.push(seg.speaker)
          }
        }
      }
      const speakerMappings: SpeakerMapping[] = speakers.map(s => ({
        speakerId: s,
        name: '',
        role: '' as const,
      }))
      set({ directorSpeakers: speakers, directorSpeakerMappings: speakerMappings })

      // Plan beat-aligned clip structure
      set({ directorLoadingMessage: 'Planning clip structure...' })
      const structure = await api.planClipStructure({
        analysis,
        energy_bias: get().directorEnergyBias,
        fps: get().modelOptions?.fps ?? 16,
        frames_steps: get().modelOptions?.frames_steps ?? 4,
        frames_minimum: get().modelOptions?.frames_minimum ?? 5,
        // Authoritative: the Director's video model (modelOptions above may
        // belong to a music model — e.g. ACE-Step after generating a track —
        // whose fps fallback of 16 used to shrink clips by 16/25).
        video_model: get().selectedModelPerMode.video || undefined,
      })
      // Music Video skips the manual clip-structure review step entirely —
      // the beat-aligned clips are used as-is. Short Film keeps it.
      const skipStructure = get().directorSkill === 'music_video'
      set({
        directorPlannedClips: structure.clips,
        directorStep: skipStructure ? 'style' : 'structure',
        directorLoading: false,
        directorLoadingMessage: null,
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Analysis failed'
      console.error('Director analysis failed:', e)
      set({ directorLoading: false, directorLoadingMessage: null, directorError: msg, directorStep: 'upload' })
      throw e
    } finally {
      stopAnalyzePolling()
    }
  },

  // Music Video: write the song (Style + Lyrics) from the description, with
  // the optional reference image informing the style via the vision LLM.
  // Throws on failure so the UI can surface it inline.
  directorWriteSong: async () => {
    const s = get()
    const description = s.directorSongDescription.trim()
    if (!description) return
    let refPath = s.directorReferenceImagePath
    if (!refPath && s.directorReferenceImage) {
      try {
        refPath = (await api.uploadImage(s.directorReferenceImage)).path
        set({ directorReferenceImagePath: refPath })
      } catch { /* image upload is best-effort */ }
    }
    set({ directorError: null })
    const r = await api.writeSong({
      description,
      instrumental: s.directorSongInstrumental,
      duration_seconds: s.directorSongDuration,
      reference_image_path: refPath || undefined,
      model_type: s.directorMusicModel,
    })
    set({
      directorSongStyle: r.style || '',
      directorSongLyrics: s.directorSongInstrumental ? '[Instrumental]' : (r.lyrics || ''),
    })
  },

  // Music Video: generate the track (writing the song first if the user only
  // gave a description), then hand off to the SAME analyze → plan-structure
  // chain the upload flow uses. In Auto mode, continue straight into the
  // pipeline so it's fully hands-off.
  directorGenerateTrack: async () => {
    const s = get()
    const instrumental = s.directorSongInstrumental
    const description = s.directorSongDescription.trim()
    const style = s.directorSongStyle.trim()
    const lyrics = s.directorSongLyrics.trim()
    if (!description && !style && !lyrics) {
      set({ directorError: 'Describe your song (or fill in Style / Lyrics) first.' })
      return
    }
    // Upload the reference image so it can inform BOTH the music and visuals.
    let refPath = s.directorReferenceImagePath
    if (!refPath && s.directorReferenceImage) {
      try {
        refPath = (await api.uploadImage(s.directorReferenceImage)).path
        set({ directorReferenceImagePath: refPath })
      } catch { /* image upload is best-effort */ }
    }
    set({
      directorTrackGenerating: true,
      directorError: null,
      directorLoading: true,
      directorLoadingMessage: 'Generating music track…',
      directorStep: 'analyze',
    })
    try {
      // generateMusic is a BLOCKING POST — the browser only learns the job id
      // when it finishes — but the backend registers the job immediately. Run
      // the same discovery a fresh browser uses at page load (reconnectJobs:
      // deduped, self-polling) so the gallery shows a live placeholder card
      // during the render instead of nothing until LLM planning.
      const trackPromise = api.generateMusic({
        description: description || undefined,
        style: style || undefined,
        lyrics: instrumental ? '[Instrumental]' : (lyrics || undefined),
        instrumental,
        duration_seconds: s.directorSongDuration,
        reference_image_path: refPath || undefined,
        model_type: s.directorMusicModel,
        workspace: get().activeWorkspace || undefined,
      })
      setTimeout(() => { void get().reconnectJobs() }, 1200)
      setTimeout(() => { void get().reconnectJobs() }, 5000)
      const r = await trackPromise
      // Persist the (possibly LLM-written) song back into the editable fields.
      set({
        directorSongStyle: r.style || style,
        directorSongLyrics: instrumental ? '[Instrumental]' : (r.lyrics || lyrics),
        directorTrackGenerating: false,
      })
      // Pre-fill the scene description from the song brief so the visual
      // planner has context. The 'style' step shows it (editable); Auto mode
      // uses it directly.
      if (!get().directorSceneDescription.trim() && description) {
        set({ directorSceneDescription: description })
      }
      // Same analyze → plan-structure chain as the upload flow. Instrumental
      // tracks skip transcription (no lyrics to find). For vocal tracks we
      // KNOW the written lyrics — seed Whisper with them so the timed
      // transcription matches what ACE-Step actually sang.
      await get().directorAnalyzeAndPlan(r.audio_path, {
        transcribe: !instrumental,
        lyricsHint: instrumental ? undefined : (r.lyrics || lyrics || undefined),
      })
      // The song description doubles as the scene description, so the manual
      // 'style' step isn't needed — proceed straight to planning. Auto runs the
      // full server-side pipeline; manual runs the frontend plan→review chain.
      if (get().directorStep === 'style') {
        if (get().directorAutoMode) {
          await get().startDirectorPipeline()
        } else {
          await get().directorPlanPrompts()
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Music generation failed'
      console.error('Director music generation failed:', e)
      set({
        directorTrackGenerating: false,
        directorLoading: false,
        directorLoadingMessage: null,
        directorError: msg,
        directorStep: 'upload',
      })
    }
  },

  directorSetEnergyBias: async (bias) => {
    const { directorAnalysis } = get()
    if (!directorAnalysis) return
    set({ directorLoading: true, directorEnergyBias: bias })
    try {
      const structure = await api.planClipStructure({
        analysis: directorAnalysis,
        energy_bias: bias,
        fps: get().modelOptions?.fps ?? 16,
        frames_steps: get().modelOptions?.frames_steps ?? 4,
        frames_minimum: get().modelOptions?.frames_minimum ?? 5,
        video_model: get().selectedModelPerMode.video || undefined,
      })
      set({ directorPlannedClips: structure.clips, directorLoading: false })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to update structure'
      set({ directorLoading: false, directorError: msg })
    }
  },

  directorConfirmStructure: () => {
    set({ directorStep: 'style', directorLoading: false })
  },

  directorSetReferenceImage: (file) => set({ directorReferenceImage: file }),
  directorAddCharacterRef: (file) => set(s => ({
    directorCharacterRefs: [...s.directorCharacterRefs, file],
    directorCharacterRefLabels: [...s.directorCharacterRefLabels, ''],
  })),
  directorRemoveCharacterRef: (index) => set(s => ({
    directorCharacterRefs: s.directorCharacterRefs.filter((_, i) => i !== index),
    directorCharacterRefPaths: s.directorCharacterRefPaths.filter((_, i) => i !== index),
    directorCharacterRefLabels: s.directorCharacterRefLabels.filter((_, i) => i !== index),
  })),
  directorSetCharacterRefLabel: (index, label) => set(s => {
    const labels = [...s.directorCharacterRefLabels]
    labels[index] = label
    return { directorCharacterRefLabels: labels }
  }),
  directorReorderCharacterRefs: (from, to) => set(s => {
    const refs = [...s.directorCharacterRefs]
    const paths = [...s.directorCharacterRefPaths]
    const labels = [...s.directorCharacterRefLabels]
    const [rF] = refs.splice(from, 1); refs.splice(to, 0, rF)
    const [pF] = paths.splice(from, 1); paths.splice(to, 0, pF)
    const [lF] = labels.splice(from, 1); labels.splice(to, 0, lF)
    return { directorCharacterRefs: refs, directorCharacterRefPaths: paths, directorCharacterRefLabels: labels }
  }),
  directorAddLocationRef: (file) => set(s => ({
    directorLocationRefs: [...s.directorLocationRefs, file],
    directorLocationRefLabels: [...s.directorLocationRefLabels, ''],
  })),
  directorRemoveLocationRef: (index) => set(s => ({
    directorLocationRefs: s.directorLocationRefs.filter((_, i) => i !== index),
    directorLocationRefPaths: s.directorLocationRefPaths.filter((_, i) => i !== index),
    directorLocationRefLabels: s.directorLocationRefLabels.filter((_, i) => i !== index),
  })),
  directorSetLocationRefLabel: (index, label) => set(s => {
    const labels = [...s.directorLocationRefLabels]
    labels[index] = label
    return { directorLocationRefLabels: labels }
  }),
  directorReorderLocationRefs: (from, to) => set(s => {
    const refs = [...s.directorLocationRefs]
    const paths = [...s.directorLocationRefPaths]
    const labels = [...s.directorLocationRefLabels]
    const [rF] = refs.splice(from, 1); refs.splice(to, 0, rF)
    const [pF] = paths.splice(from, 1); paths.splice(to, 0, pF)
    const [lF] = labels.splice(from, 1); labels.splice(to, 0, lF)
    return { directorLocationRefs: refs, directorLocationRefPaths: paths, directorLocationRefLabels: labels }
  }),

  directorSetSceneDescription: (prompt) => set({ directorSceneDescription: prompt }),

  // Helper: upload all Director reference images (main + characters + locations)
  _uploadDirectorRefs: async () => {
    const s = get()
    // Upload main reference
    let refImagePath = s.directorReferenceImagePath
    if (s.directorReferenceImage && !refImagePath) {
      const uploaded = await api.uploadImage(s.directorReferenceImage)
      refImagePath = uploaded.path
      set({ directorReferenceImagePath: refImagePath })
    }
    // Upload character refs
    const charPaths = [...s.directorCharacterRefPaths]
    for (let i = charPaths.length; i < s.directorCharacterRefs.length; i++) {
      const uploaded = await api.uploadImage(s.directorCharacterRefs[i])
      charPaths.push(uploaded.path)
    }
    if (charPaths.length > s.directorCharacterRefPaths.length) {
      set({ directorCharacterRefPaths: charPaths })
    }
    // Upload location refs
    const locPaths = [...s.directorLocationRefPaths]
    for (let i = locPaths.length; i < s.directorLocationRefs.length; i++) {
      const uploaded = await api.uploadImage(s.directorLocationRefs[i])
      locPaths.push(uploaded.path)
    }
    if (locPaths.length > s.directorLocationRefPaths.length) {
      set({ directorLocationRefPaths: locPaths })
    }
    return { refImagePath, charPaths, locPaths }
  },

  directorPlanPrompts: async () => {
    const { directorPlannedClips, directorSceneDescription, directorAnalysis } = get()
    if (!directorPlannedClips.length || !directorSceneDescription.trim()) return
    set({ directorLoading: true, directorError: null, directorStep: 'plan' })
    try {
      // Upload all reference images
      const { refImagePath, charPaths, locPaths } = await get()._uploadDirectorRefs()
      const { directorCharacterRefLabels: charLabels, directorLocationRefLabels: locLabels } = get()
      const extraRefs = {
        ...(charPaths.length > 0 ? { character_ref_paths: charPaths, character_ref_labels: charLabels } : {}),
        ...(locPaths.length > 0 ? { location_ref_paths: locPaths, location_ref_labels: locLabels } : {}),
      }
      const generateShotImages = _directorUsesGeneratedShotImages(get())
      const promptType = generateShotImages ? 'both' : 'video'

      // Build speaker_mappings from user-assigned names (only those with names filled in)
      const speakerMappings: Record<string, { name: string; role: string }> = {}
      for (const m of get().directorSpeakerMappings) {
        if (m.name.trim()) {
          speakerMappings[m.speakerId] = { name: m.name, role: m.role }
        }
      }

      // Generate both image and video prompts
      // ?? not || — an explicit user-toggled `false` must be respected
      // (legacy v1 path); only fall back to true when servicesConfig
      // hasn't loaded yet or the field is undefined.
      const useV2 = get().servicesConfig?.use_director_v2 ?? true
      let plans: Array<{ video_prompt: string; image_prompt: string }>

      if (useV2) {
        // Director v2: structured planning → rendering → validation
        const result = await api.directorV2Plan({
          skill_type: 'music_video',
          clips: directorPlannedClips,
          scene_description: directorSceneDescription,
          lyrics: directorAnalysis?.lyrics ?? undefined,
          bpm: directorAnalysis?.bpm ?? 120,
          reference_image_path: refImagePath ?? undefined,
          ...extraRefs,
          speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
          prompt_type: promptType,
        })
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
      } else {
        // Legacy: direct LLM prompt generation
        const result = await api.planClipPromptsAndImages({
          clips: directorPlannedClips,
          scene_description: directorSceneDescription,
          lyrics: directorAnalysis?.lyrics ?? undefined,
          bpm: directorAnalysis?.bpm ?? 120,
          reference_image_path: refImagePath,
          ...extraRefs,
          speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
          prompt_type: promptType,
        })
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
      }
      set({
        directorClipPlans: plans,
        directorStep: generateShotImages ? 'review' : 'review_video',
        directorLoading: false,
      })

      // Auto mode follows the image selector: generate consistent scene starts
      // with a concrete image model, or go directly to prompt-only video when
      // the selector is None.
      if (get().directorAutoMode) {
        if (generateShotImages) {
          get().directorGenerateStartImages()
        } else {
          get().directorGenerate()
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Planning failed'
      console.error('Director planning failed:', e)
      set({ directorLoading: false, directorError: msg, directorStep: 'style' })
    }
  },

  directorPlanVideoPrompts: async () => {
    const { directorPlannedClips, directorSceneDescription, directorAnalysis, directorClipPlans, directorReferenceImagePath } = get()
    if (!directorPlannedClips.length || !directorClipPlans.length) return
    set({ directorLoading: true, directorError: null, directorStep: 'plan_video' })
    try {
      // Build speaker_mappings
      const speakerMappings: Record<string, { name: string; role: string }> = {}
      for (const m of get().directorSpeakerMappings) {
        if (m.name.trim()) {
          speakerMappings[m.speakerId] = { name: m.name, role: m.role }
        }
      }

      // Phase 2: generate video prompts, passing existing image prompts as context
      const existingImagePrompts = directorClipPlans.map(p => p.image_prompt || '')
      const { directorCharacterRefPaths: crp, directorLocationRefPaths: lrp } = get()
      const result = await api.planClipPromptsAndImages({
        clips: directorPlannedClips,
        scene_description: directorSceneDescription,
        lyrics: directorAnalysis?.lyrics ?? undefined,
        bpm: directorAnalysis?.bpm ?? 120,
        reference_image_path: directorReferenceImagePath,
        ...(crp.length > 0 ? { character_ref_paths: crp } : {}),
        ...(lrp.length > 0 ? { location_ref_paths: lrp } : {}),
        speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
        prompt_type: 'video',
        existing_image_prompts: existingImagePrompts,
      })
      // Merge video prompts into existing clip plans
      const updatedPlans = directorClipPlans.map((plan, i) => ({
        ...plan,
        video_prompt: result.clip_plans[i]?.video_prompt || '',
      }))
      set({
        directorClipPlans: updatedPlans,
        directorStep: 'review_video',
        directorLoading: false,
      })

      // Auto-mode: skip review, apply to editor and start generation
      if (get().directorAutoMode) {
        get().directorGenerate()
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Video prompt planning failed'
      console.error('Director video planning failed:', e)
      set({ directorLoading: false, directorError: msg, directorStep: 'generate_images' })
    }
  },

  directorEditClipPlan: (index, field, value) => {
    set(s => {
      const plans = [...s.directorClipPlans]
      if (plans[index]) {
        plans[index] = { ...plans[index], [field]: value }
      }
      return { directorClipPlans: plans }
    })
  },

  directorGenerateStartImages: async () => {
    const { directorClipPlans, directorPlannedClips, params, selectedModelPerMode, savedParamsPerMode, savedLoraPerMode, directorResolution, directorAspectRatio, directorSceneDescription } = get()
    if (!directorClipPlans.length) return

    // Use saved image-mode settings if available, otherwise fall back to defaults
    const imageModel = selectedModelPerMode.image || 'flux2_klein_9b'
    const imageOptions = await api.fetchModelOptions(imageModel).catch(() => null)
    const directorRes = resolveResolution(
      imageOptions,
      directorResolution,
      directorAspectRatio,
    )
    // Director's hardcoded image_model fallback is flux2_klein_9b, which is
    // step-distilled to 4 inference steps (per app/defaults/flux2_klein_9b.json).
    const imageParams = savedParamsPerMode.image || { num_inference_steps: 4, guidance_scale: 1, resolution: directorRes }
    imageParams.resolution = directorRes
    const imageLora = savedLoraPerMode.image

    const buildImgPostProc = (): Record<string, unknown> => {
      const pp: Record<string, unknown> = {}
      const imgSpatial = get().directorImageSpatialUpsampling
      if (imgSpatial) pp.spatial_upsampling = imgSpatial
      const imgGrainIntensity = get().directorImageFilmGrainIntensity
      if (imgGrainIntensity > 0) {
        pp.film_grain_intensity = imgGrainIntensity
        pp.film_grain_saturation = get().directorImageFilmGrainSaturation
      }
      return pp
    }

    // Submit one image generation, poll to completion, download the result as a File.
    const genImage = async (prompt: string, refs: string[], label: string): Promise<{ file: File; filename: string }> => {
      const genParams = {
        model_type: imageModel,
        prompt,
        image_refs: refs,
        image_mode: 1,
        num_inference_steps: imageParams.num_inference_steps,
        guidance_scale: imageParams.guidance_scale,
        // 'KI' carries an image reference; plain T2I (the anchor) needs no ref flag.
        video_prompt_type: refs.length ? 'KI' : '',
        resolution: imageParams.resolution,
        seed: -1,
        settings_version: 2.52,
        generation_mode: 'image',
        repeat_generation: 1,
        negative_prompt: '',
        video_length: 1,
        activated_loras: imageLora?.activated_loras || params.activated_loras || [],
        loras_multipliers: imageLora?.loras_multipliers || params.loras_multipliers || '',
        ...buildImgPostProc(),
      }
      const { job_id } = await api.submitGeneration(genParams)
      let outputFiles: string[] = []
      let attempts = 0
      const maxAttempts = 300  // 300 × 2s = 10 minutes
      while (attempts < maxAttempts) {
        await new Promise(r => setTimeout(r, 2000))
        const status = await api.fetchJobStatus(job_id)
        if (status.status === 'completed') { outputFiles = status.output_files; break }
        if (status.status === 'failed') throw new Error(status.error || `${label} generation failed`)
        attempts++
      }
      if (attempts >= maxAttempts) throw new Error(`${label} generation timed out`)
      if (outputFiles.length === 0) throw new Error(`No output file for ${label}`)
      const filename = outputFiles[0]
      const imgRes = await fetch(api.getFileUrl(filename))
      const blob = await imgRes.blob()
      const file = new File([blob], filename, { type: blob.type || 'image/png' })
      return { file, filename }
    }

    // Auto-unload LLM before GPU-heavy image generation to free VRAM
    if (get().llmStatus?.loaded) {
      try {
        await api.unloadLlm()
        set({ llmStatus: { loaded: false, model_id: null, device: null, provider: '' } })
      } catch { /* best-effort */ }
    }

    set({ directorStep: 'generate_images', directorLoading: true, directorError: null, directorClipImages: [], directorImageGenProgress: null })

    try {
      // If no reference image was provided, generate a single establishing /
      // "anchor" image from the scene description and adopt it as the reference,
      // so every clip's start image shares a consistent look.
      let anchorMade = false
      if (!get().directorReferenceImage && !get().directorReferenceImagePath) {
        anchorMade = true
        set({
          directorImageGenProgress: {
            current: 0,
            total: directorClipPlans.length + 1,
            currentClipLabel: 'Establishing image…',
            status: 'generating',
          },
        })
        const anchorPrompt = directorSceneDescription.trim() || directorClipPlans[0]?.image_prompt || 'cinematic establishing shot'
        const { file: anchorFile } = await genImage(anchorPrompt, [], 'Establishing image')
        // Adopt as the reference image (uploaded just below via _uploadDirectorRefs).
        set({ directorReferenceImage: anchorFile, directorReferenceImagePath: null })
      }

      // Upload all reference images (main/anchor + character + location)
      const { refImagePath: refPath, charPaths, locPaths } = await get()._uploadDirectorRefs()
      const allRefs = [refPath, ...charPaths, ...locPaths].filter(Boolean) as string[]

      const total = directorClipPlans.length + (anchorMade ? 1 : 0)
      const base = anchorMade ? 1 : 0
      const generatedImages: DirectorClipImage[] = []

      // Generate one start image per clip sequentially.
      for (let i = 0; i < directorClipPlans.length; i++) {
        const clip = directorPlannedClips[i]
        const plan = directorClipPlans[i]
        const clipLabel = `Clip ${i + 1} (${clip?.section_label || 'verse'})`
        set({
          directorImageGenProgress: { current: base + i, total, currentClipLabel: clipLabel, status: 'generating' },
        })
        const { file, filename } = await genImage(plan.image_prompt, allRefs, clipLabel)
        generatedImages.push({ clipIndex: i, prompt: plan.image_prompt, file, filename })
        set({ directorClipImages: [...generatedImages] })
      }

      set({
        directorImageGenProgress: { current: total, total, currentClipLabel: '', status: 'done' },
        directorLoading: false,
      })

      // Video prompts already generated in the combined LLM pass — go straight to review
      const hasVideoPrompts = get().directorClipPlans.some(p => p.video_prompt)
      if (hasVideoPrompts) {
        set({ directorStep: 'review_video' })
        if (get().directorAutoMode) {
          get().directorGenerate()
        }
      } else {
        // Fallback: if video prompts are missing, plan them separately
        get().directorPlanVideoPrompts()
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Image generation failed'
      console.error('Director image generation failed:', e)
      set({
        directorLoading: false,
        directorError: msg,
        directorImageGenProgress: get().directorImageGenProgress
          ? { ...get().directorImageGenProgress!, status: 'error' }
          : null,
      })
    }
  },

  directorApplyToClips: () => {
    const { directorClipPlans, directorPlannedClips, directorAnalysis, directorClipImages,
            directorAudioPath, directorAudioFile, directorSeamless,
            selectedModelPerMode, savedParamsPerMode, savedLoraPerMode } = get()
    if (!directorClipPlans.length) return

    // Use saved video-mode settings if available
    const videoModel = selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const videoParams = savedParamsPerMode.video
      ? { ...savedParamsPerMode.video }
      : {}
    const directorSteps = get().directorVideoInferenceStepsByModel[videoModel]
    if (directorSteps != null) videoParams.num_inference_steps = directorSteps
    const videoLora = savedLoraPerMode.video

    const directorVideoOptions = get().modelOptions?.model_type === videoModel
      ? get().modelOptions
      : null
    const isH3Video = videoModel.startsWith('minimax_h3')
    const fps = isH3Video ? (directorVideoOptions?.fps ?? 24) : (directorVideoOptions?.fps ?? 16)
    const totalDuration = directorAnalysis?.duration ?? 180
    const totalDurationCapped = Math.min(totalDuration, 300)

    // Build clips with per-clip durations and images
    const clips: MultiClip[] = directorClipPlans.map((plan, i) => {
      const plannedClip = directorPlannedClips[i]
      const clipImage = directorClipImages.find(img => img.clipIndex === i)

      // Seamless mode: use next clip's start image as this clip's end image
      let endImage: File | null = null
      if (directorSeamless && i < directorClipPlans.length - 1) {
        const nextClipImage = directorClipImages.find(img => img.clipIndex === i + 1)
        endImage = nextClipImage?.file ?? null
      }

      return {
        prompt: plan.video_prompt,
        startImage: clipImage?.file ?? null,
        startImagePath: null,
        endImage,
        endImagePath: null,
        durationFrames: plannedClip?.duration_frames,
      }
    })

    // Build per-clip frame counts for variable-duration support
    const requestedClipFrames = clips.map(
      c => c.durationFrames ?? Math.round(5 * fps),
    )
    const perClipFrames = isH3Video
      ? normalizeH3ClipFrameSchedule(
          requestedClipFrames,
          directorVideoOptions?.frames_minimum ?? 124,
          directorVideoOptions?.frames_maximum ?? 345,
          directorVideoOptions?.frames_steps ?? 17,
        )
      : requestedClipFrames
    const totalFrames = perClipFrames.reduce((sum, f) => sum + f, 0)
    const maxClipFrames = Math.max(...perClipFrames)

    // Auto-set soundtrack mode with the already-uploaded audio
    const audioParams: Record<string, unknown> = {}
    if (directorAudioPath) {
      audioParams.audio_prompt_type = 'A'
      audioParams.audio_guide = directorAudioPath
    }

    set(s => ({
      params: {
        ...s.params,
        ...(videoModel ? { model_type: videoModel } : {}),
        ...(videoParams || {}),
        ...(videoLora ? { activated_loras: videoLora.activated_loras, loras_multipliers: (videoLora.loras_multipliers || '').split(' ').map(m => m.split(';')[0]).join(' ') } : {}),
        image_mode: 2,
        video_length: totalFrames,
        sliding_window_size: maxClipFrames,
        per_clip_frames: perClipFrames,
        ...audioParams,
      },
      clips,
      singlePromptMode: false,
      durationSeconds: totalDurationCapped,
      slidingWindowSeconds: maxClipFrames / fps,
      audioGuideFilename: directorAudioFile?.name ?? null,
      sidebarMode: 'studio' as const,
    }))
  },

  directorGenerate: () => {
    const { directorClipPlans, directorPlannedClips, directorAnalysis,
            directorClipImages, directorAudioPath, directorAudioFile,
            directorSeamless, directorResolution, directorAspectRatio,
            selectedModelPerMode, savedParamsPerMode, savedLoraPerMode } = get()
    if (!directorClipPlans.length) return

    // Use saved video-mode settings if available, override resolution with director's choice
    const videoModel = selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'
    const cachedDirectorVideoOptions = get().modelOptions?.model_type === videoModel
      ? get().modelOptions
      : null
    const directorRes = resolveResolution(
      cachedDirectorVideoOptions,
      directorResolution,
      directorAspectRatio,
    )
    const videoParams = savedParamsPerMode.video ? { ...savedParamsPerMode.video, resolution: directorRes } : { num_inference_steps: 8, guidance_scale: 1, resolution: directorRes }
    const directorSteps = get().directorVideoInferenceStepsByModel[videoModel]
    if (directorSteps != null) videoParams.num_inference_steps = directorSteps
    const videoLora = savedLoraPerMode.video

    const directorVideoOptions = get().modelOptions?.model_type === videoModel
      ? get().modelOptions
      : null
    const isH3Video = videoModel.startsWith('minimax_h3')
    const fps = isH3Video ? (directorVideoOptions?.fps ?? 24) : (directorVideoOptions?.fps ?? 16)
    const totalDuration = directorAnalysis?.duration ?? 180
    const totalDurationCapped = Math.min(totalDuration, 300)

    const clips: MultiClip[] = directorClipPlans.map((plan, i) => {
      const plannedClip = directorPlannedClips[i]
      const clipImage = directorClipImages.find(img => img.clipIndex === i)

      // Seamless mode: use next clip's start image as this clip's end image
      let endImage: File | null = null
      if (directorSeamless && i < directorClipPlans.length - 1) {
        const nextClipImage = directorClipImages.find(img => img.clipIndex === i + 1)
        endImage = nextClipImage?.file ?? null
      }

      return {
        prompt: plan.video_prompt,
        startImage: clipImage?.file ?? null,
        startImagePath: null,
        endImage,
        endImagePath: null,
        durationFrames: plannedClip?.duration_frames,
      }
    })

    const requestedClipFrames = clips.map(
      c => c.durationFrames ?? Math.round(5 * fps),
    )
    const perClipFrames = isH3Video
      ? normalizeH3ClipFrameSchedule(
          requestedClipFrames,
          directorVideoOptions?.frames_minimum ?? 124,
          directorVideoOptions?.frames_maximum ?? 345,
          directorVideoOptions?.frames_steps ?? 17,
        )
      : requestedClipFrames
    const totalFrames = perClipFrames.reduce((sum, f) => sum + f, 0)
    const maxClipFrames = Math.max(...perClipFrames)

    const audioParams: Record<string, unknown> = {}
    if (get().shortFilmPath === 'story') {
      // Path C: LTX generates video + audio from text (dialogue in quotes)
      audioParams.audio_prompt_type = ''
    } else if (directorAudioPath) {
      audioParams.audio_prompt_type = 'A'
      audioParams.audio_guide = directorAudioPath
    }

    // Apply director video post-processing to shared state (read by startGeneration)
    const vidSelfRefiner = get().directorVideoSelfRefiner

    set(s => ({
      params: {
        ...s.params,
        ...(videoModel ? { model_type: videoModel } : {}),
        ...(videoParams || {}),
        ...(videoLora ? { activated_loras: videoLora.activated_loras, loras_multipliers: (videoLora.loras_multipliers || '').split(' ').map(m => m.split(';')[0]).join(' ') } : {}),
        image_mode: 2,
        video_length: totalFrames,
        sliding_window_size: maxClipFrames,
        per_clip_frames: perClipFrames,
        self_refiner_setting: vidSelfRefiner,
        ...audioParams,
      },
      clips,
      singlePromptMode: false,
      durationSeconds: totalDurationCapped,
      slidingWindowSeconds: maxClipFrames / fps,
      audioGuideFilename: directorAudioFile?.name ?? null,
      // Apply director video post-processing to shared state
      spatialUpsampling: get().directorVideoSpatialUpsampling,
      filmGrainIntensity: get().directorVideoFilmGrainIntensity,
      filmGrainSaturation: get().directorVideoFilmGrainSaturation,
    }))

    setTimeout(() => get().startGeneration(), 200)
  },

  directorReset: () => {
    set({
      sidebarMode: 'studio' as const,
      directorStep: 'upload',
      directorAudioFile: null,
      directorAudioPath: null,
      directorAnalysis: null,
      directorPlannedClips: [],
      directorEnergyBias: 0,
      directorClipPlans: [],
      directorSceneDescription: '',
      directorLoading: false,
      directorError: null,
      directorReferenceImage: null,
      directorReferenceImagePath: null,
      directorCharacterRefs: [],
      directorCharacterRefPaths: [],
      directorCharacterRefLabels: [],
      directorLocationRefs: [],
      directorLocationRefPaths: [],
      directorLocationRefLabels: [],
      directorVoiceRef: null,
      directorVoiceRefPath: null,
      directorClipImages: [],
      directorImageGenProgress: null,
      directorSpeakers: [],
      directorSpeakerMappings: [],
      directorAutoMode: true,
      directorSeamless: false,
      directorShotImageGuidance: 'auto' as DirectorShotImageGuidance,
      directorLlmLog: [],
      directorSkill: null,
      directorMusicSource: null,
      directorSongDescription: '',
      directorSongInstrumental: false,
      directorSongStyle: '',
      directorSongLyrics: '',
      directorSongDuration: 120,
      directorTrackGenerating: false,
      shortFilmCharacters: [],
      shortFilmPath: null,
      shortFilmTargetDuration: 30,
      shortFilmNarrative: false,
    })
  },

  // --- Short Film Director actions ---

  shortFilmSetCharacters: (characters) => set({ shortFilmCharacters: characters }),
  shortFilmSetPath: (path) => set({ shortFilmPath: path }),
  shortFilmSetTargetDuration: (duration) => set({ shortFilmTargetDuration: duration }),
  shortFilmSetNarrative: (v) => set({ shortFilmNarrative: v }),

  shortFilmUploadAndAnalyze: async (file) => {
    set({
      directorLoading: true,
      directorLoadingMessage: 'Uploading audio...',
      directorError: null,
      directorAudioFile: file,
      directorStep: 'analyze',
    })
    // Same polling pattern as directorUploadAndAnalyze — see comment
    // there for the full rationale on /api/v1/audio/analyze/status.
    let analyzePoll: ReturnType<typeof setInterval> | null = null
    const startAnalyzePolling = () => {
      analyzePoll = setInterval(async () => {
        try {
          const status = await api.fetchAudioAnalyzeStatus()
          if (!status.step) return
          set({ directorLoadingMessage: `${status.detail}...` })
        } catch { /* polling errors are non-fatal */ }
      }, 1000)
    }
    const stopAnalyzePolling = () => {
      if (analyzePoll !== null) {
        clearInterval(analyzePoll)
        analyzePoll = null
      }
    }
    try {
      const uploaded = await api.uploadAudio(file)
      set({ directorAudioPath: uploaded.path, directorLoadingMessage: 'Analyzing audio...' })

      startAnalyzePolling()
      const analysis = await api.analyzeAudio({
        audio_path: uploaded.path,
        transcribe: true,
        extract_vocals: true,
      })
      stopAnalyzePolling()

      set({ directorAnalysis: analysis })

      // Extract unique speakers from diarized lyrics
      const speakers: string[] = []
      if (analysis.lyrics) {
        const seen = new Set<string>()
        for (const seg of analysis.lyrics) {
          if (seg.speaker && !seen.has(seg.speaker)) {
            seen.add(seg.speaker)
            speakers.push(seg.speaker)
          }
        }
      }
      const speakerMappings: SpeakerMapping[] = speakers.map(s => ({
        speakerId: s,
        name: '',
        role: 'speaking' as const,
      }))
      set({ directorSpeakers: speakers, directorSpeakerMappings: speakerMappings })

      // Plan dialogue-paced clip structure (not beat-aligned)
      set({ directorLoadingMessage: 'Planning scenes...' })
      const structure = await api.planDialogueScenes({
        analysis,
        pacing_bias: get().directorEnergyBias,
        fps: get().modelOptions?.fps ?? 16,
        frames_steps: get().modelOptions?.frames_steps ?? 4,
        frames_minimum: get().modelOptions?.frames_minimum ?? 5,
      })
      set({
        directorPlannedClips: structure.clips,
        directorStep: 'structure',
        directorLoading: false,
        directorLoadingMessage: null,
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Analysis failed'
      console.error('Short film analysis failed:', e)
      set({ directorLoading: false, directorLoadingMessage: null, directorError: msg, directorStep: 'upload' })
    } finally {
      stopAnalyzePolling()
    }
  },

  shortFilmSetPacingBias: async (bias) => {
    const { directorAnalysis } = get()
    if (!directorAnalysis) return
    set({ directorLoading: true, directorEnergyBias: bias })
    try {
      const structure = await api.planDialogueScenes({
        analysis: directorAnalysis,
        pacing_bias: bias,
        fps: get().modelOptions?.fps ?? 16,
        frames_steps: get().modelOptions?.frames_steps ?? 4,
        frames_minimum: get().modelOptions?.frames_minimum ?? 5,
      })
      set({ directorPlannedClips: structure.clips, directorLoading: false })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to update structure'
      set({ directorLoading: false, directorError: msg })
    }
  },

  shortFilmPlanPrompts: async () => {
    const { directorPlannedClips, directorSceneDescription, directorAnalysis,
            shortFilmCharacters } = get()
    if (!directorPlannedClips.length || !directorSceneDescription.trim()) return
    set({ directorLoading: true, directorError: null, directorStep: 'plan' })
    try {
      // Upload all reference images
      const { refImagePath, charPaths, locPaths } = await get()._uploadDirectorRefs()
      const { directorCharacterRefLabels: charLabels, directorLocationRefLabels: locLabels } = get()
      const extraRefs = {
        ...(charPaths.length > 0 ? { character_ref_paths: charPaths, character_ref_labels: charLabels } : {}),
        ...(locPaths.length > 0 ? { location_ref_paths: locPaths, location_ref_labels: locLabels } : {}),
      }
      const generateShotImages = _directorUsesGeneratedShotImages(get())
      const promptType = generateShotImages ? 'both' : 'video'

      // Build speaker mappings
      const speakerMappings: Record<string, { name: string; role: string }> = {}
      for (const m of get().directorSpeakerMappings) {
        if (m.name.trim()) {
          speakerMappings[m.speakerId] = { name: m.name, role: m.role }
        }
      }

      // Generate prompts
      // ?? not || — an explicit user-toggled `false` must be respected
      // (legacy v1 path); only fall back to true when servicesConfig
      // hasn't loaded yet or the field is undefined.
      const useV2 = get().servicesConfig?.use_director_v2 ?? true
      let plans: Array<{ video_prompt: string; image_prompt: string }>

      if (useV2) {
        const result = await api.directorV2Plan({
          skill_type: 'short_film',
          clips: directorPlannedClips,
          scene_description: directorSceneDescription,
          lyrics: directorAnalysis?.lyrics ?? undefined,
          reference_image_path: refImagePath ?? undefined,
          ...extraRefs,
          speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
          characters: shortFilmCharacters.length > 0 ? shortFilmCharacters : undefined,
          prompt_type: promptType,
        })
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
      } else {
        const result = await api.planShortFilmPrompts({
          clips: directorPlannedClips,
          scene_description: directorSceneDescription,
          lyrics: directorAnalysis?.lyrics ?? undefined,
          reference_image_path: refImagePath,
          ...extraRefs,
          speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
          characters: shortFilmCharacters.length > 0 ? shortFilmCharacters : undefined,
          prompt_type: promptType,
        })
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
      }
      set({
        directorClipPlans: plans,
        directorStep: generateShotImages ? 'review' : 'review_video',
        directorLoading: false,
      })

      // Auto-mode: skip review
      if (get().directorAutoMode) {
        if (generateShotImages) {
          get().directorGenerateStartImages()
        } else {
          get().directorGenerate()
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Planning failed'
      console.error('Short film planning failed:', e)
      set({ directorLoading: false, directorError: msg, directorStep: 'style' })
    }
  },

  shortFilmPlanVideoPrompts: async () => {
    const { directorPlannedClips, directorSceneDescription, directorAnalysis,
            directorClipPlans, directorReferenceImagePath, shortFilmCharacters } = get()
    if (!directorPlannedClips.length || !directorClipPlans.length) return
    set({ directorLoading: true, directorError: null, directorStep: 'plan_video' })
    try {
      const speakerMappings: Record<string, { name: string; role: string }> = {}
      for (const m of get().directorSpeakerMappings) {
        if (m.name.trim()) {
          speakerMappings[m.speakerId] = { name: m.name, role: m.role }
        }
      }

      const existingImagePrompts = directorClipPlans.map(p => p.image_prompt || '')
      const { directorCharacterRefPaths: crp2, directorLocationRefPaths: lrp2 } = get()
      const result = await api.planShortFilmPrompts({
        clips: directorPlannedClips,
        scene_description: directorSceneDescription,
        lyrics: directorAnalysis?.lyrics ?? undefined,
        reference_image_path: directorReferenceImagePath,
        ...(crp2.length > 0 ? { character_ref_paths: crp2 } : {}),
        ...(lrp2.length > 0 ? { location_ref_paths: lrp2 } : {}),
        speaker_mappings: Object.keys(speakerMappings).length > 0 ? speakerMappings : undefined,
        characters: shortFilmCharacters.length > 0 ? shortFilmCharacters : undefined,
        prompt_type: 'video',
        existing_image_prompts: existingImagePrompts,
      })
      const updatedPlans = directorClipPlans.map((plan, i) => ({
        ...plan,
        video_prompt: result.clip_plans[i]?.video_prompt || '',
      }))
      set({
        directorClipPlans: updatedPlans,
        directorStep: 'review_video',
        directorLoading: false,
      })

      if (get().directorAutoMode) {
        get().directorGenerate()
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Video prompt planning failed'
      console.error('Short film video planning failed:', e)
      set({ directorLoading: false, directorError: msg, directorStep: 'generate_images' })
    }
  },

  shortFilmPlanFromStory: async () => {
    const { directorSceneDescription,
            shortFilmCharacters, shortFilmTargetDuration, shortFilmNarrative } = get()
    if (!directorSceneDescription.trim()) return
    set({ directorLoading: true, directorError: null, directorStep: 'plan', llmStreamText: '', llmStreamDone: false })
    try {
      // Upload all reference images
      const { refImagePath, charPaths, locPaths } = await get()._uploadDirectorRefs()
      const { directorCharacterRefLabels: charLabels, directorLocationRefLabels: locLabels } = get()
      const extraRefs = {
        ...(charPaths.length > 0 ? { character_ref_paths: charPaths, character_ref_labels: charLabels } : {}),
        ...(locPaths.length > 0 ? { location_ref_paths: locPaths, location_ref_labels: locLabels } : {}),
      }
      const generateShotImages = _directorUsesGeneratedShotImages(get())
      const promptType = generateShotImages ? 'both' : 'video'

      // ?? not || — an explicit user-toggled `false` must be respected
      // (legacy v1 path); only fall back to true when servicesConfig
      // hasn't loaded yet or the field is undefined.
      const useV2 = get().servicesConfig?.use_director_v2 ?? true
      let plans: Array<{ video_prompt: string; image_prompt: string }>
      let storyClips: any[] | undefined

      if (useV2) {
        const result = await api.directorV2Plan({
          skill_type: 'short_film',
          scene_description: directorSceneDescription,
          story_description: directorSceneDescription,
          characters: shortFilmCharacters.length > 0 ? shortFilmCharacters : undefined,
          reference_image_path: refImagePath ?? undefined,
          ...extraRefs,
          target_duration: shortFilmTargetDuration,
          narrative_mode: shortFilmNarrative,
          fps: get().modelOptions?.fps ?? 24,
          frames_steps: get().modelOptions?.frames_steps ?? 4,
          frames_minimum: get().modelOptions?.frames_minimum ?? 5,
          prompt_type: promptType,
        })
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
        // Extract clips from production plan shots
        const pp = result.production_plan as any
        if (pp?.shots) {
          let cumulative = 0
          storyClips = pp.shots.map((s: any) => {
            const clip = {
              start: cumulative,
              end: cumulative + (s.duration_sec || 15),
              duration_frames: s.metadata?.duration_frames || Math.round((s.duration_sec || 15) * (get().modelOptions?.fps ?? 24)),
              label: s.narrative_role || s.scene_type || 'scene',
              beat_count: 0,
            }
            cumulative += s.duration_sec || 15
            return clip
          })
        }
      } else {
        const result = await api.planShortFilmScript({
          story_description: directorSceneDescription,
          characters: shortFilmCharacters.length > 0 ? shortFilmCharacters : undefined,
          reference_image_path: refImagePath ?? undefined,
          ...extraRefs,
          target_duration: shortFilmTargetDuration,
          narrative_mode: shortFilmNarrative,
          fps: get().modelOptions?.fps ?? 24,
          frames_steps: get().modelOptions?.frames_steps ?? 4,
          frames_minimum: get().modelOptions?.frames_minimum ?? 5,
        })
        storyClips = result.clips
        plans = result.clip_plans.map(p => ({
          video_prompt: p.video_prompt || '',
          image_prompt: p.image_prompt || '',
        }))
      }

      set({ llmStreamDone: true })

      set({
        directorPlannedClips: storyClips || get().directorPlannedClips,
        directorClipPlans: plans,
        directorStep: generateShotImages ? 'review' : 'review_video',
        directorLoading: false,
      })

      // Auto-mode: skip review steps
      if (get().directorAutoMode) {
        if (generateShotImages) {
          get().directorGenerateStartImages()
        } else {
          get().directorGenerate()
        }
      }
    } catch (e: unknown) {
      set({ llmStreamDone: true })
      const msg = e instanceof Error ? e.message : 'Story planning failed'
      console.error('Short film story planning failed:', e)
      set({ directorLoading: false, directorError: msg, directorStep: 'style' })
    }
  },

  selectModel: (modelType) => {
    const currentMode = get().generationMode
    set(s => ({
      params: {
        ...s.params,
        model_type: modelType,
        activated_loras: [],
        loras_multipliers: '',
        minimax_h3_turbo_mode: false,
        minimax_h3_turbo_preset: undefined,
      },
      selectedModelPerMode: { ...s.selectedModelPerMode, [currentMode]: modelType },
      h3WindowPlan: null,
      loraWeights: {},
      availableLoras: [],
    }))
    // Virtual SFX models don't have backend model options or LoRAs
    if (!sfxModelTypes.has(modelType)) {
      get().loadLoras(modelType)
      get().loadModelOptions(modelType)
      _applyModelDefaults(get, set, modelType)
    }
    // Persist to localStorage
    const s = get()
    _saveSettings({
      generationMode: s.generationMode,
      selectedModelPerMode: s.selectedModelPerMode,
      savedParamsPerMode: s.savedParamsPerMode,
      savedLoraPerMode: s.savedLoraPerMode,
    }, s.loraIdByFilename)
  },

  // Workspaces
  workspaces: [],
  activeWorkspace: 'default',
  browsingUploads: false,
  loadWorkspaces: async () => {
    try {
      const data = await api.fetchWorkspaces()
      set({ workspaces: data.workspaces, activeWorkspace: data.active })
    } catch (e) {
      console.error('Failed to load workspaces:', e)
    }
  },
  switchWorkspace: async (name) => {
    // Virtual "Uploads" view: browse the uploads folder WITHOUT touching
    // the server-side active workspace — generations keep saving to the
    // real workspace; uploads are read-only in the gallery.
    if (name === '__uploads__') {
      set({ browsingUploads: true, outputs: [], outputsTotal: 0, selectedOutput: 0, selectedOutputMeta: null })
      get().loadOutputs()
      return
    }
    try {
      await api.setActiveWorkspace(name)
      set({ browsingUploads: false, activeWorkspace: name, outputs: [], outputsTotal: 0, selectedOutput: 0, selectedOutputMeta: null })
      get().loadOutputs()
      get().loadWorkspaces()
    } catch (e) {
      console.error('Failed to switch workspace:', e)
    }
  },
  createWorkspace: async (name) => {
    try {
      await api.createWorkspace(name)
      await api.setActiveWorkspace(name)
      set({ browsingUploads: false, activeWorkspace: name, outputs: [], outputsTotal: 0, selectedOutput: 0, selectedOutputMeta: null })
      get().loadOutputs()
      get().loadWorkspaces()
    } catch (e) {
      console.error('Failed to create workspace:', e)
      throw e
    }
  },
  deleteWorkspace: async (name) => {
    // The server refuses 'default', refuses while anything generates, and
    // auto-switches to default when the deleted workspace was active —
    // its switched_to_default answer is authoritative (a client-side
    // activeWorkspace comparison could disagree after a desync and would
    // widen it by force-resetting state the server never changed).
    const result = await api.deleteWorkspace(name)
    if (result.switched_to_default) {
      set({ browsingUploads: false, activeWorkspace: 'default', outputs: [], outputsTotal: 0, selectedOutput: 0, selectedOutputMeta: null })
      get().loadOutputs()
    }
    get().loadWorkspaces()
  },

  storageDashboardOpen: false,
  setStorageDashboardOpen: (open) => set({ storageDashboardOpen: open }),

  loraPickerSort: (() => {
    try { return localStorage.getItem('maestro_lora_picker_sort') === 'newest' ? 'newest' as const : 'name' as const } catch { return 'name' as const }
  })(),
  setLoraPickerSort: (sort) => {
    try { localStorage.setItem('maestro_lora_picker_sort', sort) } catch { /* private mode */ }
    set({ loraPickerSort: sort })
  },

  outputs: [],
  outputsTotal: 0,
  selectedOutput: 0,
  setSelectedOutput: (i) => {
    set({ selectedOutput: i })
    const outputs = get().filteredOutputs()
    const output = outputs[i]
    if (output) {
      get().loadOutputMetadata(output.name)
    } else {
      set({ selectedOutputMeta: null })
    }
  },
  mediaFilter: 'all',
  outputSearchQuery: '',
  setMediaFilter: (f) => {
    const prevFilter = get().mediaFilter
    set({ mediaFilter: f, selectedOutput: 0 })
    // Backend-filtered modes: reload from server to get ALL matches
    const backendFilters: MediaFilter[] = ['favorites', 'multiclip']
    if (backendFilters.includes(f) || backendFilters.includes(prevFilter)) {
      get().loadOutputs()
      return
    }
    // Load metadata for first item in new filtered list
    const filtered = get().filteredOutputs()
    if (filtered.length > 0) {
      get().loadOutputMetadata(filtered[0].name)
    } else {
      set({ selectedOutputMeta: null })
    }
  },
  setOutputSearchQuery: (q) => {
    set({ outputSearchQuery: q, selectedOutput: 0 })
    if (q.trim()) {
      get().loadOutputs()
    } else if (get().mediaFilter === 'all') {
      // Clear search: reload normal paginated view
      get().loadOutputs()
    }
  },
  filteredOutputs: () => {
    const { outputs, mediaFilter } = get()
    return computeFilteredOutputs(outputs, mediaFilter)
  },

  outputsLoading: false,
  loadOutputs: async () => {
    const PAGE_SIZE = 100
    const { mediaFilter, outputSearchQuery, browsingUploads } = get()
    const isBackendFilter = mediaFilter === 'favorites' || mediaFilter === 'multiclip' || outputSearchQuery.trim()
    const ws = browsingUploads ? '__uploads__' : undefined
    set({ outputsLoading: true })
    try {
      const { outputs: apiOutputs, total } = isBackendFilter
        ? await api.fetchOutputs(0, 0, {
            favoritesOnly: mediaFilter === 'favorites',
            multiclipOnly: mediaFilter === 'multiclip',
            search: outputSearchQuery.trim() || undefined,
            workspace: ws,
          })
        : await api.fetchOutputs(PAGE_SIZE, 0, { workspace: ws })
      const outputs: OutputFile[] = apiOutputs.map(o => ({
        name: o.name,
        url: o.url,
        type: o.type,
        mode: (o.mode as OutputFile['mode']) || null,
        edit_sub_mode: (o.edit_sub_mode as OutputFile['edit_sub_mode']) || null,
        favorite: o.favorite || false,
        size: o.size,
        created_at: o.created_at,
      }))
      set({ outputs, outputsTotal: total, selectedOutput: 0, outputsLoading: false })
      if (outputs.length > 0) {
        get().loadOutputMetadata(outputs[0].name)
      }
    } catch (e) {
      console.error('Failed to load outputs:', e)
      set({ outputsLoading: false })
    }
  },

  // Load next page of outputs (infinite scroll)
  loadMoreOutputs: async () => {
    const PAGE_SIZE = 100
    const current = get().outputs
    const total = get().outputsTotal
    if (current.length >= total) return // All loaded
    try {
      const { outputs: apiOutputs, total: newTotal } = await api.fetchOutputs(PAGE_SIZE, current.length, {
        workspace: get().browsingUploads ? '__uploads__' : undefined,
      })
      const more: OutputFile[] = apiOutputs.map(o => ({
        name: o.name,
        url: o.url,
        type: o.type,
        mode: (o.mode as OutputFile['mode']) || null,
        edit_sub_mode: (o.edit_sub_mode as OutputFile['edit_sub_mode']) || null,
        favorite: o.favorite || false,
        size: o.size,
        created_at: o.created_at,
      }))
      // Deduplicate (in case items shifted during generation)
      const existingNames = new Set(current.map(o => o.name))
      const unique = more.filter(o => !existingNames.has(o.name))
      if (unique.length > 0) {
        set({ outputs: [...current, ...unique], outputsTotal: newTotal })
      }
    } catch {
      // Silent fail
    }
  },

  // Incremental refresh: only fetch the newest items to detect new outputs during generation
  refreshOutputs: async () => {
    try {
      // Only fetch first page — new outputs appear at the top (newest first)
      const { outputs: apiOutputs, total } = await api.fetchOutputs(50, 0)
      const fresh: OutputFile[] = apiOutputs.map(o => ({
        name: o.name,
        url: o.url,
        type: o.type,
        mode: (o.mode as OutputFile['mode']) || null,
        edit_sub_mode: (o.edit_sub_mode as OutputFile['edit_sub_mode']) || null,
        favorite: o.favorite || false,
        size: o.size,
        created_at: o.created_at,
      }))
      const current = get().outputs
      const currentNames = new Set(current.map(o => o.name))
      const newItems = fresh.filter(o => !currentNames.has(o.name))
      if (newItems.length > 0) {
        // Prepend new items (newest first) and shift selectedOutput to keep the same item active
        const merged = [...newItems, ...current]
        const sel = get().selectedOutput
        set({ outputs: merged, outputsTotal: total, selectedOutput: sel + newItems.length })
      }
    } catch {
      // Silent fail for background refresh
    }
  },

  toggleFavorite: async (name) => {
    try {
      const result = await api.toggleFavorite(name)
      set(s => ({
        outputs: s.outputs.map(o => o.name === name ? { ...o, favorite: result.favorite } : o),
      }))
    } catch (e) {
      console.error('Failed to toggle favorite:', e)
    }
  },

  // Output metadata
  selectedOutputMeta: null,
  metadataLoading: false,

  loadOutputMetadata: async (name) => {
    set({ metadataLoading: true, selectedOutputMeta: null })
    try {
      const meta = await api.fetchOutputMetadata(name)
      set({ selectedOutputMeta: meta, metadataLoading: false })
    } catch (e) {
      // Diagnostic: surface metadata-fetch failures (the usual cause of a
      // "Load Settings does nothing" report on slow/VPN links) instead of
      // swallowing them silently.
      console.error('[LoadSettings] fetchOutputMetadata FAILED for', name, '-', e)
      set({ selectedOutputMeta: null, metadataLoading: false })
    }
  },

  loadSettingsFromOutput: async () => {
    // Metadata is normally fetched in the background when an output is selected.
    // On a slow/high-latency link (e.g. the user is remote over VPN) that fetch
    // may not have landed — or may have failed — by the time "Load Settings" is
    // clicked, leaving selectedOutputMeta null and this a silent no-op. Re-fetch
    // on demand so the click is self-healing regardless of the background state.
    let selectedOutputMeta = get().selectedOutputMeta
    console.log('[LoadSettings] clicked — meta present:', !!selectedOutputMeta?.params,
                '| metadataLoading:', get().metadataLoading, '| selectedOutput idx:', get().selectedOutput)
    if (!selectedOutputMeta?.params) {
      const pendingOutput = get().filteredOutputs()[get().selectedOutput]
      console.log('[LoadSettings] no meta yet — on-demand fetch for:', pendingOutput?.name ?? '(no output at index)')
      if (pendingOutput) {
        await get().loadOutputMetadata(pendingOutput.name)
        selectedOutputMeta = get().selectedOutputMeta
        console.log('[LoadSettings] after on-demand fetch — params present:', !!selectedOutputMeta?.params,
                    '| source:', selectedOutputMeta?.source)
      }
    }
    if (!selectedOutputMeta?.params) {
      console.warn('[LoadSettings] ABORT — no params available after fetch attempt; button is a no-op')
      return
    }
    const { models } = get()
    const p = selectedOutputMeta.params as Record<string, unknown>
    const uploadFilenames = selectedOutputMeta.upload_filenames as Record<string, string> | undefined
    console.log('[LoadSettings] applying settings — model_type:', p.model_type, '| param keys:', Object.keys(p).length)

    let modelType = (p.model_type as string) || ''
    if (!modelType) return

    // Migrate Recast sidecars made before the dedicated model existed. Those
    // jobs used the general I2V Fast accelerator with replacement conditioning;
    // loading them now should reproduce the corrected native-replacement recipe.
    const migratedLegacyRecast = p.edit_sub_mode === 'recast'
      && modelType === 'scail2_14B_fast'
      && models.some(m => m.model_type === 'scail2_14B_recast_fast')
    if (migratedLegacyRecast) modelType = 'scail2_14B_recast_fast'

    // SFX generations swap the virtual MMAudio model for a video carrier
    // at submit, so the sidecar records the carrier. Restore the virtual
    // id — resubmitting re-swaps it, and mode/sub-tab detection below
    // classifies it as audio/sfx instead of video.
    const sfxVirtual = p._sfx_virtual_model as string | undefined
    if ((p._audio_sub_mode === 'sfx' || p.sfx_mode) && sfxVirtual && models.some(m => m.model_type === sfxVirtual)) {
      modelType = sfxVirtual
    }

    // Per-sub-mode isolation: pencil-load may jump the sidebar to another
    // video sub-mode (or clobber the current one) by writing params
    // wholesale. Stash the active sub-mode's working set first so
    // in-progress work (e.g. a Frames setup) survives loading an Extend
    // clip's settings — switching back restores it.
    {
      const cur = get()
      if (cur.generationMode === 'video') {
        set({
          videoSubModeStash: {
            ...cur.videoSubModeStash,
            [(cur.params.image_mode as number) ?? 0]: captureVideoSubModeStash(cur),
          },
        })
      }
    }

    // Determine generation mode from model (respects per-model avatar overrides)
    const model = models.find(m => m.model_type === modelType)
    if (model) {
      const mode = getModelMode(modelType, model.family)
      set({ generationMode: mode })
      // Audio outputs restore the SUB-TAB too (Speech / Music / SFX) —
      // previously the pencil landed on the Audio tab but left whatever
      // sub-tab was last open. Newer sidecars record _audio_sub_mode;
      // older ones fall back to classifying the model. Direct set, NOT
      // setAudioSubMode — that would call selectModel and clobber the
      // params restored below.
      if (mode === 'audio') {
        const recordedSub = p._audio_sub_mode as import('../types').AudioSubMode | undefined
        const inferredSub: import('../types').AudioSubMode =
          sfxModelTypes.has(modelType) || p.sfx_mode ? 'sfx'
          : isMusicModelType(modelType) ? 'music'
          : 'speech'
        const subMode = (recordedSub === 'speech' || recordedSub === 'music' || recordedSub === 'sfx')
          ? recordedSub : inferredSub
        const restoredLyrics = (p._tts_original_prompt as string) || (p.prompt as string) || ''
        set(s => ({
          audioSubMode: subMode,
          selectedModelPerAudioSubMode: { ...s.selectedModelPerAudioSubMode, [subMode]: modelType },
          // Music: restore the song-writer inputs alongside the fields.
          // Older sidecars lack _music_description — clear rather than
          // leave a stale description that didn't produce this song
          // (instrumental still infers from the lyrics sentinel).
          ...(subMode === 'music' ? {
            musicDescription: (p._music_description as string) || '',
            musicInstrumental: !!p._music_instrumental
              || restoredLyrics.trim().toLowerCase() === '[instrumental]',
          } : {}),
        }))
      }
    }

    // Load model capabilities BEFORE applying the restored params.
    // loadModelOptions merges model-default steps/guidance into params when
    // its fetch resolves; it used to be fired at the END of this restore,
    // so the defaults landed after the sidecar values and silently reverted
    // num_inference_steps / guidance_scale on every pencil click. Awaiting
    // it here means defaults land first and the restored values win — and
    // modelOptions matches the restored model before rerollGeneration
    // submits (stale capabilities used to strip stg_scale/perturbation_*
    // from the request, which then poisoned the next sidecar with zeros).
    // (Virtual SFX models have no LoRAs/options endpoints — same guard
    // as boot.)
    if (!sfxModelTypes.has(modelType)) {
      get().loadLoras(modelType)
      await get().loadModelOptions(modelType)
    }
    const restoredModelOptions = get().modelOptions?.model_type === modelType
      ? get().modelOptions
      : null
    const restoredIsH3 = String(
      restoredModelOptions?.architecture || modelType,
    ).startsWith('minimax_h3')

    // Detect I2V: if image_start was used or image_prompt_type contains "S"
    const hadStartImage = !!(p.image_start || (p.image_prompt_type as string || '').includes('S'))
    const hadEndImage = !!(p.image_end || (p.image_prompt_type as string || '').includes('E'))

    const restoredH3WindowPlan = (
      p.h3_window_plan
      && typeof p.h3_window_plan === 'object'
      && Array.isArray((p.h3_window_plan as Record<string, unknown>).windows)
    ) ? p.h3_window_plan as unknown as H3WindowPlan : null
    const restoredH3WindowPrompts = Array.isArray(p.h3_window_prompts)
      ? p.h3_window_prompts
        .filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
        .map(item => item.trim())
      : []
    const savedRuntimePrompt = typeof p.prompt === 'string' ? p.prompt.trim() : ''
    const savedSourcePrompt = restoredH3WindowPlan?.source_prompt?.trim() || ''
    const serializedH3Prompt = restoredH3WindowPrompts.join('\n---CLIP_BOUNDARY---\n')
    const restoredH3SourcePrompt = savedSourcePrompt && (
      savedRuntimePrompt === savedSourcePrompt
      || savedRuntimePrompt === serializedH3Prompt
      || (restoredH3WindowPrompts.length === 1
        && savedRuntimePrompt === restoredH3WindowPrompts[0])
    ) ? savedSourcePrompt : ''

    // First / Last sidecars created before the explicit prompt-mode field
    // used minimax_h3_window_storyboard as the UI's Auto/Manual switch.
    // Prefer the explicit field, while keeping those existing clips durable.
    const restoredH3SequencePromptMode: 'auto' | 'manual' | undefined = (
      p.minimax_h3_sequence_prompt_mode === 'manual'
        ? 'manual'
        : p.minimax_h3_sequence_prompt_mode === 'auto'
          ? 'auto'
          : p.minimax_h3_multi_window === true
            ? (p.minimax_h3_window_storyboard === false ? 'manual' : 'auto')
            : undefined
    )

    const restoredTurboOption = restoredModelOptions?.minimax_h3_turbo
    const restoredTurboPresets = restoredTurboOption?.presets?.length
      ? restoredTurboOption.presets
      : restoredTurboOption
        ? [{
            id: restoredTurboOption.preset_id,
            filename: restoredTurboOption.filename,
          }]
        : []
    const savedTurboPreset = restoredTurboPresets.find(
      preset => preset.id === p.minimax_h3_turbo_preset,
    )
    const savedActivatedLoras = Array.isArray(p.activated_loras)
      ? (p.activated_loras as unknown[]).map(item => String(item))
      : []
    const activeTurboPreset = restoredTurboPresets.find(
      preset => savedActivatedLoras.some(
        filename => filename.replace(/\\/g, '/').split('/').pop()?.toLowerCase()
          === preset.filename.toLowerCase(),
      ),
    )
    const restoredTurboPreset = (
      savedTurboPreset
      || activeTurboPreset
      || restoredTurboPresets.find(
        preset => preset.id === restoredTurboOption?.preset_id,
      )
      || restoredTurboPresets[0]
    )
    const legacyTurboEnabled = (
      p.minimax_h3_turbo_mode == null
      && activeTurboPreset != null
    )
    const savedTextEncoder = p.minimax_h3_text_encoder
    const restoredTextEncoder = (
      savedTextEncoder === 'nvfp4_awq'
      || savedTextEncoder === 'gguf_q2_k'
      || savedTextEncoder === 'gguf_q4_k_m'
      || savedTextEncoder === 'int8'
      || savedTextEncoder === 'bf16'
    ) && restoredModelOptions?.minimax_h3_text_encoder_choices?.some(
      choice => choice.value === savedTextEncoder,
    ) ? savedTextEncoder : undefined
    const restoredLtx25VideoVae = restoredModelOptions?.ltx25_video_vae_choices
      ?.find(choice => choice.value === p.ltx25_video_vae)?.value

    // TTS restores names before Speaker 1/2 substitution. Edit workflows
    // restore the user's text rather than internal conditioning guidance.
    const originalPrompt = (p._tts_original_prompt as string) || (
      p.edit_sub_mode === 'recast' && typeof p.edit_recast_raw_prompt === 'string'
        ? p.edit_recast_raw_prompt as string
        : p.edit_sub_mode === 'outpaint' && typeof p.edit_outpaint_raw_prompt === 'string'
          ? p.edit_outpaint_raw_prompt as string
          : restoredH3SourcePrompt || p.prompt as string
    ) || ''

    // Build params from metadata
    // For image_mode: use 1 (I2V UI toggle) if start image was used, else 0
    const newParams: Partial<GenerateParams> = {
      prompt: originalPrompt,
      model_type: modelType,
      resolution: (p.resolution as string) || '1280x720',
      video_length: (p.video_length as number) || 81,
      num_inference_steps: migratedLegacyRecast ? 8 : (p.num_inference_steps as number) || 20,
      guidance_scale: migratedLegacyRecast ? 1 : (p.guidance_scale as number) || 5.0,
      seed: (p.seed as number) ?? -1,
      // Restore the ACTUAL saved output mode (0 = video, 1 = image). The old
      // `hadStartImage ? 1 : 0` was wrong: an I2V *video* clip has a start image
      // but image_mode 0 — inferring 1 from the start image put the UI in image-
      // output mode, so a later T2V (after clearing the start image) emitted a PNG.
      image_mode: (p.image_mode as number) ?? 0,
      negative_prompt: (p.negative_prompt as string) || '',
      repeat_generation: 1,
      activated_loras: (p.activated_loras as string[]) || [],
      loras_multipliers: (p.loras_multipliers as string) || '',
      minimax_h3_references: Array.isArray(p.minimax_h3_references)
        ? (p.minimax_h3_references as GenerateParams['minimax_h3_references'])?.filter(
            reference => !(
              reference as { _maestro_generated_continuity?: boolean }
            )._maestro_generated_continuity,
          )
        : undefined,
      minimax_h3_reference_detail: (
        p.minimax_h3_reference_detail === 'max'
          ? 'max'
          : (p.minimax_h3_reference_detail === 'match' ? 'match' : undefined)
      ),
      settings_version: p.settings_version as number,
    }

    // Copy optional fields — explicitly clear when absent to prevent stale values leaking
    newParams.sliding_window_size = (p.sliding_window_size as number) ?? undefined
    newParams.sliding_window_overlap = (p.sliding_window_overlap as number) ?? undefined
    newParams.sliding_window_discard_last_frames = (
      p.sliding_window_discard_last_frames as number
    ) ?? undefined
    newParams.sliding_window_memory_override = p.sliding_window_memory_override === true
    newParams.guidance_phases = (p.guidance_phases as number) ?? undefined
    newParams.video_prompt_type = (p.video_prompt_type as string) || ''
    newParams.audio_prompt_type = (p.audio_prompt_type as string) || ''
    newParams.image_prompt_type = (p.image_prompt_type as string) || ''
    newParams.input_video_strength = (p.input_video_strength as number) ?? undefined
    newParams.flow_shift = migratedLegacyRecast ? 1 : (p.flow_shift as number) ?? undefined
    newParams.self_refiner_setting = (p.self_refiner_setting as number) ?? undefined
    newParams.audio_guide = (p.audio_guide as string) || ''
    newParams.audio_scale = (p.audio_scale as number) ?? undefined
    newParams.audio_guide2 = (p.audio_guide2 as string) || ''
    // Style / Music Caption (ACE-Step). Was never copied here, so the
    // pencil restored only the lyrics — clear when absent so a stale
    // caption can't leak into an unrelated restore.
    newParams.alt_prompt = (p.alt_prompt as string) || ''
    newParams.video_guide = (p.video_guide as string) || ''
    newParams.video_mask = (p.video_mask as string) || ''
    newParams.denoising_strength = (p.denoising_strength as number) ?? undefined
    newParams.masking_strength = (p.masking_strength as number) ?? undefined
    newParams.minimax_h3_control_visual_mode = (
      p.minimax_h3_control_visual_mode === 'prompt'
      || p.minimax_h3_control_visual_mode === 'whole'
      || p.minimax_h3_control_visual_mode === 'inside'
      || p.minimax_h3_control_visual_mode === 'outside'
    ) ? p.minimax_h3_control_visual_mode : (
      String(p.video_prompt_type || '').includes('A')
        ? (String(p.video_prompt_type || '').includes('N') ? 'outside' : 'inside')
        : Number(p.denoising_strength ?? 1) < 1
          ? 'whole'
          : 'prompt'
    )
    newParams.image_refs = Array.isArray(p.image_refs) ? (p.image_refs as string[]) : []
    newParams.frames_positions = (p.frames_positions as string) || ''
    newParams.injection_strength = (p.injection_strength as number) ?? undefined
    newParams.remove_background_images_ref = (p.remove_background_images_ref as number) ?? 0

    // Progressive 3-stage pipeline settings
    if (p.progressive_pipeline) {
      (newParams as Record<string, unknown>).progressive_pipeline = true;
      (newParams as Record<string, unknown>).progressive_stage1_image_weight = (p.progressive_stage1_image_weight as number) ?? 0.7;
      (newParams as Record<string, unknown>).progressive_stage2_steps = (p.progressive_stage2_steps as number) ?? 8;
      (newParams as Record<string, unknown>).progressive_stage3_steps = (p.progressive_stage3_steps as number) ?? 3;
      (newParams as Record<string, unknown>).progressive_stage2_sigma = (p.progressive_stage2_sigma as number) ?? 1.0;
      (newParams as Record<string, unknown>).progressive_stage3_sigma = (p.progressive_stage3_sigma as number) ?? 0.85;
      (newParams as Record<string, unknown>).progressive_stage3_image_weight = (p.progressive_stage3_image_weight as number) ?? 0.7
    }
    // Single-stage distilled mode — mutually exclusive with progressive above
    if (p.single_stage_pipeline) {
      (newParams as Record<string, unknown>).single_stage_pipeline = true;
      (newParams as Record<string, unknown>).progressive_pipeline = false;
    }
    // Reference two-stage pipeline (10Eros) — restore so re-generating an
    // STG-era sidecar reproduces the pipeline that made it.
    (newParams as Record<string, unknown>).reference_pipeline = (p.reference_pipeline as boolean) ?? undefined;

    // Advanced pipeline settings
    (newParams as Record<string, unknown>).stage2_steps = (p.stage2_steps as number) ?? undefined;
    (newParams as Record<string, unknown>).stg_scale = (p.stg_scale as number) ?? undefined;
    // Perturbation config rides along with stg_scale so re-generating an STG
    // run is faithful. Old sidecars (pre-STG-wiring) simply lack these keys.
    (newParams as Record<string, unknown>).perturbation_switch = (p.perturbation_switch as number) ?? undefined;
    (newParams as Record<string, unknown>).perturbation_layers = Array.isArray(p.perturbation_layers) ? (p.perturbation_layers as number[]) : undefined;
    (newParams as Record<string, unknown>).perturbation_start_perc = (p.perturbation_start_perc as number) ?? undefined;
    (newParams as Record<string, unknown>).perturbation_end_perc = (p.perturbation_end_perc as number) ?? undefined;
    (newParams as Record<string, unknown>).cfg_rescale = (p.cfg_rescale as number) ?? undefined;
    (newParams as Record<string, unknown>).modality_scale = (p.modality_scale as number) ?? undefined;
    (newParams as Record<string, unknown>).use_gradient_estimation = (p.use_gradient_estimation as boolean) ?? undefined;
    (newParams as Record<string, unknown>).ge_gamma = (p.ge_gamma as number) ?? undefined;
    (newParams as Record<string, unknown>).keyframe_conditioning_mode = (p.keyframe_conditioning_mode as string) ?? undefined;
    (newParams as Record<string, unknown>).keyframe_inject_mode = (p.keyframe_inject_mode as string) ?? undefined;
    (newParams as Record<string, unknown>).temperature = (p.temperature as number) ?? undefined;
    (newParams as Record<string, unknown>).audio_guidance_scale = (p.audio_guidance_scale as number) ?? undefined
    // H3 optimization controls are a cohesive saved recipe. Explicit off
    // values matter: undefined would retain the clip selected before this one.
    newParams.override_attention = p.override_attention === 'sol' ? 'sol' : ''
    newParams.skip_steps_cache_type = (
      p.skip_steps_cache_type === 'first_block' ? 'first_block' : ''
    )
    newParams.skip_steps_multiplier = Number.isFinite(Number(p.skip_steps_multiplier))
      ? Number(p.skip_steps_multiplier)
      : restoredModelOptions?.default_skip_steps_multiplier
    newParams.skip_steps_start_step_perc = Number.isFinite(
      Number(p.skip_steps_start_step_perc),
    )
      ? Math.max(0, Math.min(100, Number(p.skip_steps_start_step_perc)))
      : restoredModelOptions?.default_skip_steps_start_step_perc
    newParams.minimax_h3_turbo_mode = (
      p.minimax_h3_turbo_mode === true || legacyTurboEnabled
    )
    newParams.minimax_h3_turbo_preset = restoredTurboPreset?.id
    newParams.minimax_h3_text_encoder = restoredTextEncoder
    newParams.ltx25_video_vae = restoredLtx25VideoVae
    newParams.minimax_h3_window_storyboard = (p.minimax_h3_window_storyboard as boolean) ?? undefined
    newParams.minimax_h3_multi_window = (p.minimax_h3_multi_window as boolean) ?? undefined
    const legacyLtxLongForm = (
      /^ltx(?:v|2)/i.test(String(p.model_type || ''))
      && Number(p.video_length || 0) > Number(p.sliding_window_size || 0)
    )
    newParams.ltx_multi_window = (p.ltx_multi_window as boolean)
      ?? (legacyLtxLongForm ? true : undefined)
    newParams.ltx_window_prompt_mode = (
      p.ltx_window_prompt_mode === 'manual'
        ? 'manual'
        : (p.ltx_window_prompt_mode === 'auto'
            ? 'auto'
            : (legacyLtxLongForm ? 'auto' : undefined))
    )
    newParams.ltx_window_prompts = Array.isArray(p.ltx_window_prompts)
      ? (p.ltx_window_prompts as string[]).filter(item => typeof item === 'string' && item.trim())
      : undefined
    newParams._ltx_original_prompt = (
      typeof p._ltx_original_prompt === 'string'
      && p._ltx_original_prompt.trim()
    ) ? p._ltx_original_prompt : undefined
    newParams.minimax_h3_reference_sequence = (p.minimax_h3_reference_sequence as boolean) ?? undefined
    newParams.minimax_h3_sequence_prompt_mode = restoredH3SequencePromptMode
    newParams.minimax_h3_sequence_continuity = (p.minimax_h3_sequence_continuity as boolean) ?? undefined
    newParams.minimax_h3_sequence_clip_frames = (
      p.minimax_h3_sequence_clip_frames as number
    ) ?? undefined
    newParams.minimax_h3_sequence_memory_override = (
      p.minimax_h3_sequence_memory_override as boolean
    ) ?? undefined
    newParams.minimax_h3_camera_coverage = (
      p.minimax_h3_camera_coverage === 'continuous'
      || p.minimax_h3_camera_coverage === 'multi_shot'
    ) ? p.minimax_h3_camera_coverage : 'auto'
    newParams._h3_original_prompt = (
      typeof p._h3_original_prompt === 'string'
      && p._h3_original_prompt.trim()
    ) ? p._h3_original_prompt : undefined
    // Detect multi-clip output and reconstruct clips
    if (p.multi_prompts_gen_type === 3 && Array.isArray(p.image_start)) {
      // Director Mode joins per-clip prompts with `\n---CLIP_BOUNDARY---\n`
      // (see app/launch.py:7279). Studio Mode multi-shot joins with plain
      // `\n` (single-line prompts only). Split on the boundary token first
      // so Director prompts that contain their own newlines survive; fall
      // back to plain newline split for legacy Studio multi-clip sidecars
      // that don't carry the boundary marker.
      //
      // Before this fix: every internal `\n` in a Director clip prompt
      // became a clip break, doubling+ the clip count and leaving half of
      // them with the literal string `---CLIP_BOUNDARY---` as their prompt.
      // The visible symptom was "some prompts populate but others don't"
      // and start-image indices going to the wrong clips.
      const promptText = (p.prompt as string) || ''
      const CLIP_BOUNDARY = '\n---CLIP_BOUNDARY---\n'
      const promptLines = promptText.includes(CLIP_BOUNDARY)
        ? promptText.split(CLIP_BOUNDARY).map(s => s.trim()).filter(Boolean)
        : promptText.split('\n').map(s => s.trim()).filter(Boolean)
      const imagePaths = p.image_start as string[]
      // Per-clip durations (Director Mode populates this; Studio mode may not).
      // Saved by app/launch.py as part of raw_params before per-clip split;
      // survives onto the concat multiclip sidecar (see real sidecar example
      // in app/outputs/Testing04/...multiclip.meta.json line 13-26).
      const rawPerClipFrames = Array.isArray(p.per_clip_frames)
        ? (p.per_clip_frames as number[])
        : []
      const perClipFrames = restoredIsH3
        ? normalizeH3ClipFrameSchedule(
            rawPerClipFrames,
            restoredModelOptions?.frames_minimum ?? 124,
            restoredModelOptions?.frames_maximum ?? 345,
            restoredModelOptions?.frames_steps ?? 17,
          )
        : rawPerClipFrames
      if (perClipFrames.length > 0) {
        newParams.per_clip_frames = perClipFrames
        newParams.video_length = perClipFrames.reduce((total, value) => total + value, 0)
        newParams.sliding_window_size = Math.max(...perClipFrames)
      } else {
        newParams.per_clip_frames = undefined
      }
      // Per-clip keyframe images (Director Mode KFI feature). Array of arrays
      // — each inner array holds the keyframe paths for that clip. Studio
      // Mode multi-shot generations don't use this field today.
      const perClipKeyframes = Array.isArray(p.per_clip_keyframes) ? (p.per_clip_keyframes as string[][]) : []
      const clipCount = Math.max(promptLines.length, imagePaths.length, perClipFrames.length)
      const clips: MultiClip[] = []
      for (let i = 0; i < clipCount; i++) {
        clips.push({
          prompt: promptLines[i] || '',
          startImage: null,
          startImagePath: imagePaths[i] || null,
          endImage: null,
          endImagePath: null,
          durationFrames: perClipFrames[i] || undefined,
        })
      }
      set({ clips, singlePromptMode: false })
      newParams.image_mode = 2
      newParams.multi_prompts_gen_type = 3

      // Surface per-clip keyframes via image_refs + frames_positions so
      // ControlVideoSection's restore picks them up. NOTE: MultiClip's type
      // doesn't yet carry per-clip keyframes, so all clips' keyframes get
      // concatenated into a single image_refs array with "L" positions
      // (the same encoding launch.py uses at line 7353). Re-running the
      // generation will dispatch keyframes to clips by position order,
      // matching the original layout. Documented as a known limitation:
      // editing one clip's keyframes after restore affects the whole pool.
      if (perClipKeyframes.length > 0) {
        const flatRefs: string[] = []
        const flatPositions: string[] = []
        for (const clipKfs of perClipKeyframes) {
          if (Array.isArray(clipKfs)) {
            for (const kf of clipKfs) {
              if (kf) {
                flatRefs.push(kf)
                flatPositions.push('L')
              }
            }
          }
        }
        if (flatRefs.length > 0) {
          newParams.image_refs = flatRefs
          newParams.frames_positions = flatPositions.join(' ')
          // Ensure KFI is in video_prompt_type so ControlVideoSection
          // recognizes the inject-frame mode on restore.
          const vpt = newParams.video_prompt_type || ''
          if (!vpt.includes('KFI')) {
            newParams.video_prompt_type = vpt + 'KFI'
          }
        }
      }

      // Fetch clip images from upload URLs to show previews. Prefer
      // upload_filenames.image_start (already-extracted basenames) when
      // present; fall back to deriving basenames from params.image_start
      // paths so older sidecars without upload_filenames still restore.
      const uploadNames = Array.isArray(uploadFilenames?.image_start)
        ? uploadFilenames.image_start as string[]
        : imagePaths.map(p => (p || '').replace(/\\/g, '/').split('/').pop() || '')
      for (let i = 0; i < clipCount; i++) {
        const fname = uploadNames[i]
        if (fname) {
          const idx = i
          fetch(api.getUploadUrl(fname))
            .then(r => r.ok ? r.blob() : null)
            .then(blob => {
              if (!blob) return
              const file = new File([blob], fname, { type: blob.type })
              get().setClipStartImage(idx, file)
            })
            .catch(() => {})
        }
      }
    } else {
      // Set or clear attachment paths from sidecar
      newParams.image_start = p.image_start ? (p.image_start as string) : ''
      set({ clips: [], singlePromptMode: false })
    }
    newParams.image_end = p.image_end ? (p.image_end as string) : ''

    // Rebuild lora weights from multipliers string
    const loraWeights: Record<string, number[]> = {}
    const loras = newParams.activated_loras || []
    const multParts = (newParams.loras_multipliers || '').split(' ').filter(Boolean)
    for (let i = 0; i < loras.length; i++) {
      const parts = (multParts[i] || '1.00').split(';').map(Number)
      loraWeights[loras[i]] = parts
    }

    // Restore duration from metadata
    const restoredDuration = (p.duration_seconds as number) || 0
    // Restore post-processing settings from metadata
    const restoredSpatialUpsampling = (p.spatial_upsampling as string) || ''
    const restoredFilmGrainIntensity = (p.film_grain_intensity as number) || 0
    const restoredFilmGrainSaturation = (p.film_grain_saturation as number) || 0.5

    // Restore audio guide filename from upload_filenames. Fall back to
    // deriving basename from params.audio_guide for sidecars that pre-date
    // the upload_filenames extraction code.
    const _deriveBase = (val: unknown): string | null => {
      if (typeof val !== 'string' || !val) return null
      const bn = val.replace(/\\/g, '/').split('/').pop()
      return bn || null
    }
    const restoredAudioGuideFilename =
      (typeof uploadFilenames?.audio_guide === 'string' ? uploadFilenames.audio_guide : null)
      || _deriveBase(p.audio_guide)
    const restoredAudioGuide2Filename =
      (typeof uploadFilenames?.audio_guide2 === 'string' ? uploadFilenames.audio_guide2 : null)
      || _deriveBase(p.audio_guide2)
    // Restore TTS speaker names (1-6)
    const restoredSpeakerName1 = (p._tts_speaker_name1 as string) || ''
    const restoredSpeakerName2 = (p._tts_speaker_name2 as string) || ''
    const restoredVoiceCount = (p._tts_voice_count as number) || 0
    const restoredVoices: { name: string; filename: string | null; path: string | null }[] = []
    for (let i = 0; i < Math.max(restoredVoiceCount, 2); i++) {
      const name = (p[`_tts_speaker_name${i + 1}`] as string) || ''
      if (name || i < restoredVoiceCount) {
        restoredVoices.push({ name, filename: null, path: null })
      }
    }

    set(s => ({
      params: { ...s.params, ...newParams },
      h3WindowPlan: restoredH3WindowPlan,
      loraWeights,
      startImage: null,
      endImage: null,
      imageRefs: [],  // Clear — will repopulate below if image_refs exist
      outputCount: 1,
      ...(restoredDuration > 0 ? { durationSeconds: restoredDuration } : {}),
      spatialUpsampling: restoredSpatialUpsampling,
      filmGrainIntensity: restoredFilmGrainIntensity,
      filmGrainSaturation: restoredFilmGrainSaturation,
      audioGuideFilename: restoredAudioGuideFilename,
      audioGuide2Filename: restoredAudioGuide2Filename,
      // TTS state
      ...(restoredSpeakerName1 || restoredSpeakerName2 || restoredVoiceCount > 0 ? {
        ttsSpeakerName1: restoredSpeakerName1,
        ttsSpeakerName2: restoredSpeakerName2,
        ttsSpeakerNamesManual: true,
        ttsVoiceCount: restoredVoiceCount,
        ttsVoices: restoredVoices,
      } : {}),
    }))

    // Restore image refs as File objects (for image mode reference images)
    // Skip if this is a KFI (frames injection) output — those refs are handled by ControlVideoSection
    const imageRefPaths = newParams.image_refs || []
    const isKFI = (newParams.video_prompt_type || '').includes('KFI')
    if (imageRefPaths.length > 0 && !isKFI) {
      // Set the ref type from saved params
      const vpt = newParams.video_prompt_type || ''
      const refType = vpt.includes('K') && vpt.includes('I') ? 'KI' : vpt.includes('I') ? 'I' : 'KI'
      set({ imageRefType: refType })

      // Fetch all ref images in parallel, then set in original order
      const refPromises = imageRefPaths.map(refPath => {
        const fname = refPath.replace(/\\/g, '/').split('/').pop() || ''
        if (!fname) return Promise.resolve(null)
        // /file searches active/all workspaces and uploads, so both generated
        // references and newly uploaded images restore correctly.
        const url = api.getFileUrl(fname)
        return fetch(url)
          .then(r => r.ok ? r.blob() : null)
          .then(blob => blob ? new File([blob], fname, { type: blob.type || 'image/png' }) : null)
          .catch(() => null)
      })
      Promise.all(refPromises).then(files => {
        const ordered = files.filter((f): f is File => f !== null)
        set({ imageRefs: ordered })
      })
    }

    // Derive duration and sliding window from video_length and fps
    const fps = model?.fps || 16
    const frames = newParams.video_length || 81
    set({ durationSeconds: Math.round((frames / fps) * 10) / 10 })
    const restoredNativePassFrames = newParams.minimax_h3_sequence_clip_frames
      ?? newParams.sliding_window_size
    if (restoredNativePassFrames) {
      set({
        slidingWindowSeconds: restoredNativePassFrames / fps,
        slidingWindowLocked: newParams.minimax_h3_reference_sequence === true
          ? newParams.minimax_h3_sequence_memory_override === true
          : newParams.sliding_window_memory_override === true,
      })
    }
    if (newParams.sliding_window_overlap != null) {
      set({ slidingWindowOverlap: newParams.sliding_window_overlap })
    }

    // Derive resolution preset and aspect ratio
    const res = newParams.resolution || '1280x720'
    const resolutionSelection = findResolutionSelection(res, get().modelOptions)
    if (resolutionSelection) {
      set({
        resolutionPreset: resolutionSelection.preset,
        aspectRatio: resolutionSelection.ratio,
      })
    }

    // Restore start/end images from upload URLs as File objects. Prefer
    // upload_filenames.image_{start,end} (basename); fall back to deriving
    // from the full path in params for sidecars missing upload_filenames.
    const startFile = (typeof uploadFilenames?.image_start === 'string'
      ? uploadFilenames.image_start
      : null) || _deriveBase(p.image_start)
    const endFile = (typeof uploadFilenames?.image_end === 'string'
      ? uploadFilenames.image_end
      : null) || _deriveBase(p.image_end)
    if (hadStartImage && startFile) {
      fetch(api.getUploadUrl(startFile))
        .then(r => r.ok ? r.blob() : null)
        .then(blob => {
          if (!blob) return
          const file = new File([blob], startFile, { type: blob.type })
          set({ startImage: file })
        })
        .catch(() => {})
    }
    if (hadEndImage && endFile) {
      fetch(api.getUploadUrl(endFile))
        .then(r => r.ok ? r.blob() : null)
        .then(blob => {
          if (!blob) return
          const file = new File([blob], endFile, { type: blob.type })
          set({ endImage: file })
        })
        .catch(() => {})
    }

    // ── Edit Mode restore ───────────────────────────────────────────────
    // If the sidecar carries edit_sub_mode, this output was made by the
    // Retake / Inpaint / Outpaint / Restyle / Edit Anything sub-modes.
    // Switch the sidebar into the matching mode and re-populate the
    // sub-mode-specific controls. The standard restore above already set
    // generationMode from the model family, so we override here when the
    // sidecar tag is authoritative.
    const editSubMode = (p.edit_sub_mode as string) || ''
    if (editSubMode) {
      set({
        generationMode: 'avatar',
        editSubMode: editSubMode as 'retake' | 'inpaint' | 'restyle' | 'outpaint' | 'edit_anything' | 'recast',
      })

      // Re-link the source video. The sidecar stores either edit_video_path
      // (preferred — set by the new endpoints) or falls back to retake_video.
      // We fetch the file by URL so the EditVideoUpload UI shows the same
      // clip the user originally edited.
      const editVideoPath = (p.edit_video_path as string) || (p.retake_video as string) || ''
      if (editVideoPath) {
        const fname = editVideoPath.replace(/\\/g, '/').split('/').pop() || ''
        const url = api.getFileUrl(fname)
        // Probe metadata via a hidden <video> first so duration/resolution
        // are correct, then fetch the blob to populate editVideoFile.
        if (fname) {
          const video = document.createElement('video')
          video.src = url
          video.muted = true
          video.onloadedmetadata = () => {
            const duration = video.duration && isFinite(video.duration) ? video.duration : 0
            const resolution = `${video.videoWidth}x${video.videoHeight}`
            fetch(url)
              .then(r => r.ok ? r.blob() : null)
              .then(blob => {
                if (!blob) return
                const file = new File([blob], fname, { type: blob.type || 'video/mp4' })
                get().setEditVideo(file, editVideoPath, url, duration, resolution)
              })
              .catch(() => {})
          }
          // If metadata never loads (file moved/deleted), still set the path
          // so the user can re-attach manually.
          set({ editVideoPath, editVideoUrl: url })
        }
      }

      // Trim range — applies to retake, inpaint, edit_anything, outpaint.
      const trimStart = (p.edit_start_time as number) ?? (p.outpaint_trim_start as number)
      const trimEnd = (p.edit_end_time as number) ?? (p.outpaint_trim_end as number)
      if (trimStart != null && trimStart >= 0) {
        set({ editStartTime: trimStart })
        if (editSubMode === 'outpaint') set({ outpaintTrimStart: trimStart })
      }
      if (trimEnd != null && trimEnd > 0) {
        set({ editEndTime: trimEnd })
        if (editSubMode === 'outpaint') set({ outpaintTrimEnd: trimEnd })
      }

      // Sub-mode-specific knobs
      if (editSubMode === 'retake' || editSubMode === 'inpaint' || editSubMode === 'edit_anything') {
        if (p.retake_strength != null) set({ editRetakeStrength: p.retake_strength as number })
        if (p.retake_engine) set({ editRetakeEngine: p.retake_engine as 'native' | 'legacy' })
        if (p.regenerate_audio != null) set({ editRegenerateAudio: !!p.regenerate_audio })
      }
      if (editSubMode === 'inpaint') {
        if (p.edit_target) set({ editDetectedTarget: p.edit_target as string })
        if (p.retake_masks_path) set({ editMasksPath: p.retake_masks_path as string })
      }
      if (editSubMode === 'edit_anything') {
        if (p.edit_anything_lora_strength != null) {
          set({ editAnythingLoraStrength: p.edit_anything_lora_strength as number })
        }
      }
      if (editSubMode === 'restyle') {
        const savedRepaintMappings = Array.isArray(p.edit_repaint_region_mappings)
          ? p.edit_repaint_region_mappings
            .slice(0, 5)
            .map((raw, index): RepaintRegionMapping | null => {
              if (!raw || typeof raw !== 'object') return null
              const mapping = raw as Record<string, unknown>
              const source = String(mapping.source || '').trim()
              const target = String(mapping.target || '').trim()
              if (!source || !target) return null
              return {
                id: String(mapping.id || `repaint-${index + 1}`),
                source,
                target,
              }
            })
            .filter((mapping): mapping is RepaintRegionMapping => mapping !== null)
          : []
        const repaintFrame = String(p.edit_repaint_target_frame || p.image_start || '')
        const repaintFrameName = repaintFrame.replace(/\\/g, '/').split('/').pop() || ''
        set({
          editRepaintMappings: savedRepaintMappings,
          editRepaintResolutionProfile: p.edit_repaint_resolution_profile === '704p'
            ? '704p'
            : p.edit_repaint_resolution_profile === '512p'
              ? '512p'
              : '480p',
          editRepaintFrameFile: null,
          editRepaintFramePath: repaintFrame,
          editRepaintFrameUrl: repaintFrameName ? api.getFileUrl(repaintFrameName) : '',
        })
        if (repaintFrame && repaintFrameName) {
          const repaintUrl = api.getFileUrl(repaintFrameName)
          fetch(repaintUrl)
            .then(r => r.ok ? r.blob() : null)
            .then(blob => {
              if (!blob) return
              get().setEditRepaintFrame(
                new File([blob], repaintFrameName, { type: blob.type || 'image/png' }),
                repaintFrame,
                URL.createObjectURL(blob),
              )
            })
            .catch(() => {})
        }
      }
      if (editSubMode === 'recast') {
        const savedMappings = p.edit_recast_character_mappings
        if (Array.isArray(savedMappings)) {
          const restoredMappings = savedMappings
            .slice(0, 5)
            .map((raw, index): RecastCharacterMapping | null => {
              if (!raw || typeof raw !== 'object') return null
              const mapping = raw as Record<string, unknown>
              const refPath = String(mapping.ref_image_path || '')
              const target = String(mapping.target || '').trim()
              if (!refPath || !target) return null
              const refName = refPath.replace(/\\/g, '/').split('/').pop() || ''
              const additionalPaths = Array.isArray(mapping.additional_ref_image_paths)
                ? mapping.additional_ref_image_paths.map(path => String(path || '')).filter(Boolean)
                : []
              return {
                id: String(mapping.id || `recast-${index + 1}`),
                target,
                refFile: null,
                refPath,
                refUrl: api.getFileUrl(refName),
                additionalRefs: additionalPaths.map(path => {
                  const name = path.replace(/\\/g, '/').split('/').pop() || ''
                  return { file: null, path, url: api.getFileUrl(name) }
                }),
                referenceAlignedToSource: mapping.reference_aligned_to_source === true,
              }
            })
            .filter((mapping): mapping is RecastCharacterMapping => mapping !== null)
          if (restoredMappings.length > 0) {
            set({
              editRecastMappings: restoredMappings,
              editRecastTarget: restoredMappings[0].target,
              editRecastPersonCount: restoredMappings.length,
              editRecastRefFile: null,
              editRecastRefPath: restoredMappings[0].refPath,
              editRecastRefUrl: restoredMappings[0].refUrl,
              editRecastRefAligned: restoredMappings[0].referenceAlignedToSource,
            })
          }
        }
        if (!Array.isArray(savedMappings) || savedMappings.length === 0) {
          set(s => ({
            editRecastMappings: [{
              ...(s.editRecastMappings[0] || DEFAULT_RECAST_MAPPING),
              target: String(p.edit_recast_target || 'person'),
              referenceAlignedToSource: p.edit_recast_ref_aligned === true,
            }],
          }))
        }
        if (p.edit_recast_target) set({ editRecastTarget: p.edit_recast_target as string })
        if (p.edit_recast_person_count != null) {
          const count = Number(p.edit_recast_person_count)
          set({ editRecastPersonCount: Math.min(5, Math.max(1, Number.isFinite(count) ? Math.round(count) : 1)) })
        }
        set({
          editRecastIsolateReference: p.edit_recast_isolate_reference !== false,
          editRecastAutoFaceDetail: p.edit_recast_auto_face_detail !== false,
          editRecastEnhancePrompt: p.edit_recast_enhance_prompt === true,
          editRecastProtectBystanders: p.edit_recast_protect_bystanders === true,
          editRecastPreserveBystanders: p.edit_recast_preserve_bystanders !== undefined
            ? p.edit_recast_preserve_bystanders === true
            : p.edit_recast_preserve_scene_reference !== undefined
              ? p.edit_recast_preserve_scene_reference === true
              : true,
          editRecastUseRelighting: p.edit_recast_use_relighting === true,
          editRecastResolutionProfile: p.edit_recast_resolution_profile === '704p'
            ? '704p'
            : p.edit_recast_resolution_profile === '512p'
              ? '512p'
              : '480p',
        })
        const recastRef = (p.edit_recast_ref_path as string) || ''
        if (recastRef) {
          const refName = recastRef.replace(/\\/g, '/').split('/').pop() || ''
          // Recast references can be either uploads or Image-mode outputs.
          const refUrl = api.getFileUrl(refName)
          fetch(refUrl)
            .then(r => r.ok ? r.blob() : null)
            .then(blob => {
              if (!blob) return
              const file = new File([blob], refName, { type: blob.type || 'image/png' })
              get().setEditRecastRef(
                file,
                recastRef,
                URL.createObjectURL(file),
                p.edit_recast_ref_aligned === true,
              )
            })
            .catch(() => {})
        }
      }
      if (editSubMode === 'outpaint') {
        // Padding (pixels) — preserved as-is; the OutpaintCanvas reads
        // outpaintAspect + outpaintVideoBox to compose, but we also mirror
        // the pixel pads to outpaintPadding so legacy code paths line up.
        const padTop = (p.outpaint_pad_top as number) ?? 0
        const padBottom = (p.outpaint_pad_bottom as number) ?? 0
        const padLeft = (p.outpaint_pad_left as number) ?? 0
        const padRight = (p.outpaint_pad_right as number) ?? 0
        set({ outpaintPadding: { top: padTop, bottom: padBottom, left: padLeft, right: padRight } })

        const canvasW = (p._outpaint_canvas_w as number) || 0
        const canvasH = (p._outpaint_canvas_h as number) || 0
        const savedAspect = String(p.outpaint_aspect || '') as OutpaintAspect
        const validSavedAspect = (
          savedAspect === 'source'
          || _OUTPAINT_ASPECT_RATIOS.some(([aspect]) => aspect === savedAspect)
        )
        let restoredAspect: OutpaintAspect | null = validSavedAspect ? savedAspect : null
        if (!restoredAspect) {
          let aspectW = canvasW
          let aspectH = canvasH
          if (aspectW <= 0 || aspectH <= 0) {
            const resolutionMatch = /^(\d+)x(\d+)$/i.exec(String(p.resolution || '').trim())
            if (resolutionMatch) {
              aspectW = Number(resolutionMatch[1])
              aspectH = Number(resolutionMatch[2])
            }
          }
          restoredAspect = _inferOutpaintAspect(aspectW, aspectH)
        }
        if (restoredAspect) set({ outpaintAspect: restoredAspect })
        if (p.outpaint_resolution_preset) {
          set({ outpaintResolutionPreset: p.outpaint_resolution_preset as 'auto' | '480p' | '540p' | '720p' | '1080p' })
        }
        if (p.outpaint_source_preservation != null) {
          set({ outpaintSourcePreservation: p.outpaint_source_preservation as number })
        }
        if (p.outpaint_lora_strength_ui != null) {
          set({ outpaintLoraStrength: p.outpaint_lora_strength_ui as number })
        }
        if (p.outpaint_mask_preserving != null) {
          set({ outpaintMaskPreserving: !!p.outpaint_mask_preserving })
        }

        // Recompute the canvas-relative video box from saved pad pixels +
        // saved canvas dimensions, so the OutpaintCanvas reproduces the
        // exact composition. Falls back to centered-fit if anything is
        // missing.
        if (canvasW > 0 && canvasH > 0) {
          const savedX = (p._outpaint_overlay_x as number) ?? padLeft
          const savedY = (p._outpaint_overlay_y as number) ?? padTop
          const srcW = (p._outpaint_overlay_w as number)
            || (canvasW - padLeft - padRight)
          const srcH = (p._outpaint_overlay_h as number)
            || (canvasH - padTop - padBottom)
          if (srcW > 0 && srcH > 0) {
            set({
              outpaintVideoBox: {
                x: savedX / canvasW,
                y: savedY / canvasH,
                w: srcW / canvasW,
                h: srcH / canvasH,
              },
            })
          }
        }

        // Audio/sync toggles
        if (p._outpaint_preserve_audio != null) {
          set({ outpaintPreserveSourceAudio: !!p._outpaint_preserve_audio })
        }
        if (p._outpaint_lock_source_pixels != null) {
          set({ outpaintLockSourcePixels: !!p._outpaint_lock_source_pixels })
        }
        if (p._outpaint_trim_smear != null) {
          set({ outpaintTrimSmear: !!p._outpaint_trim_smear })
        }
      }
    }
  },

  rerollGeneration: async () => {
    // Await the (now async, self-healing) settings load before generating, so a
    // slow on-demand metadata fetch can't let the reroll fire with stale params.
    await get().loadSettingsFromOutput()
    // Small delay to let state settle, then generate
    setTimeout(() => get().startGeneration(), 100)
  },

  rejoinClipGroup: async (groupId) => {
    try {
      const result = await api.rejoinClips(groupId)
      // Refresh outputs list to include the new concatenated file
      const outputsRes = await fetch('/api/v1/outputs')
      if (outputsRes.ok) {
        const data = await outputsRes.json()
        set({ outputs: data.files || [] })
      }
      // Select the new file
      const allOutputs = get().outputs
      const newIdx = allOutputs.findIndex(o => o.name === result.filename)
      if (newIdx >= 0) {
        set({ selectedOutput: newIdx })
        get().loadOutputMetadata(result.filename)
      }
    } catch (e) {
      console.error('Failed to rejoin clips:', e)
    }
  },

  deleteSelectedOutput: async () => {
    const outputs = get().filteredOutputs()
    const idx = get().selectedOutput
    const output = outputs[idx]
    if (!output) return

    try {
      await api.deleteOutput(output.name)
      // Remove from local state
      const allOutputs = get().outputs.filter(o => o.name !== output.name)
      const newIdx = Math.min(idx, Math.max(0, allOutputs.length - 1))
      set({ outputs: allOutputs, selectedOutput: newIdx })
      // Load metadata for new selection
      const newFiltered = get().filteredOutputs()
      if (newFiltered[newIdx]) {
        get().loadOutputMetadata(newFiltered[newIdx].name)
      } else {
        set({ selectedOutputMeta: null })
      }
    } catch (e) {
      console.error('Failed to delete output:', e)
    }
  },

  // ── Director Pipeline (server-side) ──────────────────────────────
  startDirectorPipeline: async () => {
    const state = get()
    const { directorPlannedClips, directorSceneDescription,
            directorAudioPath, directorAnalysis, directorReferenceImagePath,
            directorAutoMode, directorSeamless, directorShotImageGuidance,
            directorResolution, directorAspectRatio,
            directorVideoMaxShotFramesByModel, directorH3TurboModeByModel,
            directorH3TurboPresetByModel, directorH3SolModeByModel,
            directorH3FirstBlockCacheByModel,
            directorH3FirstBlockCacheMultiplierByModel,
            directorH3FirstBlockCacheWarmupByModel,
            selectedModelPerMode, savedParamsPerMode, savedLoraPerMode,
            directorSpeakerMappings, directorImageSpatialUpsampling,
            directorImageFilmGrainIntensity, directorImageFilmGrainSaturation,
            directorVideoSpatialUpsampling, directorVideoFilmGrainIntensity,
            directorVideoFilmGrainSaturation, directorVideoSelfRefiner,
            shortFilmPath, shortFilmCharacters, shortFilmTargetDuration,
            shortFilmNarrative } = state

    const selectedImageModel = selectedModelPerMode.image || 'flux2_klein_9b'
    const selectedVideoModel = selectedModelPerMode.video || 'ltx2_22B_distilled_1_1'

    // Director model choices are independent from whichever Studio model was
    // edited most recently. Hydrate each selected model's own tuned defaults,
    // then reuse saved Studio overrides only when they explicitly belong to
    // that same model. This prevents a distilled model's settings from
    // leaking into a full model (or vice versa) after a Director-only switch.
    const [imageModelDefaults, videoModelDefaults, fetchedImageOptions, fetchedVideoOptions] = await Promise.all([
      api.fetchDefaults(selectedImageModel).catch(() => ({})),
      api.fetchDefaults(selectedVideoModel).catch(() => ({})),
      api.fetchModelOptions(selectedImageModel).catch(() => null),
      api.fetchModelOptions(selectedVideoModel).catch(() => null),
    ])
    const cachedVideoOptions = state.modelOptions?.model_type === selectedVideoModel
      ? state.modelOptions
      : null
    const directorVideoOptions = fetchedVideoOptions || cachedVideoOptions
    const directorFixedMediaStrength = directorModelUsesFixedMediaStrength(
      selectedVideoModel,
      directorVideoOptions?.architecture
        || state.models.find(model => model.model_type === selectedVideoModel)?.architecture,
    )
    const directorImageOptions = fetchedImageOptions
    const directorImageResolution = resolveResolution(
      directorImageOptions,
      directorResolution,
      directorAspectRatio,
    )
    const directorVideoResolution = resolveResolution(
      directorVideoOptions,
      directorResolution,
      directorAspectRatio,
    )
    const fps = directorVideoOptions?.fps ?? 16
    const savedImageParams = savedParamsPerMode.image || {}
    const savedVideoParams = savedParamsPerMode.video || {}
    const matchingImageParams = savedImageParams.model_type === selectedImageModel
      ? savedImageParams
      : {}
    const matchingVideoParams = savedVideoParams.model_type === selectedVideoModel
      ? savedVideoParams
      : {}
    const rawDefaultVideoSteps = (
      directorVideoOptions?.default_num_inference_steps
      ?? (videoModelDefaults as Record<string, unknown>).num_inference_steps
      ?? 8
    )
    const parsedDefaultVideoSteps = Number(rawDefaultVideoSteps)
    const defaultVideoSteps = Number.isFinite(parsedDefaultVideoSteps) && parsedDefaultVideoSteps > 0
      ? Math.max(1, Math.min(50, Math.round(parsedDefaultVideoSteps)))
      : 8
    const configuredVideoSteps = state.directorVideoInferenceStepsByModel[selectedVideoModel]
    // A model may publish a fixed distilled recipe. In that case the model
    // default wins even if an older adjustable build left an override behind.
    let directorVideoSteps = directorVideoOptions?.lock_inference_steps
      ? defaultVideoSteps
      : (configuredVideoSteps ?? defaultVideoSteps)
    const directorTurboOption = directorVideoOptions?.minimax_h3_turbo
    const directorTurboPresets = directorTurboOption?.presets?.length
      ? directorTurboOption.presets
      : directorTurboOption
        ? [{
            id: directorTurboOption.preset_id,
            label: directorTurboOption.version_label,
            status: 'validated',
            filename: directorTurboOption.filename,
            steps: directorTurboOption.steps,
            weight: directorTurboOption.weight,
            weight_min: 0.5,
            weight_max: 1.0,
            description: directorTurboOption.guide,
            revision: '',
          }]
        : []
    const directorTurboPreset = (
      directorTurboPresets.find(
        preset => preset.id === directorH3TurboPresetByModel[selectedVideoModel],
      )
      || directorTurboPresets.find(
        preset => preset.id === directorTurboOption?.preset_id,
      )
      || directorTurboPresets[0]
    )
    const savedDirectorVideoLoras = savedLoraPerMode.video
    const directorTurboEnabled = Boolean(
      directorTurboOption && directorTurboPreset
      && directorH3TurboModeByModel[selectedVideoModel] === true
      && savedDirectorVideoLoras?.activated_loras?.includes(directorTurboPreset.filename)
    )
    if (directorTurboEnabled) directorVideoSteps = directorTurboPreset!.steps
    const directorSolEnabled = Boolean(
      directorVideoOptions?.sol_attention
      && directorVideoOptions.sol_attention_status?.supported
      && directorH3SolModeByModel[selectedVideoModel] === true
    )
    const directorFirstBlockCacheEnabled = Boolean(
      directorVideoOptions?.first_block_cache
      && directorH3FirstBlockCacheByModel[selectedVideoModel] === true
    )
    const cacheChoices = directorVideoOptions?.skip_steps_multiplier_choices || []
    const requestedCacheMultiplier = (
      directorH3FirstBlockCacheMultiplierByModel[selectedVideoModel]
      ?? directorVideoOptions?.default_skip_steps_multiplier
      ?? 0.08
    )
    const directorCacheMultiplier = cacheChoices.length
      ? cacheChoices.reduce((closest, choice) => (
          Math.abs(choice[1] - requestedCacheMultiplier)
            < Math.abs(closest - requestedCacheMultiplier)
            ? choice[1]
            : closest
        ), cacheChoices[0][1])
      : requestedCacheMultiplier
    const directorCacheWarmup = Math.max(0, Math.min(75, Math.round((
      directorH3FirstBlockCacheWarmupByModel[selectedVideoModel]
      ?? directorVideoOptions?.default_skip_steps_start_step_perc
      ?? 25
    ) / 5) * 5))
    const directorMaxShotFrames = directorVideoMaxShotFramesByModel[selectedVideoModel]

    // Upload all reference images (main + character + location) if not already uploaded
    let refImagePath = directorReferenceImagePath
    if (!refImagePath && state.directorReferenceImage) {
      try {
        const uploaded = await api.uploadImage(state.directorReferenceImage)
        refImagePath = uploaded.path
        set({ directorReferenceImagePath: refImagePath })
      } catch (e) {
        console.error('Failed to upload reference image for pipeline:', e)
      }
    }
    // Upload character refs that haven't been uploaded yet
    const charPaths = [...state.directorCharacterRefPaths]
    for (let i = charPaths.length; i < state.directorCharacterRefs.length; i++) {
      try {
        const uploaded = await api.uploadImage(state.directorCharacterRefs[i])
        charPaths.push(uploaded.path)
      } catch { /* skip failed uploads */ }
    }
    if (charPaths.length > state.directorCharacterRefPaths.length) {
      set({ directorCharacterRefPaths: charPaths })
    }
    // Upload location refs that haven't been uploaded yet
    const locPaths = [...state.directorLocationRefPaths]
    for (let i = locPaths.length; i < state.directorLocationRefs.length; i++) {
      try {
        const uploaded = await api.uploadImage(state.directorLocationRefs[i])
        locPaths.push(uploaded.path)
      } catch { /* skip failed uploads */ }
    }
    if (locPaths.length > state.directorLocationRefPaths.length) {
      set({ directorLocationRefPaths: locPaths })
    }
    const selectedVideoDefinition = state.models.find(
      model => model.model_type === selectedVideoModel,
    )
    const supportsVoiceReference = (
      selectedVideoDefinition?.director?.supports_voice_reference === true
    )
    const voiceReferenceMode = (
      selectedVideoDefinition?.director?.voice_reference_mode ?? 'none'
    )

    // Voice Reference is an LTX-2 ID-LoRA or a native H3 Omni audio
    // reference. Keep the local selection across model switches, but only
    // upload and submit it when the selected Director model can consume it.
    let voiceRefPath = state.directorVoiceRefPath
    if (supportsVoiceReference && !voiceRefPath && state.directorVoiceRef) {
      try {
        const uploaded = await api.uploadAudio(state.directorVoiceRef)
        voiceRefPath = uploaded.path
        set({ directorVoiceRefPath: voiceRefPath })
      } catch { /* skip */ }
    }

    // Determine pipeline type
    let pipelineType = 'music_video'
    if (shortFilmPath === 'story') pipelineType = 'short_film_story'
    else if (shortFilmPath === 'audio') pipelineType = 'short_film_audio'

    const pipelineParams: Record<string, unknown> = {
      pipeline_type: pipelineType,
      auto_mode: directorAutoMode,
      workspace: get().activeWorkspace,
      scene_description: directorSceneDescription,
      audio_path: directorAudioPath,
      // Audio analysis already produced this reusable stem for transcription.
      // LTX-2.5 can condition mouth motion on it while Director keeps the
      // untouched song as the final joined soundtrack.
      audio_vocals_path: directorAnalysis?.vocals_path || undefined,
      reference_image_path: refImagePath,
      character_ref_paths: charPaths.length > 0 ? charPaths : undefined,
      character_ref_labels: state.directorCharacterRefLabels.length > 0 ? state.directorCharacterRefLabels : undefined,
      location_ref_paths: locPaths.length > 0 ? locPaths : undefined,
      location_ref_labels: state.directorLocationRefLabels.length > 0 ? state.directorLocationRefLabels : undefined,
      planned_clips: directorPlannedClips,
      seamless: directorSeamless,
      shot_image_guidance: directorShotImageGuidance,
      director_resolution_preset: directorResolution,
      director_aspect_ratio: directorAspectRatio,
      director_max_shot_frames: directorMaxShotFrames,
      fps,
      frames_steps: directorVideoOptions?.frames_steps ?? 4,
      frames_minimum: directorVideoOptions?.frames_minimum ?? 5,

      // Director v2 flag — see prior callsites: ?? not || so explicit
      // user toggle-off is respected (legacy v1), only fall back to
      // true when the field is undefined.
      use_director_v2: state.servicesConfig?.use_director_v2 ?? true,

      // LLM
      llm_model_id: state.servicesConfig?.llm_model_id || state.llmStatus?.model_id,
      llm_device: state.servicesConfig?.llm_device || state.llmStatus?.device,
      llm_provider: state.servicesConfig?.llm_provider || 'local',
      lyrics: directorAnalysis?.lyrics || '',
      bpm: directorAnalysis?.bpm,
      speaker_mappings: directorSpeakerMappings,
      characters: shortFilmCharacters,
      target_duration: shortFilmTargetDuration,
      narrative_mode: shortFilmNarrative,

      // Image gen settings
      image_model: selectedImageModel,
      image_params: {
        ...imageModelDefaults,
        ...matchingImageParams,
        resolution: directorImageResolution,
      },
      image_loras: savedLoraPerMode.image || {},
      image_spatial_upsampling: directorImageSpatialUpsampling,
      image_film_grain_intensity: directorImageFilmGrainIntensity,
      image_film_grain_saturation: directorImageFilmGrainSaturation,

      // Video gen settings
      video_model: selectedVideoModel,
      video_params: {
        ...videoModelDefaults,
        ...matchingVideoParams,
        // Director owns this value. Studio's Advanced step count is separate
        // state and must not leak into a new Director project.
        num_inference_steps: directorVideoSteps,
        resolution: directorVideoResolution,
        minimax_h3_turbo_mode: directorTurboEnabled,
        minimax_h3_turbo_preset: directorTurboPreset?.id,
        override_attention: directorSolEnabled ? 'sol' : '',
        skip_steps_cache_type: directorFirstBlockCacheEnabled ? 'first_block' : '',
        skip_steps_multiplier: directorCacheMultiplier,
        skip_steps_start_step_perc: directorCacheWarmup,
        ...(directorFixedMediaStrength ? { input_video_strength: 1.0 } : {}),
      },
      video_loras: savedLoraPerMode.video || {},
      video_spatial_upsampling: directorVideoSpatialUpsampling,
      video_film_grain_intensity: directorVideoFilmGrainIntensity,
      video_film_grain_saturation: directorVideoFilmGrainSaturation,
      video_self_refiner: directorVideoSelfRefiner,
      audio_scale: directorFixedMediaStrength ? 1.0 : get().directorAudioScale,

      // Voice identity: LTX uses the CelebVHQ ID-LoRA; H3 Omni maps the
      // same upload into each shot's native Ref2VA manifest.
      ...(supportsVoiceReference && voiceRefPath ? {
        voice_reference: voiceRefPath,
        ...(voiceReferenceMode === 'id_lora' ? {
          identity_guidance_scale: state.directorIdentityGuidanceScale,
        } : {}),
      } : {}),
    }

    try {
      const { pipeline_id } = await api.startPipeline(pipelineParams)
      set({
        pipelineId: pipeline_id,
        pipelineStatus: null,
        pipelinePolling: true,
        directorStep: 'plan',
        directorLoading: true,
        directorError: null,
      })
      get().pollPipelineStatus()
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Pipeline failed to start'
      set({ directorError: msg })
    }
  },

  continuePipeline: async (updates) => {
    const pid = get().pipelineId
    if (!pid) return
    try {
      await api.continuePipeline(pid, updates)
      set({ directorLoading: true })
    } catch (e) {
      console.error('Failed to continue pipeline:', e)
    }
  },

  stopPipeline: async () => {
    const pid = get().pipelineId
    if (!pid) return
    try {
      await api.stopPipeline(pid)
      set({ pipelineId: null, pipelineStatus: null, pipelinePolling: false, directorLoading: false })
    } catch (e) {
      console.error('Failed to stop pipeline:', e)
    }
  },

  pollPipelineStatus: () => {
    const pid = get().pipelineId
    if (!pid) return

    const poll = async () => {
      if (!get().pipelinePolling || get().pipelineId !== pid) return

      try {
        const status = await api.fetchPipelineStatus(pid)
        set({ pipelineStatus: status })

        // Sync the backend's model-adapted plan, not just an initially empty
        // UI. H3 can split broad 20-30s music sections into additional native
        // <=14.4s shots after the browser has already populated its draft
        // timeline. The old empty-only guard left those stale durations and
        // prompts visible even though the worker queued the shorter plan.
        const currentPlans = get().directorClipPlans
        const currentTimeline = get().directorPlannedClips
        const plansChanged = Boolean(status.clip_plans?.length) && (
          currentPlans.length !== status.clip_plans.length
          || status.clip_plans.some((plan, index) => (
            plan.video_prompt !== currentPlans[index]?.video_prompt
            || plan.image_prompt !== currentPlans[index]?.image_prompt
          ))
        )
        const timelineChanged = Boolean(status.planned_clips?.length) && (
          currentTimeline.length !== status.planned_clips!.length
          || status.planned_clips!.some((clip, index) => (
            clip.start !== currentTimeline[index]?.start
            || clip.end !== currentTimeline[index]?.end
            || clip.duration_frames !== currentTimeline[index]?.duration_frames
          ))
        )
        if (plansChanged || timelineChanged) {
          set({
            ...(plansChanged ? { directorClipPlans: status.clip_plans } : {}),
            ...(timelineChanged ? { directorPlannedClips: status.planned_clips! } : {}),
            ...(!currentPlans.length && plansChanged ? { directorStep: 'review' as const } : {}),
          })
        }

        if (status.clip_images?.length) {
          // Strip empty filenames — those are failed-shot sentinels from the
          // pipeline (clip_images.append("") on exception). If we keep them,
          // downstream <img src={getFileUrl("")} /> hits /api/v1/file/ which
          // can resolve to a stale cached file rather than nothing, producing
          // the "same unrelated image over and over" symptom users see when
          // image gen fails (e.g. incompatible LoRA architecture).
          // clipIndex is captured BEFORE filtering so it stays aligned to
          // the original clip plan position even when failed shots drop out.
          const images = status.clip_images
            .map((filename, i) => ({
              clipIndex: i,
              prompt: status.clip_plans?.[i]?.image_prompt || '',
              file: null as unknown as File,
              filename,
            }))
            .filter(img => img.filename && img.filename.length > 0)
          set({ directorClipImages: images })
        }

        // Handle phase transitions
        if (status.phase === 'polishing_prompts') {
          set({
            directorImageGenProgress: {
              current: status.progress.current,
              total: status.progress.total,
              currentClipLabel: status.progress.message || 'Polishing prompts (3rd pass)...',
              status: 'generating',
            },
          })
        } else if (status.phase === 'generating_images') {
          set({
            directorStep: 'generate_images',
            directorImageGenProgress: {
              current: status.progress.current,
              total: status.progress.total,
              currentClipLabel: status.progress.message,
              status: 'generating',
            },
          })
          // Refresh media feed to show new images as they're generated
          get().refreshOutputs()
        } else if (status.phase === 'preparing_video') {
          // H3 prompt-only/direct-reference projects intentionally skip the
          // image review stage and proceed straight to video rendering.
          set({
            directorStep: 'review_video',
            directorImageGenProgress: null,
          })
        } else if (status.phase === 'generating_video') {
          set({ directorStep: 'review_video' })
          // Refresh media feed to show new video clips as they complete
          get().refreshOutputs()
        }

        // Handle LLM streaming
        if (status.llm_streaming) {
          set({ llmStreamDone: false })
        }

        // Handle pause
        if (status.status === 'paused') {
          set({ directorLoading: false })
          if (status.pause_reason === 'review_prompts') {
            set({ directorStep: 'review' })
          } else if (status.pause_reason === 'review_images') {
            set({ directorStep: 'review_video' })
          }
        }

        // Handle completion
        if (status.status === 'completed') {
          set({
            pipelinePolling: false,
            directorLoading: false,
            directorStep: 'review_video',
          })
          get().loadOutputs()
          return  // Stop polling
        }

        // Handle failure
        if (status.status === 'failed' || status.status === 'cancelled') {
          set({
            pipelinePolling: false,
            directorLoading: false,
            directorError: status.error || 'Pipeline stopped',
          })
          return  // Stop polling
        }

      } catch (e) {
        console.error('Pipeline poll error:', e)
      }

      // Continue polling
      if (get().pipelinePolling) {
        setTimeout(poll, 2000)
      }
    }

    setTimeout(poll, 1000)
  },
}))
