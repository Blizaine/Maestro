export interface ModelFamily {
  id: string
  label: string
  order: number
}

export interface ModelDef {
  model_type: string
  name: string
  family: string
  architecture: string
  is_i2v: boolean
  is_t2v: boolean
  guidance_max_phases: number
  fps: number
  supports_end_frame?: boolean
  supports_audio?: boolean
  supports_ref_images?: boolean
  is_downloaded?: boolean
  // True when this model is only available with Mature Mode enabled.
  // Backend always returns the entry; UI filters it out when
  // servicesConfig.nsfw_mode is false. When nsfw_mode flips on, the
  // store auto-adds these models to enabledModels so they appear in
  // selectors without the user having to enable each one manually.
  nsfw_only?: boolean
}

export interface Resolution {
  label: string
  value: string
}

export interface GenerateParams {
  prompt: string
  /** ACE-Step "Music Caption" — style/genre/instruments/mood (music mode). */
  alt_prompt?: string
  model_type: string
  resolution: string
  video_length: number
  num_inference_steps: number
  guidance_scale: number
  seed: number
  image_mode: number
  negative_prompt: string
  repeat_generation: number
  activated_loras: string[]
  loras_multipliers: string
  image_start?: string | string[] | null
  image_end?: string | string[] | null
  multi_prompts_gen_type?: number
  sliding_window_size?: number
  sliding_window_overlap?: number
  guidance_phases?: number
  video_prompt_type?: string
  audio_prompt_type?: string
  image_prompt_type?: string
  input_video_strength?: number
  flow_shift?: number
  audio_guide?: string
  audio_scale?: number
  video_guide?: string
  image_refs?: string[]
  frames_positions?: string
  injection_strength?: number
  settings_version?: number
  self_refiner_setting?: number
  stage2_steps?: number
  generation_mode?: string
  per_clip_frames?: number[]
  remove_background_images_ref?: number
  // TTS-specific
  audio_guide2?: string
  duration_seconds?: number
  pause_seconds?: number
  temperature?: number
  custom_settings?: Record<string, unknown>
  // Loose params: backend accepts additional optional fields. Declared
  // explicitly here so TypeScript narrows JSX children correctly (an
  // index signature widens explicit fields to `unknown` in some contexts).
  progressive_pipeline?: boolean
  single_stage_pipeline?: boolean
  // Runs the reference two-stage pipeline (baked-in TenStrip 10Eros V5
  // workflow config) instead of the standard one. Only sent for models
  // whose def declares reference_pipeline support.
  reference_pipeline?: boolean
  progressive_stage1_image_weight?: number
  progressive_stage2_steps?: number
  progressive_stage2_sigma?: number
  progressive_stage3_steps?: number
  progressive_stage3_sigma?: number
  progressive_stage3_image_weight?: number
  stg_scale?: number
  // STG only runs when the backend sees perturbation_switch === 2 with the
  // model-correct perturbation_layers; startGeneration derives the switch
  // from stg_scale and _applyModelDefaults supplies the layers/window.
  perturbation_switch?: number
  perturbation_layers?: number[]
  perturbation_start_perc?: number
  perturbation_end_perc?: number
  cfg_rescale?: number
  use_gradient_estimation?: boolean
  ge_gamma?: number
  ge_alpha?: number
  keyframe_conditioning_mode?: string
  keyframe_inject_mode?: string
  MMAudio_setting?: number
  MMAudio_prompt?: string
  MMAudio_neg_prompt?: string
  // Continue / Blend mode
  video_source?: string
  // TTS post-processing extras
  tts_dynaudnorm?: boolean
  tts_comp_threshold?: number
  tts_comp_attack?: number
  tts_comp_release?: number
  tts_comp_makeup?: number
  tts_voice_count?: number
}

/** OOM (out-of-VRAM) failure metadata. Set on jobs and pipelines that
 *  failed with a CUDA OutOfMemoryError. The OomRecoveryBanner watches
 *  for this on the latest failure and surfaces a "Lower VRAM headroom?"
 *  banner with a one-click permanent-fix button. Backend logic in
 *  app/services/oom_detect.py. */
export interface OomInfo {
  is_oom: true
  /** The vram_safety_coefficient value in effect when the OOM happened. */
  current_coefficient: number
  /** Suggested next-lower coefficient (current - 0.10), or null if
   *  current is already at the 0.50 floor — at that point coefficient
   *  can't help and the user needs a smaller model / lower resolution. */
  suggested_coefficient: number | null
  /** Truncated stringified exception for UI display (≤300 chars). */
  message: string
}

export interface GenerationJob {
  id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  step: number
  totalSteps: number
  phase: string
  message: string
  outputFiles: string[]
  error: string | null
  /** Present only on failed jobs that look like CUDA OOMs (see OomInfo). */
  oomInfo?: OomInfo | null
}

export interface OutputFile {
  name: string
  url: string
  type: 'video' | 'image' | 'audio'
  mode: GenerationMode | null
  /** Edit sub-mode tag from the .meta.json sidecar params (set by the
   *  retake/inpaint/outpaint/restyle/edit_anything endpoints). The gallery's
   *  Edits filter checks this to identify edit-mode outputs regardless of
   *  the parent `mode`, since e.g. outpaint endpoints write mode='video'. */
  edit_sub_mode?: EditSubMode | null
  favorite: boolean
  size: number
  created_at: number
}

export type MediaFilter = 'all' | 'images' | 'videos' | 'audio' | 'avatars' | 'multiclip' | 'favorites'
export type AspectRatio = 'auto' | '16:9' | '9:16' | '1:1' | '4:3' | '3:4'
export type ResolutionPreset = 'auto' | '480p' | '540p' | '720p' | '1080p'
export type GenerationMode = 'image' | 'video' | 'audio' | 'avatar' | 'tools'
export type EditSubMode = 'retake' | 'inpaint' | 'restyle' | 'outpaint' | 'edit_anything'
export type AudioSubMode = 'speech' | 'music' | 'sfx' | 'mixer'

export interface ChoiceConfig {
  selection?: string[]
  choices?: [string, string][]
  labels?: Record<string, string>
  default?: string
  label?: string
  show_label?: boolean
  letters_filter?: string
}

export interface ModelOptions {
  model_type: string
  architecture: string
  guidance_max_phases: number
  lock_guidance_phases: boolean
  sliding_window: boolean
  motion_amplitude: boolean
  flow_shift: boolean
  tea_cache: boolean
  returns_audio: boolean
  any_audio_prompt: boolean
  audio_scale_name: string
  lock_inference_steps: boolean
  lock_guidance_scale: boolean
  no_negative_prompt: boolean
  i2v_class: boolean
  t2v_class: boolean
  image_outputs: boolean
  supports_end_frame: boolean
  guide_preprocessing: ChoiceConfig | null
  guide_custom_choices: ChoiceConfig | null
  image_ref_choices: ChoiceConfig | null
  audio_prompt_type_sources: ChoiceConfig | null
  background_removal_label: string | null
  sample_solvers: [string, string][] | null
  self_refiner: boolean
  self_refiner_max_plans: number
  sliding_window_defaults: Record<string, number> | null
  // LTX-2 Dev pipeline capabilities (guidance controls in Advanced Settings)
  perturbation?: boolean
  reference_pipeline?: boolean
  cfg_star?: boolean
  adaptive_projected_guidance?: boolean
  audio_guidance?: boolean
  fps: number
  frames_minimum: number
  frames_steps: number
  default_num_inference_steps: number | null
  default_guidance_scale: number | null
  hide_resolution_presets: boolean
  input_video_strength_label: string
  vae_upsampler_modes: number[]
  // TTS-specific
  audio_only: boolean
  duration_slider: { label: string; min: number; max: number; increment: number; default: number } | null
  pause_between_sentences: boolean
  temperature_enabled: boolean
  custom_settings_def: { id: string; label: string; name: string; type: string }[] | null
}

export interface SystemConfig {
  attention_mode: string
  transformer_quantization: string
  vae_config: number
  compile: string
  video_profile: number
  image_profile: number
  audio_profile: number
  video_output_codec: string
  image_output_codec: string
  enhancer_enabled: number
  prompt_enhancer_quantization: string
  attention_modes_available: string[]
  vram_safety_coefficient: number
  // Linked model folders (absolute paths outside the Maestro install,
  // e.g. an existing Wan2GP install's ckpts). Searched read-only for
  // already-downloaded checkpoints; new downloads always go to Maestro's
  // own ckpts folder.
  model_folders: string[]
}

export interface ModelFolderCandidate {
  app: string
  path: string
  files: number
  folders: number
  size_gb: number
  linked: boolean
}

export interface OutputMetadata {
  source: 'sidecar' | 'embedded' | 'none'
  params: Record<string, unknown> | null
  upload_filenames?: Record<string, string>
  job_id?: string
  generation_time?: number
  created_at?: number
}

export interface MultiClip {
  prompt: string
  startImage: File | null
  startImagePath: string | null
  endImage: File | null
  endImagePath: string | null
  durationFrames?: number
}

export type SettingsTab = 'performance' | 'integrations'

export interface ServicesConfig {
  llm_model_id: string
  llm_device: string
  llm_provider: string
  llm_remote_url: string
  enhance_llm_model_id: string
  enhance_llm_device: string
  google_api_key: string
  google_api_key_set: boolean
  openai_api_key: string
  openai_api_key_set: boolean
  anthropic_api_key: string
  anthropic_api_key_set: boolean
  use_director_v2: boolean
  nsfw_mode: boolean
  nsfw_accepted_at: string | null
  director_prompt_polish: 'off' | 'full_guide' | 'light_guide' | 'third_pass'
  civitai_api_key: string
  civitai_api_key_set: boolean
  voice_reference_enabled: boolean
  ltx_progressive_pipeline: boolean
  /** Master gate for experimental / power-user features. When false
   *  (default), the Services panel hides Director v2 engine, Voice
   *  Reference, external API keys (Google/OpenAI/Anthropic), and the
   *  Studio prompt enhancer config; the Edit mode picker hides
   *  Inpaint and Restyle. Flipping this on surfaces all of them. */
  show_experimental: boolean
  /** Performance auto-tune master switch. When true (default for fresh
   *  installs), Settings → System Performance shows a single auto card
   *  with detected hardware + recommended profile, and the underlying
   *  knobs collapse under "Show advanced settings". When false (set
   *  automatically on migration for pre-existing installs), the
   *  advanced fields are visible by default and the user is in
   *  manual mode. Editing any field while auto is on flips this off. */
  auto_performance: boolean
  /** Multi-shot LoRA mode. When true, Pass 2 emits storyboard-format
   *  video_prompts for 20s shots, letting an IC-LoRA (e.g. Maque AI
   *  LTX-2.3 IC-LoRA) cut between camera angles inside a single
   *  generation. Short reaction shots (≤15s) and long sustained
   *  shots (≥40s) keep the regular single-camera flowing format.
   *  User must also have the matching LoRA in their video_loras
   *  selection for the cuts to actually render. */
  director_multishot_lora_mode: boolean
  /** FlashVSR (DiT super-resolution) spatial-upsampling settings.
   *  flashvsr_mode: 1=tiny, 2=full, 3=tiny-long. topk_ratio 0..4 (sparse-attn
   *  density). backend: 'auto' | 'triton_sparse' | 'sparge'. */
  flashvsr_mode: number
  flashvsr_topk_ratio: number
  flashvsr_backend: string
}

// Performance Auto-Tune (Settings → System Performance card) — backed
// by GET /api/v1/system-detect and POST /api/v1/system-detect/apply.
// The card shows the user's detected hardware + the recommended
// profile in plain English; the apply endpoint writes the
// recommendation into wgp_config.json.

/** Hardware detection result from /api/v1/system-detect. Mirrors the
 *  schema documented in app/services/hardware_detect.py — keep in sync
 *  if you add new probe fields there. */
export interface HardwareInfo {
  cuda_available: boolean
  gpu_name: string
  gpu_vram_gb: number
  gpu_capability: string  // e.g. "sm89", "sm120", or "" if no CUDA
  ram_gb: number
  cpu_count: number
  ram_tier: 'high' | 'low' | 'very_low'
  vram_tier: 'high' | 'low' | 'tight' | 'none'
  supports_fp8: boolean
  supports_nvfp4: boolean
  supports_sage: boolean
  supports_sage2: boolean
  supports_flash: boolean
  supports_triton: boolean
}

/** Recommended settings the auto-tune engine produced for the detected
 *  hardware. Underscore-prefixed fields are display-only metadata —
 *  the rest are config keys that get written to wgp_config.json. */
export interface RecommendedSettings {
  video_profile: number
  image_profile: number
  audio_profile: number
  transformer_quantization: 'int8' | 'fp8' | 'bf16'
  vae_config: number
  vram_safety_coefficient: number
  attention_mode: string
  compile: string
  /** Friendly label for the auto card, e.g. "Profile 1 — Optimized for fastest generation" */
  _recommendation_label: string
  /** Verbose reason string for tooltips and debug logs */
  _recommendation_reason: string
}

/** Response shape from GET /api/v1/system-detect. */
export interface SystemDetectResponse {
  hardware: HardwareInfo
  recommended: RecommendedSettings
  auto_enabled: boolean
}

/** Response shape from POST /api/v1/system-detect/apply. */
export interface SystemDetectApplyResponse {
  status: string
  hardware: HardwareInfo
  applied: Record<string, unknown>
  label: string
  reason: string
  /** True when one of the *_profile keys changed — UI should show
   *  "changes take effect on next model load" toast. */
  profile_changed: boolean
}

// CivitAI Browser types
export interface CivitAIModel {
  id: number
  name: string
  description?: string
  type: string
  nsfw: boolean
  tags: string[]
  creator: { username: string; image: string | null }
  stats: { downloadCount: number; favoriteCount: number; thumbsUpCount: number; rating: number; ratingCount: number }
  modelVersions: CivitAIModelVersion[]
}

export interface CivitAIModelVersion {
  id: number
  name: string
  baseModel: string
  trainedWords: string[]
  files: CivitAIFile[]
  images: CivitAIImage[]
  description?: string
  localArch?: string | null
}

export interface CivitAIFile {
  id: number
  name: string
  sizeKB: number
  type: string
  downloadUrl: string
  metadata: { format?: string; size?: string; fp?: string }
}

export interface CivitAIImage {
  url: string
  type: string
  width: number
  height: number
  nsfwLevel: number
  meta?: { prompt?: string; negativePrompt?: string; steps?: number; cfgScale?: number; sampler?: string }
}

export interface CivitAISearchResult {
  items: CivitAIModel[]
  metadata: { nextCursor?: string; totalItems?: number }
}

export interface CivitAIDownload {
  id: string
  filename: string
  status: 'downloading' | 'completed' | 'failed'
  progress: number
  bytes_downloaded: number
  bytes_total: number
  error: string | null
  // Non-fatal warnings raised after the download finished — most
  // commonly the architecture-mismatch warning when a Klein-4B-trained
  // LoRA lands in flux2_klein_9b/ or vice versa. UI shows these inline
  // on the download row.
  warnings?: string[]
}

export interface LoraWeightPhase {
  phase: number
  default: number
  min: number
  max: number
  label: string
}

export interface LoraRecommendedWeights {
  source?: 'civitai' | 'default'
  default: number
  min: number
  max: number
  phases?: LoraWeightPhase[]
}

export interface LoraInfo {
  filename: string
  trained_words: string[]
  preview_url: string | null
  civitai_model_id: number | null
  recommended_weights: LoraRecommendedWeights | null
  has_guide: boolean
  guide?: string | null
  /** NSFW flag from the .civitai.json sidecar (or inferred from filename/tags).
   *  Used to filter out adult-content LoRAs from the Advanced Settings list
   *  unless the user explicitly opts in. */
  nsfw?: boolean
  /** True when the user has manually overridden the NSFW classification via
   *  /api/v1/loras/nsfw-override. The UI surfaces this so the user can tell
   *  at a glance which LoRAs they've corrected vs which are using CivitAI's
   *  raw flag. */
  nsfw_overridden?: boolean
  /** Stable identifier that survives version updates.
   *  Format: `civitai:{modelId}` when sidecar has a CivitAI modelId,
   *  otherwise `local:{filename}`. Use this as the persistence key for
   *  activations, weights, and other LoRA-keyed state instead of the
   *  filename, so updating a LoRA from v1.2 → v1.5 carries settings forward. */
  lora_id: string
  /** Update status from the cached CivitAI manifest. Populated by
   *  /api/v1/loras/check-updates and surfaced through this endpoint
   *  without an extra round-trip. The UI uses this to render badges. */
  update_status?: LoraUpdateStatus
  latest_version_id?: number | null
  current_version_id?: number | null
  latest_published_at?: string | null
  latest_changelog?: string | null
}

/** Per-LoRA update state surfaced from the cached manifest.
 *  - `current`:   sidecar version matches CivitAI's latest
 *  - `available`: a newer version exists on CivitAI
 *  - `unknown`:   not yet checked, no sidecar, or transient API failure
 *  - `local`:     no CivitAI sidecar at all (hand-installed / personal LoRA)
 *  - `removed`:   CivitAI returned 404 (creator unpublished or deleted) */
export type LoraUpdateStatus = 'current' | 'available' | 'unknown' | 'local' | 'removed'

export interface LlmStatus {
  loaded: boolean
  model_id: string | null
  device: string | null
  provider: string
}

/** Live hardware telemetry for the sidebar status indicators.
 *  Backs HardwareStatusBar; polled ~2s via GET /api/v1/system-stats. */
export interface SystemStats {
  cpu: { percent: number }
  ram: { percent: number; used_gb: number; total_gb: number }
  gpu: {
    available: boolean
    /** Headline GPU utilization. On Windows this is the 3D-engine perf
     *  counter (matches Task Manager); elsewhere the NVML/nvidia-smi value. */
    percent: number
    /** NVML / nvidia-smi compute utilization, kept for the tooltip. */
    compute_percent?: number
    vram_used_gb: number
    vram_total_gb: number
    vram_percent: number
  }
  /** Generation model currently resident in VRAM (WGP/mmgp). `loaded`
   *  distinguishes "actually in memory now" from "last/selected type". */
  model: { name: string | null; model_type: string | null; loaded: boolean }
}

export interface LlmModelOption {
  id: string
  label: string
  size_hint: string
}

export interface AudioBeat {
  time: number
  strength: number
}

export interface AudioSection {
  start: number
  end: number
  label: string
  energy: number
}

export interface LyricSegment {
  start: number
  end: number
  text: string
  speaker?: string | null
}

export interface SongStructureEntry {
  label: string
  display_label: string
  start: number
}

export interface AudioAnalysisResult {
  duration: number
  sample_rate: number
  bpm: number
  beats: AudioBeat[]
  downbeats: number[]
  sections: AudioSection[]
  onset_envelope: number[]
  lyrics: LyricSegment[] | null
  vocals_path: string | null
  song_structure?: SongStructureEntry[] | null
}

export interface SuggestedClip {
  start: number
  end: number
  section_label: string
  energy: number
  suggested_prompt_hint: string
}

export interface PlannedClip extends SuggestedClip {
  beat_count: number
  duration_frames: number
  dominant_speaker?: string | null
}

export interface SpeakerMapping {
  speakerId: string
  name: string
  role: 'rapping' | 'singing' | 'speaking' | ''
}

export interface ClipPlan {
  video_prompt: string
  image_prompt: string
}

/** Partial plan returned from single-phase LLM calls */
export interface PartialClipPlan {
  video_prompt?: string
  image_prompt?: string
}

export interface DirectorClipImage {
  clipIndex: number
  prompt: string
  file: File
  filename: string
}

export interface DirectorImageGenProgress {
  current: number
  total: number
  currentClipLabel: string
  status: 'generating' | 'polling' | 'downloading' | 'done' | 'error'
}

export type DirectorSkill = 'music_video' | 'short_film' | 'podcast' | 'viral_video'
export type ShortFilmPath = 'audio' | 'story'

export interface ShortFilmCharacter {
  name: string
  description: string
}

export interface ShortFilmScene {
  scene_number: number
  title: string
  start: number
  end: number
  duration_frames: number
  characters: string[]
  dialogue: string[]
  action: string
  mood: string
}

// ── Director v2 Schema Types ──────────────────────────────────────────

export interface DirectorFlags {
  use_shared_shot_schema?: boolean
  use_mode_specific_renderers?: boolean
  use_prompt_validation?: boolean
  use_prompt_compression?: boolean
  use_llm_refinement?: boolean
  aggressive_compression?: boolean
  log_validation_details?: boolean
  log_compression_deltas?: boolean
}

export interface SubjectRef {
  visual_description: string
  character_id?: string
  position_or_relation?: string
}

export interface DialogueBeat {
  spoken_text: string
  speaker_id?: string
  delivery?: string
  physical_cue?: string
  priority?: 'low' | 'medium' | 'high'
}

export interface CameraPlan {
  framing: string
  angle?: string
  movement?: string
  movement_intensity?: 'static' | 'subtle' | 'moderate' | 'dynamic'
  lens_feel?: string
  reframing_notes?: string
}

export interface AudioPlan {
  mode: 'generated_audio' | 'audio_driven' | 'dialogue_driven' | 'music_driven' | 'ambient_only'
  ambience?: string
  effects?: string[]
  vocal_style?: string
  timing_anchor?: 'audio' | 'video' | 'balanced'
  lip_sync_critical?: boolean
}

export interface ShotPlan {
  shot_id: string
  index: number
  duration_sec: number
  skill_type: DirectorSkill
  scene_goal: string
  narrative_role?: string
  scene_type?: string
  source_mode_preference?: 't2v' | 'i2v' | 'a2v' | 'retake' | 'extend'
  image_strategy?: 'reference_edit' | 'reference_inspired' | 'fresh_generation' | 'none'
  continuity_strategy?: 'independent' | 'continuous' | 'extend_previous'
  subjects_on_screen: SubjectRef[]
  spatial_setup: string
  environment: string
  visual_style: string
  lighting: string
  mood: string
  action_beats: string[]
  performance_beats?: string[]
  dialogue_beats?: DialogueBeat[]
  camera_plan: CameraPlan
  audio_plan: AudioPlan
  ending_beat: string
  constraints?: string[]
  continuity_refs?: string[]
  metadata?: Record<string, any>
}

export interface CharacterProfile {
  id: string
  physical_description: string
  display_name?: string
  wardrobe?: string
  voice_description?: string
}

export interface ProductionPlan {
  skill_type: DirectorSkill
  shots: ShotPlan[]
  title?: string
  global_style?: string
  total_duration_sec?: number
  characters?: CharacterProfile[]
  continuity_notes?: string[]
}

export interface DirectorV2PlanResponse {
  clip_plans: Array<{ video_prompt: string; image_prompt: string }>
  production_plan: ProductionPlan
  skill_type: DirectorSkill
}

// ── Director Pipeline Dashboard ──────────────────────────────────────────

export interface PipelineClipState {
  index: number
  planned_clip: PlannedClip | null
  image_prompt: string
  video_prompt: string
  keyframe_prompts: string[]
  window_prompts: string[]
  window_count: number
  image_prompt_pre_polish: string | null
  video_prompt_pre_polish: string | null
  window_prompts_pre_polish: string[] | null
  keyframe_prompts_pre_polish: string[] | null
  start_image_filename: string | null
  keyframe_filenames: string[]
  video_filename: string | null
  tag: 'good' | 'needs_work' | null
  image_gen_time_sec: number | null
  video_gen_time_sec: number | null
}

export interface PipelineLlmPass {
  pass: string
  system_prompt: string
  response_text: string
  thinking_text: string | null
}

export interface PipelineLlmLog {
  provider: string
  model_id: string
  passes?: PipelineLlmPass[]
  system_prompt: string
  response_text: string
  thinking_text: string | null
  planning_time_sec: number
}

export interface SavedPipelineState {
  version: number
  pipeline_id: string
  created_at: number
  completed_at: number | null
  status: string
  pipeline_type: string
  scene_description: string
  reference_image_path: string | null
  auto_mode: boolean
  seamless: boolean
  image_model: string
  video_model: string
  llm_log: PipelineLlmLog | null
  clips: PipelineClipState[]
  output_files: string[]
  total_time_sec: number | null
}

export interface PipelineListItem {
  id: string
  status: string
  pipeline_type: string
  created_at: number
  clip_count: number
  output_count: number
  scene_description: string
  workspace: string
}
