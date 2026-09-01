export interface ModelFamily {
  id: string
  label: string
  order: number
}

export type DirectorPipelineType = 'music_video' | 'short_film_audio' | 'short_film_story'
export type DirectorShotImageGuidance = 'auto' | 'prompt_only' | 'generate'
export type DirectorShotImagePolicy = 'generate' | 'prompt_only' | 'direct_references'

export interface DirectorCapabilityResult {
  compatible: boolean
  reason: string
}

export interface DirectorModelCompatibility {
  image: DirectorCapabilityResult
  video: Record<DirectorPipelineType | 'seamless', DirectorCapabilityResult>
  supports_audio_input: boolean
  generates_audio: boolean
  supports_voice_reference: boolean
  voice_reference_mode?: 'none' | 'id_lora' | 'native_reference'
  video_strategy?: 'rolling_window' | 'bounded_start_end' | 'omni_reference'
  audio_input_mode?: 'none' | 'generic_audio_guide' | 'reference_manifest'
  reference_mode?: 'none' | 'start_frame' | 'start_end' | 'omni_manifest'
  shot_image_support?: 'required' | 'optional' | 'direct_references'
  supports_endpoint_continuity?: boolean
  clip_min_frames?: number | null
  clip_max_frames?: number | null
  clip_frame_step?: number | null
  max_image_refs: number | null
}

export interface ModelDef {
  model_type: string
  name: string
  description?: string
  selector_help?: string
  lora_compatibility_note?: string
  family: string
  architecture: string
  is_i2v: boolean
  is_t2v: boolean
  guidance_max_phases: number
  fps: number
  supports_end_frame?: boolean
  /** Legacy broad flag: accepts input audio OR generates output audio. */
  supports_audio?: boolean
  supports_audio_input?: boolean
  generates_audio?: boolean
  supports_ref_images?: boolean
  /** Native MiniMax H3 Ref2VA/Omni reference workflow. */
  omni_reference?: boolean
  /** Image-suite capability flags published by the model definition. */
  supports_image_edit?: boolean
  requires_image_reference?: boolean
  supports_image_inpaint?: boolean
  supports_image_outpaint?: boolean
  director?: DirectorModelCompatibility
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
  /** Frames trimmed from the tail of each rolling window before stitching. */
  sliding_window_discard_last_frames?: number
  /** Explicitly honor a manually locked window above the model's VRAM-aware recommendation. */
  sliding_window_memory_override?: boolean
  /** Optional model-specific transformer step cache. */
  skip_steps_cache_type?: '' | 'first_block'
  /** First Block Cache residual-change threshold. */
  skip_steps_multiplier?: number
  /** Percentage of denoising steps to run before caching may begin. */
  skip_steps_start_step_perc?: number
  /** Per-generation attention override. Sol is H3-only and experimental. */
  override_attention?: '' | 'sol'
  guidance_phases?: number
  video_prompt_type?: string
  audio_prompt_type?: string
  image_prompt_type?: string
  input_video_strength?: number
  flow_shift?: number
  audio_guide?: string
  audio_scale?: number
  video_guide?: string
  video_mask?: string
  /** Still-image control inputs used by Image Edit/Inpaint/Outpaint. */
  image_guide?: string
  image_mask?: string
  /** Top, bottom, left, right expansion percentages. */
  video_guide_outpainting?: string
  denoising_strength?: number
  masking_strength?: number
  /** Maestro's friendly UI state for MiniMax H3 FL2VA control-video editing. */
  minimax_h3_control_visual_mode?: 'prompt' | 'whole' | 'inside' | 'outside'
  image_refs?: string[]
  frames_positions?: string
  injection_strength?: number
  settings_version?: number
  self_refiner_setting?: number
  stage2_steps?: number
  generation_mode?: string
  per_clip_frames?: number[]
  remove_background_images_ref?: number
  /** UI-only workflow marker retained in output sidecars. */
  _studio_image_workflow?: StudioImageWorkflow
  /** UI-only Video workflow marker retained in output sidecars. */
  _studio_video_workflow?: StudioVideoWorkflow
  /** UI-only long-form selector retained for reload/sidecar fidelity. */
  _duration_planning_mode?: 'duration' | 'windows' | 'auto'
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
  // MiniMax H3 Ref2VA ordered Omni-reference manifest.
  minimax_h3_references?: MiniMaxH3Reference[]
  minimax_h3_reference_detail?: 'match' | 'max'
  /** Experimental long-form Omni orchestration using native continuation or hard-cut clips. */
  minimax_h3_reference_sequence?: boolean
  /** Enable native multi-window continuation for H3 First / Last. Defaults on. */
  minimax_h3_multi_window?: boolean
  /** Choose faithful AI planning, creative AI writing, or one exact prompt per pass. */
  minimax_h3_sequence_prompt_mode?: WindowPromptMode
  /** Native Ref2VA clip ceiling selected by Auto or the Advanced override. */
  minimax_h3_sequence_clip_frames?: number
  /** Honor the user's locked Omni clip length above Auto's recommendation. */
  minimax_h3_sequence_memory_override?: boolean
  /** Add bounded generated look/blocking references between sequence clips. */
  minimax_h3_sequence_continuity?: boolean
  minimax_h3_text_encoder?: 'nvfp4_awq' | 'gguf_q2_k' | 'gguf_q4_k_m' | 'int8' | 'bf16'
  /** LTX-2.5 video decoder. Fast ConvVAE is recommended; NAD is experimental. */
  ltx25_video_vae?: 'fast' | 'nad'
  /** One-click managed H3 Turbo recipe for Full or Pruned H3. */
  minimax_h3_turbo_mode?: boolean
  /** Immutable validated/candidate Turbo checkpoint selected by its manifest id. */
  minimax_h3_turbo_preset?: string
  /** Automatically expand one long H3 concept into window-local prompts. */
  minimax_h3_window_storyboard?: boolean
  /** H3 First/Last edit grammar used by the automatic window planner. */
  minimax_h3_camera_coverage?: 'auto' | 'continuous' | 'multi_shot'
  /** Compiled Context-IR prompts, one per continuation pass. */
  h3_window_prompts?: string[]
  h3_window_plan_signature?: string
  h3_window_plan?: H3WindowPlan
  /** Plain user concept retained when a one-clip H3 prompt is enhanced. */
  _h3_original_prompt?: string
  /** Explicitly enable rolling long-form generation for the LTX family. */
  ltx_multi_window?: boolean
  /** Expand one idea faithfully/creatively, or consume one exact line per window. */
  ltx_window_prompt_mode?: WindowPromptMode
  /** Compiled, single-line prompts consumed by WanGP's native window router. */
  ltx_window_prompts?: string[]
  /** Original overall idea retained while the compiled prompts are visible. */
  _ltx_original_prompt?: string
}

export type WindowPromptMode = 'auto' | 'creative' | 'manual'
export type WindowPlanningStyle = 'faithful' | 'creative'

export interface LTXWindowPlan {
  source_prompt: string
  window_count: number
  window_prompts: string[]
  planned_by: 'llm' | 'manual' | 'reviewed' | 'deterministic_fallback' | string
  planning_style?: WindowPlanningStyle
  planning_error?: string | null
}

export type MiniMaxH3ReferenceType = 'image' | 'video' | 'audio'
export type MiniMaxH3AudioIntent = 'voice' | 'drive' | 'style'

export interface MiniMaxH3Reference {
  id: string
  type: MiniMaxH3ReferenceType
  path: string
  filename: string
  url?: string
  role?: string
  audio_intent?: MiniMaxH3AudioIntent
  image_intent?: 'identity' | 'scene' | 'style' | 'composition'
  remove_background?: boolean
  video_intent?: 'character' | 'motion' | 'scene'
  library_character_id?: string
  character_name?: string
  include_audio?: boolean
  has_audio?: boolean
  audio_path?: string
  audio_filename?: string
  audio_duration_seconds?: number | null
  duration_seconds?: number | null
  source_duration_seconds?: number | null
  effective_duration_seconds?: number | null
}

export interface SavedOmniCharacterMedia {
  type?: 'image' | 'video' | 'audio'
  path: string
  filename: string
  url: string
  duration_seconds?: number | null
  has_audio?: boolean
}

export interface SavedOmniCharacter {
  id: string
  name: string
  created_at: number
  updated_at: number
  visual: SavedOmniCharacterMedia & { type: 'image' | 'video' }
  voice?: SavedOmniCharacterMedia | null
}

export interface H3InjectedKeyframe {
  path: string
  position: string
  source_index?: number
  absolute_frame?: number
  global_seconds?: number
  window?: number
  local_frame?: number
  local_seconds?: number
  picture_index?: number
}

export interface H3WindowPlanWindow {
  index: number
  title: string
  start_frame: number
  end_frame: number
  start_seconds: number
  end_seconds: number
  opening_state: string
  closing_state: string
  coverage?: string
  pacing?: string
  shot_count?: number
  injected_keyframes?: H3InjectedKeyframe[]
  prompt: string
  prompt_tokens?: number
  prompt_quality_target?: number
  prompt_compacted?: boolean
}

export interface H3WindowPlan {
  source_prompt: string
  signature: string
  planned_by: 'llm' | 'hybrid_repair' | 'deterministic_fallback' | 'not_needed' | 'manual'
  planning_warnings?: string[]
  planning_diagnostics?: string[]
  planning_notes?: string[]
  planning_style?: WindowPlanningStyle
  plan_kind?: 'sliding_window' | 'reference_sequence'
  camera_coverage?: 'auto' | 'continuous' | 'multi_shot'
  total_frames: number
  window_frames: number
  effective_window_frames?: number
  window_count: number
  per_clip_frames?: number[]
  trim_tail_frames?: number
  overlap_frames?: number
  native_continuation?: boolean
  resolution: string
  model_type: string
  subject_continuity?: string
  setting_continuity?: string
  source_intent?: Record<string, unknown>
  injected_keyframes?: H3InjectedKeyframe[]
  windows: H3WindowPlanWindow[]
  window_prompts: string[]
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
  /** Direct submissions stay visible while the backend is queued/planning. */
  showInGallery?: boolean
  kind?: 'generation' | 'editor_export' | string
  status: 'held' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  step: number
  totalSteps: number
  phase: string
  message: string
  outputFiles: string[]
  error: string | null
  /** Present only on failed jobs that look like CUDA OOMs (see OomInfo). */
  oomInfo?: OomInfo | null
  /** Exact prompts assigned to an in-flight H3 sliding-window generation. */
  h3WindowPlan?: H3WindowPlan | null
  /** Adaptive video ETA fields; omitted for image/audio/tool jobs. */
  currentClip?: number
  totalClips?: number
  currentWindow?: number
  totalWindows?: number
  windowEtaSeconds?: number | null
  clipEtaSeconds?: number | null
  generationEtaSeconds?: number | null
  projectEtaSeconds?: number | null
  windowCompletionAt?: number | null
  clipCompletionAt?: number | null
  generationCompletionAt?: number | null
  projectCompletionAt?: number | null
  etaConfidence?: 'calibrating' | 'low' | 'medium' | 'high'
  etaBasis?: 'waiting-for-first-clip' | 'historical' | 'historical-adaptive' | 'live-adaptive' | 'live-cache-aware'
  etaHistorySamples?: number
  etaHistoryMatch?: 'exact' | 'family' | null
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

export type AppMode = 'director' | 'studio' | 'editor'
export type EditorMediaType = 'video' | 'image' | 'audio'
export type EditorTrackType = 'video' | 'audio' | 'text'
export type EditorTransitionType = 'none' | 'dissolve' | 'fade_black'
export type EditorAIRoundTripTool =
  | 'retake'
  | 'edit_anything'
  | 'recast'
  | 'repaint'
  | 'outpaint'
  | 'upscale'
  | 'film_grain'
  | 'revoice'
export type EditorAIReturnMode = 'replace' | 'alternate'

export interface EditorAsset {
  id: string
  name: string
  type: EditorMediaType
  origin: 'output' | 'upload' | 'project'
  workspace?: string
  favorite?: boolean
  path?: string
  url: string
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  size?: number
  created_at?: number
  /** Runtime-only availability flag populated when an Editor project opens. */
  missing?: boolean
}

export interface EditorTransform {
  x: number
  y: number
  scale: number
  rotation: number
}

export interface EditorTextStyle {
  x: number
  y: number
  font_family?: string
  font_size: number
  color: string
  background_color?: string
  background_opacity?: number
  text_align?: 'left' | 'center' | 'right'
}

export interface EditorMarker {
  id: string
  time: number
  label: string
  color: string
}

export interface EditorAIHistoryEntry {
  id: string
  tool: EditorAIRoundTripTool | 'director_rerun'
  asset_id: string
  created_at: number
}

export interface EditorTakeState {
  source_in: number
  speed: number
}

/** Saved Director provenance for a shot imported into the Editor timeline. */
export interface EditorDirectorClipSource {
  pipeline_id: string
  clip_index: number
  pipeline_type: string
  workspace: string
  video_prompt: string
  window_prompts?: string[]
}

export interface EditorTimelineItem {
  id: string
  asset_id?: string
  name: string
  start: number
  duration: number
  source_in: number
  speed: number
  volume: number
  opacity: number
  fit: 'contain' | 'cover'
  transform: EditorTransform
  muted?: boolean
  disabled?: boolean
  fade_in?: number
  fade_out?: number
  transition_in?: EditorTransitionType
  transition_out?: EditorTransitionType
  /** Items with the same link id move, trim, split, and delete together. */
  link_group_id?: string
  /** Non-destructive source alternatives created manually or by Maestro AI. */
  take_asset_ids?: string[]
  /** Source timing belongs to each take so switching back restores the exact original trim. */
  take_states?: Record<string, EditorTakeState>
  ai_history?: EditorAIHistoryEntry[]
  /** Lets Editor rerun this exact shot with the original Director workflow. */
  director?: EditorDirectorClipSource
  text?: string
  style?: EditorTextStyle
}

export interface EditorTrack {
  id: string
  name: string
  type: EditorTrackType
  z_index: number
  muted: boolean
  locked: boolean
  volume?: number
  items: EditorTimelineItem[]
}

export interface EditorCanvas {
  width: number
  height: number
  fps: number
  background: string
}

export type EditorUpscaleMethod =
  | ''
  | 'flashvsr2'
  | 'flashvsr3'
  | 'flashvsr4'
  | 'flashvsr2pass2'
  | 'flashvsr2pass4'

export interface EditorExportSettings {
  quality: 'draft' | 'balanced' | 'high'
  codec: 'h264' | 'h265'
  encoder: 'auto' | 'software' | 'nvidia' | 'intel' | 'apple'
  include_audio: boolean
  resolution: 'canvas' | '2160p' | '1080p' | '720p' | '480p'
  frame_rate: 'project' | 24 | 30 | 60
  filename: string
  spatial_upsampling: EditorUpscaleMethod
  film_grain_intensity: number
  film_grain_saturation: number
}

export interface EditorExportRecord {
  id: string
  filename: string
  workspace: string
  created_at: number
  duration: number
  width: number
  height: number
  fps: number
  codec: EditorExportSettings['codec']
  quality: EditorExportSettings['quality']
  encoder?: EditorExportSettings['encoder']
  spatial_upsampling?: EditorUpscaleMethod
  film_grain_intensity?: number
  film_grain_saturation?: number
}

export interface EditorProject {
  schema_version: number
  id: string
  name: string
  workspace: string
  created_at: number
  updated_at: number
  canvas: EditorCanvas
  assets: Record<string, EditorAsset>
  tracks: EditorTrack[]
  markers: EditorMarker[]
  export: EditorExportSettings
  exports: EditorExportRecord[]
}

export interface EditorProjectSummary {
  id: string
  name: string
  workspace: string
  created_at: number
  updated_at: number
  duration: number
  asset_count: number
}

export interface EditorMediaProbe {
  name: string
  type: EditorMediaType
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
  audio_channels: number
  audio_sample_rate: number
  size: number
  path: string
}

export interface EditorMediaStatus {
  asset_id: string
  available: boolean
  path?: string
  error?: string
}

export interface EditorMediaPreview {
  preview_id: string
  thumbnail_url?: string
  proxy_url?: string
  waveform: number[]
}

export interface EditorExportCapabilities {
  encoders: {
    software: boolean
    nvidia: boolean
    intel: boolean
    apple: boolean
  }
  recommended: EditorExportSettings['encoder']
}

export type MediaFilter = 'all' | 'images' | 'videos' | 'audio' | 'avatars' | 'multiclip' | 'favorites'
export type AspectRatio = 'auto' | '21:9' | '16:9' | '9:16' | '1:1' | '4:3' | '3:4'
export type ResolutionPreset = 'auto' | '480p' | '540p' | '720p' | '768p' | '1080p'
export type ScailResolutionProfile = '480p' | '512p' | '704p'
/** Backward-compatible name for saved Recast/API callers. */
export type RecastResolutionProfile = ScailResolutionProfile
export type GenerationMode = 'image' | 'video' | 'audio' | 'avatar' | 'tools'
/**
 * User-facing Studio Video workflow. The implementation deliberately keeps
 * the legacy `avatar` and `tools` generation modes underneath so saved jobs,
 * API payloads, and the future timeline editor can reuse the proven engines.
 */
export type StudioVideoWorkflow =
  | 'frames'
  | 'references'
  | 'extend'
  | 'blend'
  | 'retake'
  | 'prompt_edit'
  | 'outpaint'
  | 'repaint'
  | 'recast'
  | 'upscale'
  | 'film_grain'
/**
 * Internal media intent inside Studio Video's Frames/References workflows.
 * Frames derives text, fixed-frame, and audio-drive routing from its inputs;
 * References always resolves to the native H3 Omni route.
 */
export type StudioVideoEffectiveCreateRoute = 'generate' | 'guided' | 'audio' | 'omni'
/** Backward-compatible persisted shape; new sessions always use Auto. */
export type StudioVideoCreateRoute = 'auto' | StudioVideoEffectiveCreateRoute
/** User-facing Studio Image workflow. */
export type StudioImageWorkflow =
  | 'generate'
  | 'inpaint'
  | 'outpaint'
  | 'upscale'
export type EditSubMode = 'retake' | 'inpaint' | 'restyle' | 'outpaint' | 'edit_anything' | 'recast'
export type AudioSubMode = 'speech' | 'music' | 'sfx' | 'mixer' | 'revoice'

export interface RecastReferenceAsset {
  file: File | null
  path: string
  url: string
}

export interface RecastCharacterMapping {
  id: string
  target: string
  refFile: File | null
  refPath: string
  refUrl: string
  additionalRefs: RecastReferenceAsset[]
  referenceAlignedToSource: boolean
}

/** Optional SCAIL-2 Repaint correspondence. The source phrase is tracked
 * through the control video and the target phrase is segmented in the edited
 * first frame; both receive the same stable semantic color. */
export interface RepaintRegionMapping {
  id: string
  source: string
  target: string
}

export interface ChoiceConfig {
  selection?: string[]
  choices?: [string, string][]
  labels?: Record<string, string>
  default?: string
  label?: string
  show_label?: boolean
  letters_filter?: string
}

export interface SlidingWindowMemoryPolicy {
  checkpoint?: 'full' | 'pruned'
  manual_override?: boolean
  /** Legal H3 frame steps reserved for Ref2VA reference-context headroom. */
  reference_margin_steps?: number
  auto_resolution_pixels?: Record<string, number>
  resolution_bands: Array<{
    min_pixels: number
    vram_tiers: Array<{
      max_vram_gb?: number
      frames: number | null
      fallback_resolution?: string
    }>
  }>
}

export interface ModelOptions {
  model_type: string
  architecture: string
  guidance_max_phases: number
  lock_guidance_phases: boolean
  sliding_window: boolean
  video_continuation?: boolean
  motion_amplitude: boolean
  flow_shift: boolean
  tea_cache: boolean
  first_block_cache?: boolean
  sol_attention?: boolean
  sol_attention_status?: {
    installed: boolean
    supported: boolean
    reason?: string | null
    capability?: string
    triton_version?: string | null
    minimum_triton?: string
    first_run_compiles_kernels?: boolean
  } | null
  skip_steps_multiplier_choices?: [string, number][] | null
  skip_steps_multiplier_label?: string
  default_skip_steps_multiplier?: number
  default_skip_steps_start_step_perc?: number
  returns_audio: boolean
  any_audio_prompt: boolean
  audio_scale_name: string
  /** Repair a standalone uploaded soundtrack whose hidden source mode was lost. */
  infer_audio_prompt_from_guide?: boolean
  lock_inference_steps: boolean
  lock_guidance_scale: boolean
  no_negative_prompt: boolean
  i2v_class: boolean
  t2v_class: boolean
  image_outputs: boolean
  inpaint_support?: boolean
  outpaint_support?: boolean
  inpaint_video_prompt_type?: string
  image_video_prompt_type?: string
  supports_end_frame: boolean
  /** Model accepts additional pictures pinned to exact target-frame positions. */
  custom_frames_injection?: boolean
  omni_reference?: boolean
  omni_reference_limits?: {
    image: number
    video: number
    audio: number
    total: number
  } | null
  omni_reference_detail_choices?: [string, 'match' | 'max'][] | null
  omni_reference_detail_default?: 'match' | 'max'
  minimax_h3_text_encoder_choices?: {
    value: string
    label: string
    size_hint: string
    recommended?: boolean
  }[] | null
  minimax_h3_text_encoder_default?: string
  ltx25_video_vae_choices?: {
    value: 'fast' | 'nad'
    label: string
    description: string
    experimental?: boolean
  }[] | null
  ltx25_video_vae_default?: 'fast' | 'nad'
  minimax_h3_turbo?: {
    filename: string
    label: string
    experimental: boolean
    preset_id: string
    version_label: string
    steps: number
    weight: number
    presets: Array<{
      id: string
      label: string
      status: 'validated' | 'candidate' | 'legacy' | string
      filename: string
      steps: number
      weight: number
      weight_min: number
      weight_max: number
      description: string
      revision: string
      workflow?: 'all' | 'fl2va' | 'ref2va'
      runtime?: 'standard_lora' | 'pdd'
      full_checkpoint_only?: boolean
    }>
    upstream_url: string
    guide: string
  } | null
  minimax_h3_runtime_advisory?: {
    level: 'warning' | 'info'
    title: string
    message: string
    reasons: Array<{
      code: 'triton_unavailable' | 'system_ram_low' | string
      message: string
    }>
    recommended_model_type?: string
    recommended_turbo?: boolean
    estimated_pipeline_ram_gb?: number
    minimum_system_ram_gb?: number
    detected_ram_gb?: number | null
    supports_triton?: boolean | null
    blocking: boolean
  } | null
  minimax_h3_media_sources?: boolean
  /** FL2VA can edit a source video globally or through a white selection mask. */
  video_to_video_inpaint?: boolean
  resolution_presets?: Partial<Record<ResolutionPreset, {
    label: string
    experimental?: boolean
    hint?: string
    values: Partial<Record<AspectRatio, string>>
  }>> | null
  resolution_preset_order?: ResolutionPreset[] | null
  supports_auto_aspect?: boolean
  guide_preprocessing: ChoiceConfig | null
  guide_custom_choices: ChoiceConfig | null
  mask_preprocessing?: ChoiceConfig | null
  image_ref_choices: ChoiceConfig | null
  audio_prompt_type_sources: ChoiceConfig | null
  background_removal_label: string | null
  max_image_refs?: number | null
  sample_solvers: [string, string][] | null
  self_refiner: boolean
  self_refiner_max_plans: number
  sliding_window_defaults: Record<string, number> | null
  sliding_window_auto_prompt_pacing?: boolean
  /** Shows Maestro's explicit single-pass / long-form sequence controls. */
  multi_window_sequence_controls?: boolean
  /** Native end image belongs to the last continuation window. */
  sliding_window_end_image_at_final?: boolean
  /** Carries generated video and stereo-audio history between native passes. */
  sliding_window_audio_history?: boolean
  sliding_window_memory_policy?: SlidingWindowMemoryPolicy | null
  /** Native per-window policy for H3 Omni reference sequences. */
  omni_sequence_memory_policy?: SlidingWindowMemoryPolicy | null
  /** Native one-pass policy used by Director. Omni publishes this without
   * exposing Studio sliding-window controls. */
  director_memory_policy?: SlidingWindowMemoryPolicy | null
  // LTX-2 Dev pipeline capabilities (guidance controls in Advanced Settings)
  perturbation?: boolean
  reference_pipeline?: boolean
  cfg_star?: boolean
  adaptive_projected_guidance?: boolean
  audio_guidance?: boolean
  fps: number
  frames_minimum: number
  frames_steps: number
  frames_maximum?: number | null
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
  music3_structured_caption?: boolean
  music_caption_label?: string
  music_caption_help?: string
  music_lyrics_help?: string
}

export interface SystemConfig {
  // Maestro release version (repo-root VERSION file), shown next to the
  // app title. Optional: older backends don't send it.
  app_version?: string
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
  /** Plays once on the computer hosting Maestro, independent of browser
   * notification permissions and per-browser preferences. */
  host_notification_sound_enabled: boolean
  host_notification_sound_volume: number
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

export interface MultiWindowTiming {
  window_count: number
  completed_windows?: number
  scene_duration_seconds?: number | null
  window_generation_seconds: number[]
  total_generation_seconds: number
}

export interface OutputMetadata {
  source: 'sidecar' | 'embedded' | 'none'
  params: Record<string, unknown> | null
  /** Standalone Studio post-processing outputs are restored through their
   *  workflow panels instead of being treated as generation models. */
  tool?: 'upscale' | 'film_grain' | 'revoice'
  tool_media_type?: 'image' | 'video'
  tool_source?: string
  upload_filenames?: Record<string, string>
  job_id?: string
  /** Director revision that produced this artifact. Gallery "Load settings"
   *  uses it to reopen the complete Director project instead of flattening
   *  the clip into an unrelated Studio job. */
  director_pipeline_id?: string
  director_clip_index?: number
  generation_time?: number
  generation_time_basis?: 'active' | 'elapsed'
  /** Exact native-window render timings captured after each successfully
   *  saved window. Available for new multi-window generations. */
  multi_window_timing?: MultiWindowTiming
  job_elapsed_time?: number
  created_at?: number
}

export interface WebPushStatus {
  supported: boolean
  public_key: string
  subscription_count: number
  reason: string | null
}

export interface WebPushMutationResult {
  subscribed?: boolean
  unsubscribed?: boolean
  subscription_count: number
}

export interface TailscaleRemoteAccessStatus {
  installed: boolean
  connected: boolean
  backend_state: string
  dns_name: string | null
  https_url: string | null
  configured: boolean
  enabled: boolean
  target_port: number
  install_url: string
  platform: string
  needs_login: boolean
  error: string | null
}

export interface MultiClip {
  prompt: string
  startImage: File | null
  startImagePath: string | null
  endImage: File | null
  endImagePath: string | null
  durationFrames?: number
}

export type SettingsTab = 'performance' | 'integrations' | 'notifications'

export interface ServicesConfig {
  llm_model_id: string
  llm_device: string
  llm_provider: string
  llm_remote_url: string
  llm_remote_api_key: string
  llm_remote_api_key_set: boolean
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
   *  external API keys (Google/OpenAI/Anthropic), and the Studio prompt
   *  enhancer config; the Edit mode picker hides Inpaint. LTX Voice
   *  Reference is a separate opt-in under Video Frames → Advanced. */
  show_experimental: boolean
  /** Storage Manager opt-in: allow removing duplicate files FROM linked
   *  installs (Recycle Bin only). Default off — informed consent. */
  storage_allow_linked_removal?: boolean
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
  /** Version release date from CivitAI — persisted into the download
   *  sidecar so My LoRAs can sort by newest release. */
  publishedAt?: string
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
  /** Unix timestamps (seconds) supplied by the download registry. */
  started_at: number | null
  completed_at: number | null
  /** Present after a downloaded checkpoint is registered as a model. */
  model_type?: string | null
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
  /** Managed choices may be listed before their first-use download. */
  managed?: boolean
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
  /** ISO timestamp of when the file was downloaded — sidecar `downloadedAt`
   *  when present, else the weight file's mtime. Shown as an age chip in
   *  the Studio/Director LoRA pickers. */
  downloaded_at?: string | null
  /** ISO timestamp of the CivitAI version's publish date (sidecar
   *  `publishedAt`). Null for HF/hand-installed LoRAs without sidecar data. */
  released_at?: string | null
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
  window_prompts?: string[]
  keyframe_prompts?: string[]
  window_count?: number
  visual_changes?: unknown[]
  image_source?: string
  /** Model-aware Director contracts are persisted with the reviewed prompt. */
  [key: `_director_${string}`]: unknown
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
  metadata?: Record<string, unknown>
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
  video_stale?: boolean
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

export type PipelineRepairStatus =
  | 'queued'
  | 'running'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export interface PipelineRepairState {
  operation_id: string
  status: PipelineRepairStatus
  phase: 'queued' | 'images' | 'videos' | 'rejoin' | 'completed' | 'failed' | 'cancelled' | 'interrupted'
  current: number
  total: number
  clip_index: number | null
  message: string
  error: string | null
  cancel_requested?: boolean
  started_at: number
  updated_at: number
  completed_at: number | null
  result_filename: string | null
}

export interface SavedPipelineState {
  version: number
  pipeline_id: string
  /** Stable project lineage. Every rerun is a new immutable pipeline revision. */
  project_id?: string
  parent_pipeline_id?: string | null
  queue_entry_id?: string | null
  created_at: number
  completed_at: number | null
  status: string
  pipeline_type: string
  workspace?: string
  scene_description: string
  reference_image_path: string | null
  generated_reference_image_filename?: string | null
  character_ref_paths?: string[]
  location_ref_paths?: string[]
  auto_mode: boolean
  seamless: boolean
  image_model: string
  video_model: string
  image_loras?: Record<string, unknown>
  video_loras?: Record<string, unknown>
  image_params?: Record<string, unknown>
  video_params?: Record<string, unknown>
  director_resolution_preset?: ResolutionPreset
  director_aspect_ratio?: AspectRatio
  director_ui_snapshot?: Record<string, unknown>
  asset_manifest?: Record<string, unknown>
  _params_snapshot?: Record<string, unknown>
  /** Effective saved behavior. Missing on legacy projects, which require images. */
  shot_image_policy?: DirectorShotImagePolicy
  shot_image_guidance?: DirectorShotImageGuidance
  llm_log: PipelineLlmLog | null
  clips: PipelineClipState[]
  output_files: string[]
  total_time_sec: number | null
  repair?: PipelineRepairState | null
}

export type DirectorQueueEntryStatus =
  | 'held'
  | 'queued'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface DirectorQueueEntry {
  id: string
  status: DirectorQueueEntryStatus
  message: string
  created_at: number
  started_at?: number | null
  completed_at?: number | null
  pipeline_id?: string | null
  error?: string | null
  scene_description: string
  pipeline_type: string
  image_model: string
  video_model: string
}

export interface DirectorQueueEntryDetail extends DirectorQueueEntry {
  params: Record<string, unknown>
}

export interface DirectorQueueState {
  version: number
  paused: boolean
  running: boolean
  entries: DirectorQueueEntry[]
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
  thumbnail_url?: string | null
  repair_status?: PipelineRepairStatus | null
}
