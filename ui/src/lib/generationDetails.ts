import type { GenerationDetails } from '../types'

// Pre-queue errors have no server job yet. Keep a small snapshot of the exact
// submission, so changing the form cannot relabel a failed attempt later.
const displayKeys = new Set([
  'model_type', 'prompt', 'negative_prompt', 'generation_mode', '_studio_video_workflow',
  'resolution', 'video_length', 'force_fps', 'num_inference_steps', 'guidance_scale',
  'guidance2_scale', 'seed', 'sample_solver', 'flow_shift', 'audio_flow_shift',
  'audio_guidance_scale', 'override_attention', 'loras_multipliers',
  'minimax_h3_turbo_mode', 'minimax_h3_turbo_preset', 'minimax_h3_text_encoder',
  'minimax_h3_experimental_loras',
  'minimax_h3_video_vae', 'minimax_h3_multi_window', 'sliding_window_size',
  'sliding_window_overlap', 'denoising_strength', 'skip_steps_cache_type',
  'skip_steps_multiplier', 'skip_steps_start_step_perc', 'spatial_upsampling',
  'temporal_upsampling', 'repeat_generation',
])

function loraFilename(value: string): string {
  let path = value
  if (value.includes('://')) {
    try { path = new URL(value).pathname } catch { path = value.split(/[?#]/)[0] }
  }
  const name = path.replace(/\\/g, '/').split('/').pop() || path
  try { return decodeURIComponent(name) } catch { return name }
}

export function captureGenerationDetails(params: Record<string, unknown>, workspace?: string): GenerationDetails {
  const visible = Object.fromEntries(Object.entries(params).filter(([key, value]) => (
    displayKeys.has(key) && (typeof value === 'string' || typeof value === 'boolean'
      || (typeof value === 'number' && Number.isFinite(value)))
  )))
  if (Array.isArray(params.activated_loras)) {
    visible.activated_loras = params.activated_loras.filter((value): value is string => typeof value === 'string').map(loraFilename)
  }
  return { params: visible, ...(workspace ? { workspace } : {}) }
}

export const generationSettingLabels: Record<string, string> = {
  generation_mode: 'Media type', _studio_video_workflow: 'Workflow', resolution: 'Resolution',
  video_length: 'Requested frames', force_fps: 'Frame rate', num_inference_steps: 'Steps',
  guidance_scale: 'Guidance', guidance2_scale: 'Guidance 2', seed: 'Seed',
  sample_solver: 'Sampler', flow_shift: 'Video shift', audio_flow_shift: 'Audio shift',
  audio_guidance_scale: 'Audio guidance', override_attention: 'Attention override',
  minimax_h3_turbo_mode: 'Managed Turbo', minimax_h3_turbo_preset: 'Managed Turbo preset',
  minimax_h3_experimental_loras: 'Experimental H3 LoRAs',
  minimax_h3_text_encoder: 'Text encoder', minimax_h3_video_vae: 'Video VAE',
  minimax_h3_multi_window: 'Multiple windows', sliding_window_size: 'Window frames',
  sliding_window_overlap: 'Window overlap', denoising_strength: 'Denoising strength',
  skip_steps_cache_type: 'Step cache', skip_steps_multiplier: 'Cache threshold',
  skip_steps_start_step_perc: 'Cache start (%)', spatial_upsampling: 'Spatial upscaling',
  temporal_upsampling: 'Temporal upscaling', repeat_generation: 'Repeats',
  applied: 'Auto-tune applied', profile: 'Memory profile', transformer_budget_mb: 'Transformer budget (MB)',
  attention_head_split: 'Attention head split', read_ahead: 'Read ahead',
  reserved_ram_fraction: 'Reserved RAM fraction', source: 'Plan source',
}

export function generationSettingValue(value: unknown): string {
  if (value === true) return 'On'
  if (value === false) return 'Off'
  if (value === '') return 'Default / disabled'
  return String(value)
}
