import policy from '../../../app/shared/h3_performance_policy.json'
import type { ModelOptions } from '../types'

export type H3PerformanceAdviceInput = {
  options: ModelOptions | null
  resolution: string
  frames: number
  totalVramGb: number
  totalRamGb: number
  steps: number
  attention: string
  loraCount?: number
  referenceCount?: number
  videoGuide?: boolean
  continuationFrames?: number
}

export type H3PerformanceAdvice = {
  split: number | null
  label: string
  targetRows: number | null
  slow: boolean
  experimental: boolean
  notes: string[]
}

/** Target-load bands shared with the backend; these are not packed-row fit guarantees. */
export function h3PerformanceAdvice(input: H3PerformanceAdviceInput): H3PerformanceAdvice | null {
  const options = input.options
  const architecture = String(options?.architecture ?? options?.model_type ?? '')
  const vram = input.totalVramGb
  const ram = input.totalRamGb
  if (!options || !architecture.startsWith(policy.scope.architecture_prefix)
    || options.audio_only || policy.scope.excluded_architectures.includes(architecture) || architecture.includes('viggle') || options.model_type.includes('viggle')
    || options.minimax_h3_full_checkpoint || options.model_type.includes('_full')
    || !Number.isFinite(vram) || vram < policy.scope.gpu_vram_gb_min || vram > policy.scope.gpu_vram_gb_max
    || !Number.isFinite(ram) || ram < policy.scope.ram_gb_min) return null

  const notes: string[] = []
  const resolution = input.resolution.trim().toLowerCase()
  const dimensions = resolution.match(/^(\d+)x(\d+)$/)
  const memory = options.omni_reference ? options.omni_sequence_memory_policy : options.sliding_window_memory_policy
  const alignment = policy.geometry.canvas_alignment
  const area = dimensions
    ? Math.ceil(Number(dimensions[1]) / alignment) * alignment
      * Math.ceil(Number(dimensions[2]) / alignment) * alignment
    : memory?.auto_resolution_pixels?.[resolution]
  if (!area || !Number.isFinite(area) || area <= 0 || !Number.isFinite(input.frames) || input.frames <= 0) {
    return { split: null, label: 'Workload size unknown', targetRows: null, slow: false,
      experimental: true, notes: ['Choose a known resolution and window length to see a head-splitting recommendation.'] }
  }
  const geometry = policy.geometry
  const frames = Math.max(geometry.video_latent_frame_offset, Math.ceil(input.frames))
  const latents = Math.ceil((frames - geometry.video_latent_frame_offset) / geometry.video_latent_frame_divisor)
    * geometry.video_latent_multiplier + geometry.video_latent_bias
  const targetRows = latents * area / geometry.area_divisor
    + Math.round(frames / geometry.audio_latent_fps * geometry.audio_latent_rate) * geometry.audio_latent_multiplier
  let split = targetRows <= policy.bands.off_max_target_rows ? 0
    : targetRows <= policy.bands.low_max_target_rows ? 1 : 2
  const beyondEvidence = frames > policy.geometry.medium_max_frames || area > policy.geometry.medium_max_area_pixels
    || targetRows > policy.bands.medium_max_target_rows
  if (beyondEvidence) split = 2
  const fused = options.minimax_h3_fused_turbo === true
  const pikaSlaUnavailable = options.minimax_h3_pika_flex === true
    && options.sla_attention_status?.supported !== true
  const attention = input.attention === 'auto' || !input.attention
    ? (options.sla_attention_default && !pikaSlaUnavailable ? 'sla' : 'auto')
    : (pikaSlaUnavailable && input.attention === 'sla' ? 'auto' : input.attention)
  const sparseAttention = attention === 'sla' || attention === 'veda'
  const slow = !fused && !sparseAttention
    && input.steps >= 20 && frames >= 243 && area >= 1920 * 1088
  const vedaAttention = attention === 'veda'
  const unverifiedRecipe = fused
    ? input.steps !== policy.benchmark.variants.fused_turbo.steps || attention !== 'sla'
    : attention === 'sla' || vedaAttention
  const hasExtraMedia = (input.referenceCount ?? 0) > 0 || input.videoGuide || (input.continuationFrames ?? 0) > 0
  const hasLoras = (input.loraCount ?? 0) > 0
  const experimental = beyondEvidence || (hasLoras && split > 0) || vedaAttention
  if (slow) notes.push('Dense/Sage2 Pruned at 20 steps took about 116 minutes at 1080p / 10.1s on the 12 GB test machine; the 15s test projected hours. This is a dense/Sage2 observation, not an SLA or Veda sparse-attention ETA. Shorter 720p windows are recommended for dense/Sage2 speed.')
  else if (fused && split > 0 && !unverifiedRecipe && !hasLoras && !beyondEvidence) {
    notes.push('Fused at 6 steps with SLA completed 1080p / 10.1s in about 18 minutes with Low, and 15.1s in about 32 minutes with Medium. These are individual test results, not an ETA.')
  }
  if (hasLoras && split > 0) notes.push('Long 1080p generations with LoRAs remain unverified and may need more VRAM or take substantially longer.')
  if (hasExtraMedia) notes.push('References and continuation add conditioning beyond this target-load estimate; large inputs may still run out of memory.')
  if (beyondEvidence) notes.push('This window exceeds the tested 1080p / 15.1s envelope. Medium is a starting choice, not a verified fit.')
  if (unverifiedRecipe) notes.push('The selected steps or attention differ from the tested recipe; speed and memory fit are unverified.')
  if (dimensions && (Number(dimensions[1]) % 32 || Number(dimensions[2]) % 32)) {
    notes.push('H3 rounds the output canvas; this is an approximate workload band.')
  }
  notes.push('Based on an RTX 3080 Ti with 12 GB VRAM and 32 GB RAM. Other GPUs and available memory can change the result.')
  return { split, label: ['Off', 'Low', 'Medium'][split], targetRows, slow, experimental, notes }
}
