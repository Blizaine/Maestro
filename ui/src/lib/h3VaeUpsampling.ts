import type { ModelOptions } from '../types'

export const H3_VAE_UPSAMPLER = 'h3_vae*2'
export const H3_VAE_UPSAMPLER_LABEL = 'H3 VAE 2×'

export function modelOptionsForSelectedModel(
  options: ModelOptions | null | undefined,
  modelType: string | null | undefined,
): ModelOptions | null {
  return options && modelType && options.model_type === modelType ? options : null
}

export function supportsH3VaeUpsampling(
  options: ModelOptions | null | undefined,
  imageMode: number,
): boolean {
  return options?.vae_upsamplers?.h3_vae?.includes(imageMode) === true
}

export function sanitizeH3VaeUpsampling(
  value: string,
  options: ModelOptions | null | undefined,
  imageMode: number,
): string {
  return value === H3_VAE_UPSAMPLER && !supportsH3VaeUpsampling(options, imageMode)
    ? ''
    : value
}

function resolutionDimensions(resolution: string | null | undefined): [number, number] | null {
  const match = String(resolution || '').match(/^\s*(\d+)\s*[x×]\s*(\d+)\s*$/i)
  return match ? [Number(match[1]), Number(match[2])] : null
}

export function h3VaeUpsamplingHelp(resolution?: string | null): string {
  const dimensions = resolutionDimensions(resolution)
  const dimensionsHelp = dimensions
    ? ' At ' + dimensions[0] + '×' + dimensions[1] + ', the output is ' + dimensions[0] * 2 + '×' + dimensions[1] * 2 + '.'
    : ' For example, 540p at 16:9 (960×544) becomes 1920×1088.'

  return 'Decodes at twice the width and height of the chosen generation resolution.' + dimensionsHelp + ' Optional; downloads about 2.8 GB on first use and adds decoding memory and time. Keeps denoising at the selected resolution.'
}
