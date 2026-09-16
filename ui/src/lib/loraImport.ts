/** A URL hint only; unrecognized names leave detection to repository metadata. */
export function suggestLoraImportDirectory(url: string): string {
  let identity: string
  try {
    const parsed = new URL(url.trim())
    if (!['http:', 'https:'].includes(parsed.protocol)) return ''
    const host = parsed.hostname.toLowerCase().replace(/^www\./, '')
    if (host !== 'huggingface.co' && host !== 'civitai.com') return ''
    const parts = decodeURIComponent(parsed.pathname).split('/').filter(Boolean)
    // Ignore the HF author's name and URL query (which can contain access tokens).
    identity = (host === 'huggingface.co' ? parts.slice(1) : parts).join(' ').toLowerCase()
  } catch {
    return ''
  }
  const compact = identity.replace(/[^a-z0-9]/g, '')
  if (compact.includes('minimaxmusic3')) return 'minimax_music3_music'
  if (compact.includes('minimax') || /\bmmh3\b/.test(identity)) return 'minimax_h3'
  if (compact.includes('ltx25')) return 'ltx25'
  if (compact.includes('ltx2') || compact.includes('ltxvideo2')) return 'ltx2'
  if (compact.includes('ltxv') || compact.includes('ltxvideo')) return 'ltxv'
  if (compact.includes('klein')) {
    if (compact.includes('4b')) return 'flux2_klein_4b'
    if (compact.includes('9b')) return 'flux2_klein_9b'
    return '' // Klein's 4B and 9B adapters belong in different folders.
  }
  if (compact.includes('flux2')) return 'flux2_dev'
  if (compact.includes('kontext')) return 'flux_dev_kontext'
  if (/\bflux(?:[\s._-]*1)?\b/.test(identity)) return 'flux'
  if (compact.includes('krea2')) return 'krea2'
  if (/\bqwen(?:[\s._-]*image)?\b/.test(identity)) return 'qwen'
  if (compact.includes('zimage')) return 'z_image'
  if (/\bwan[\s._-]*2/.test(identity)) {
    if (compact.includes('13b')) return 'wan_1.3B'
    if (compact.includes('5b') && !compact.includes('14b')) return 'wan_5B'
    // Wan 2.2's two experts share wan; Wan 2.1/2.5 I2V uses wan_i2v.
    if (compact.includes('i2v') && !compact.includes('wan22')) return 'wan_i2v'
    return 'wan'
  }
  return ''
}

export function resolveLoraImportDirectory(url: string, manualDirectory: string): string {
  return manualDirectory || suggestLoraImportDirectory(url)
}
