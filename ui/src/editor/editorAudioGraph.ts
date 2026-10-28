// One lazily opened preview mixer. Native playback remains the fallback.
let context: AudioContext | null = null
let master: GainNode | null = null
let meter: AnalyserNode | null = null
const nodes = new WeakMap<HTMLMediaElement, { source: MediaElementAudioSourceNode; gain: GainNode; connected: boolean }>()

function connect(element: HTMLMediaElement) {
  if (!context || !master) return null
  let entry = nodes.get(element)
  if (!entry) {
    try {
      entry = { source: context.createMediaElementSource(element), gain: context.createGain(), connected: false }
      nodes.set(element, entry)
    } catch { return null }
  }
  if (!entry.connected) {
    entry.source.connect(entry.gain)
    entry.gain.connect(master)
    entry.connected = true
  }
  return entry
}

export function setEditorMediaGain(element: HTMLMediaElement, rawLevel: number): void {
  const level = Number.isFinite(rawLevel) ? Math.max(0, Math.min(16, rawLevel)) : 0
  element.dataset.editorGain = String(level)
  const entry = connect(element)
  if (context && entry) {
    element.volume = 1
    entry.gain.gain.setTargetAtTime(level, context.currentTime, 0.008)
  } else element.volume = Math.min(1, level)
}

/** Call only from the user's Play gesture, including Safari audio unlock. */
export function resumeEditorAudio(elements: Iterable<HTMLMediaElement>): void {
  try {
    if (!context) {
      const Constructor = window.AudioContext
        || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
      if (!Constructor) return
      context = new Constructor()
      master = context.createGain()
      meter = context.createAnalyser()
      meter.fftSize = 512
      master.connect(meter)
      meter.connect(context.destination)
    }
    void context.resume().catch(() => {})
    for (const element of elements) setEditorMediaGain(element, Number(element.dataset.editorGain ?? element.volume))
  } catch { /* Restricted browsers keep their native media audio path. */ }
}

export function releaseEditorMediaAudio(element: HTMLMediaElement): void {
  const entry = nodes.get(element)
  if (!entry?.connected) return
  entry.source.disconnect()
  entry.gain.disconnect()
  entry.connected = false
}

const samples = new Float32Array(512)
export function editorMixPeak(): number | null {
  if (!meter || context?.state !== 'running') return null
  meter.getFloatTimeDomainData(samples)
  let peak = 0
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample))
  return peak
}
