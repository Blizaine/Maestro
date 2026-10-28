import { useEffect, useState } from 'react'
import { editorMixPeak } from './editorAudioGraph'

export function EditorAudioMeter({ playing }: { playing: boolean }) {
  const [peak, setPeak] = useState<number | null>(null)
  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => setPeak(editorMixPeak()), 80)
    return () => window.clearInterval(timer)
  }, [playing])
  const level = playing ? peak : null
  const db = level && level > 0 ? Math.max(-60, 20 * Math.log10(level)) : -60
  return <div className="hidden w-20 shrink-0 space-y-1 sm:block" role="meter" aria-label="Preview audio peak"
    aria-valuemin={-60} aria-valuemax={0} aria-valuenow={Math.min(0, db)}
    aria-valuetext={level === null ? 'Play to meter preview audio' : `${db.toFixed(1)} dBFS`}
    title={level === null ? 'Preview audio meter activates on Play' : `Preview peak ${db.toFixed(1)} dBFS${db >= 0 ? ' · clipping' : ''}`}>
    <div className="h-1.5 overflow-hidden rounded bg-bg-active"><div className={`h-full ${db >= 0 ? 'bg-red-500' : db > -6 ? 'bg-amber-400' : 'bg-accent-green'}`} style={{ width: `${Math.max(0, Math.min(100, (db + 60) / 60 * 100))}%` }} /></div>
    <span className="block text-right font-mono text-[9px] text-text-muted">{level === null ? 'Audio' : `${db.toFixed(0)} dBFS`}</span>
  </div>
}
