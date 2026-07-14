import { useState } from 'react'
import { Plus, Trash2, Play, ArrowRight } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { FileUploadZone } from '../shared/FileUploadZone'
import * as api from '../../api/client'

interface MixerTrack {
  id: number
  filename: string | null   // display name
  path: string | null        // backend path (upload path or workspace file)
  startTime: number          // seconds offset
  volume: number             // 0-100 percent
  durationSec: number | null // auto-detected
}

let _nextTrackId = 1

function emptyTrack(): MixerTrack {
  return { id: _nextTrackId++, filename: null, path: null, startTime: 0, volume: 100, durationSec: null }
}

function getAudioDuration(file: File): Promise<number | null> {
  return new Promise(resolve => {
    const url = URL.createObjectURL(file)
    const audio = new Audio()
    audio.addEventListener('loadedmetadata', () => {
      const dur = audio.duration
      URL.revokeObjectURL(url)
      resolve(Number.isFinite(dur) ? Math.round(dur * 10) / 10 : null)
    })
    audio.addEventListener('error', () => { URL.revokeObjectURL(url); resolve(null) })
    audio.src = url
  })
}

export function MixerControls() {
  const setParam = useStore(s => s.setParam)
  const setAudioGuideFilename = useStore(s => s.setAudioGuideFilename)
  const setGenerationMode = useStore(s => s.setGenerationMode)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)
  const activeWorkspace = useStore(s => s.activeWorkspace)

  const [baseTrack, setBaseTrack] = useState<MixerTrack>(() => ({ ...emptyTrack(), volume: 100 }))
  const [overlays, setOverlays] = useState<MixerTrack[]>([])
  const [mixing, setMixing] = useState(false)
  const [mixResult, setMixResult] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = async (file: File, track: MixerTrack, update: (t: MixerTrack) => void) => {
    try {
      const result = await api.uploadImage(file)
      const dur = await getAudioDuration(file)
      update({ ...track, filename: file.name, path: result.path, durationSec: dur })
    } catch (e) {
      console.error('Upload failed:', e)
    }
  }

  const updateOverlay = (id: number, partial: Partial<MixerTrack>) => {
    setOverlays(prev => prev.map(t => t.id === id ? { ...t, ...partial } : t))
  }

  const removeOverlay = (id: number) => {
    setOverlays(prev => prev.filter(t => t.id !== id))
  }

  const totalDuration = () => {
    let max = baseTrack.durationSec || 0
    for (const t of overlays) {
      if (t.path && t.durationSec) {
        max = Math.max(max, t.startTime + t.durationSec)
      }
    }
    return Math.round(max * 10) / 10
  }

  const canMix = baseTrack.path && overlays.some(t => t.path)

  const doMix = async (useAsGuide: boolean) => {
    if (!canMix) return
    setMixing(true)
    setError(null)
    setMixResult(null)
    try {
      const tracks = [
        { path: baseTrack.path!, start_time: 0, volume: baseTrack.volume / 100 },
        ...overlays.filter(t => t.path).map(t => ({
          path: t.path!,
          start_time: t.startTime,
          volume: t.volume / 100,
        })),
      ]
      const data = await api.mixAudio(tracks, activeWorkspace)
      setMixResult(data.filename)

      if (useAsGuide) {
        // Switch to Video mode with the mixed audio as audio guide
        setGenerationMode('video')
        setParam('audio_guide' as keyof import('../../types').GenerateParams, data.path)
        setAudioGuideFilename(data.filename)
        // Set video duration to match the mix
        const dur = totalDuration()
        if (dur > 0) setDurationSeconds(Math.ceil(dur))
        // Set audio mode to "A" (use audio guide)
        setParam('audio_prompt_type', 'A')
      }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setMixing(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Base Track */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Base Track <span className="normal-case text-text-muted">(full duration)</span>
        </label>
        <FileUploadZone
          label="Drop base audio (.wav, .mp3)"
          accept=".wav,.mp3,.flac,.ogg,.m4a"
          filename={baseTrack.filename}
          onFile={f => handleFileUpload(f, baseTrack, setBaseTrack)}
          onClear={() => setBaseTrack({ ...emptyTrack(), volume: 100 })}
        />
        {baseTrack.path && (
          <div className="flex items-center gap-3 mt-1.5">
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <span className="text-[9px] text-text-muted">Volume</span>
                <span className="text-[9px] text-text-muted">{baseTrack.volume}%</span>
              </div>
              <input type="range" min={0} max={100} step={1}
                value={baseTrack.volume}
                onChange={e => setBaseTrack(t => ({ ...t, volume: Number(e.target.value) }))}
                className="w-full"
              />
            </div>
            {baseTrack.durationSec && (
              <span className="text-[9px] text-text-muted shrink-0">{baseTrack.durationSec}s</span>
            )}
          </div>
        )}
      </div>

      {/* Overlay Tracks */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">
            Overlay Tracks
          </label>
          <button
            onClick={() => setOverlays(prev => [...prev, { ...emptyTrack(), volume: 30 }])}
            className="flex items-center gap-1 text-[10px] text-accent-blue hover:text-accent-blue/80 transition-colors"
          >
            <Plus size={11} /> Add
          </button>
        </div>

        {overlays.length === 0 && (
          <p className="text-[9px] text-text-muted text-center py-2">
            Add overlay tracks (SFX, ambience) to layer on top of the base track.
          </p>
        )}

        <div className="space-y-2">
          {overlays.map((track, idx) => (
            <div key={track.id} className="bg-bg-tertiary/50 border border-border rounded-lg p-2 space-y-1.5">
              <div className="flex items-center gap-1.5">
                <span className="text-[9px] text-text-muted shrink-0">{idx + 1}.</span>
                <div className="flex-1 min-w-0">
                  {!track.path ? (
                    <FileUploadZone
                      label="Drop audio"
                      accept=".wav,.mp3,.flac,.ogg,.m4a"
                      filename={track.filename}
                      onFile={f => handleFileUpload(f, track, t => updateOverlay(track.id, t))}
                      onClear={() => removeOverlay(track.id)}
                    />
                  ) : (
                    <div className="flex items-center gap-1.5 bg-bg-tertiary rounded px-2 py-1">
                      <span className="text-[10px] text-text-primary truncate flex-1">{track.filename}</span>
                      {track.durationSec && (
                        <span className="text-[9px] text-text-muted shrink-0">{track.durationSec}s</span>
                      )}
                    </div>
                  )}
                </div>
                <button
                  onClick={() => removeOverlay(track.id)}
                  className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-red-400 transition-colors shrink-0"
                >
                  <Trash2 size={11} />
                </button>
              </div>

              {track.path && (
                <div className="grid grid-cols-2 gap-2">
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] text-text-muted">Start</span>
                      <span className="text-[9px] text-text-muted">{track.startTime}s</span>
                    </div>
                    <input type="number" min={0} step={0.1}
                      value={track.startTime}
                      onChange={e => updateOverlay(track.id, { startTime: Math.max(0, parseFloat(e.target.value) || 0) })}
                      className="w-full bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue"
                    />
                  </div>
                  <div>
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] text-text-muted">Volume</span>
                      <span className="text-[9px] text-text-muted">{track.volume}%</span>
                    </div>
                    <input type="range" min={0} max={100} step={1}
                      value={track.volume}
                      onChange={e => updateOverlay(track.id, { volume: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Duration summary */}
      {canMix && (
        <div className="text-[10px] text-text-muted text-center">
          Total duration: ~{totalDuration()}s
        </div>
      )}

      {/* Mix buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => doMix(false)}
          disabled={!canMix || mixing}
          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs border border-border bg-bg-tertiary text-text-secondary hover:text-text-primary hover:border-border-light transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Play size={12} />
          {mixing ? 'Mixing...' : 'Mix & Save'}
        </button>
        <button
          onClick={() => doMix(true)}
          disabled={!canMix || mixing}
          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs bg-accent-blue text-white hover:bg-accent-blue/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <ArrowRight size={12} />
          {mixing ? 'Mixing...' : 'Mix & Use as Guide'}
        </button>
      </div>

      {/* Result / Error */}
      {mixResult && (
        <div className="text-[10px] text-indicator-success bg-green-500/10 border border-green-500/20 rounded px-2 py-1.5">
          Saved: {mixResult}
        </div>
      )}
      {error && (
        <div className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5">
          {error}
        </div>
      )}
    </div>
  )
}
