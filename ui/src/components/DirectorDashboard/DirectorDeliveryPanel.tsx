import { useEffect, useMemo, useState } from 'react'
import { Download, Loader2, Sparkles } from 'lucide-react'
import type { PipelineClipState } from '../../types'
import { dlssSpatialOptions, trackMediaFlowJobs, useDlssAvailability } from '../../lib/mediaFlow'
import { relayvsrOptions } from '../../lib/flashvsr'
import { MediaFinishingControls } from '../Sidebar/MediaFinishingControls'
import {
  getDirectorLyricsSrt,
  setDirectorSongSubtitles,
  upscaleDirectorJoinedCut,
  upscaleDirectorShot,
  type DirectorJoinedCut,
  type DirectorShowrunnerFeatures,
} from './directorShowrunnerApi'

function downloadText(filename: string, content: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: 'application/x-subrip;charset=utf-8' }))
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export function DirectorDeliveryPanel({
  pipelineId,
  status,
  pipelineType,
  clips,
  features,
  joinedCuts,
  onFeaturesChanged,
}: {
  pipelineId: string
  status: string
  pipelineType: string
  clips: PipelineClipState[]
  features: DirectorShowrunnerFeatures | null
  joinedCuts: DirectorJoinedCut[]
  onFeaturesChanged: () => Promise<void>
}) {
  const dlss = useDlssAvailability()
  const upscalable = useMemo(
    () => clips.filter(clip => clip.video_filename && !clip.video_stale),
    [clips],
  )
  const [clipIndex, setClipIndex] = useState<number | null>(null)
  const [joinedCutFilename, setJoinedCutFilename] = useState('')
  const [spatial, setSpatial] = useState('')
  const [options, setOptions] = useState<Record<string, unknown>>({})
  const [savingLyrics, setSavingLyrics] = useState(false)
  const [upscaling, setUpscaling] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    if (clipIndex == null || !upscalable.some(clip => clip.index === clipIndex)) {
      setClipIndex(upscalable[0]?.index ?? null)
    }
  }, [clipIndex, upscalable])

  useEffect(() => {
    if (!joinedCuts.some(cut => cut.filename === joinedCutFilename)) {
      setJoinedCutFilename(joinedCuts[0]?.filename ?? '')
    }
  }, [joinedCutFilename, joinedCuts])

  if (status !== 'completed') return null

  const selectClass = 'w-full rounded-lg border border-border bg-bg-tertiary p-2 text-xs text-text-primary'
  const songLyricsAvailable = pipelineType === 'music_video' && features?.song_lyric_subtitles
  const lyricsEnabled = features?.song_subtitles === true

  const toggleLyrics = async (enabled: boolean) => {
    setSavingLyrics(true)
    setError('')
    setMessage('')
    try {
      await setDirectorSongSubtitles(pipelineId, enabled)
      await onFeaturesChanged()
      setMessage(enabled
        ? 'Lyrics are ready as a separate SRT soft subtitle track.'
        : 'Song lyric subtitles are off. Spoken subtitle behavior is unchanged.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not update song subtitles.')
    } finally {
      setSavingLyrics(false)
    }
  }

  const downloadLyrics = async () => {
    setError('')
    setMessage('')
    try {
      const result = await getDirectorLyricsSrt(pipelineId)
      downloadText(result.filename, result.content)
      setMessage(`Downloaded ${result.cue_count} lyric cues. Import the SRT in Editor Captions to add a removable subtitle track.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not export song lyrics.')
    }
  }

  const queueUpscale = async () => {
    const clip = upscalable.find(item => item.index === clipIndex)
    if (!clip) return
    setUpscaling(true)
    setError('')
    setMessage('')
    try {
      const result = await upscaleDirectorShot(pipelineId, clip, spatial, options)
      trackMediaFlowJobs(result.job_ids)
      setMessage('Queued a new upscaled shot in Media Flow. The Director source take stays unchanged.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not queue this shot for upscaling.')
    } finally {
      setUpscaling(false)
    }
  }

  const queueJoinedCutUpscale = async () => {
    const cut = joinedCuts.find(item => item.filename === joinedCutFilename)
    if (!cut) return
    setUpscaling(true)
    setError('')
    setMessage('')
    try {
      const result = await upscaleDirectorJoinedCut(pipelineId, cut.filename, spatial, options)
      trackMediaFlowJobs(result.job_ids)
      setMessage('Queued a new upscaled joined cut in Media Flow. The Director assembly stays unchanged.')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not queue this joined cut for upscaling.')
    } finally {
      setUpscaling(false)
    }
  }

  return <section className="rounded-lg border border-border bg-bg-secondary p-3 space-y-3" data-testid="director-delivery">
    <div className="flex items-center gap-2">
      <Sparkles size={13} className="text-accent-blue" />
      <h3 className="text-[11px] font-medium uppercase tracking-wider text-text-secondary">Delivery</h3>
    </div>

    {songLyricsAvailable && <div className="rounded border border-border bg-bg-tertiary p-2 space-y-2">
      <label className="flex items-start gap-2 text-xs text-text-primary">
        <input
          type="checkbox"
          checked={lyricsEnabled}
          disabled={savingLyrics}
          onChange={event => void toggleLyrics(event.target.checked)}
          aria-label="Include song lyrics as subtitles"
          className="mt-0.5"
        />
        <span>
          <span className="block">Include song lyrics as subtitles</span>
          <span className="mt-0.5 block text-[10px] text-text-muted">
            Off by default. Spoken subtitle behavior stays unchanged. Export the lyrics as SRT, then import them in Editor Captions for a removable subtitle track.
          </span>
        </span>
      </label>
      {lyricsEnabled && <button
        type="button"
        onClick={() => void downloadLyrics()}
        className="inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-[10px] text-text-secondary hover:text-text-primary"
      ><Download size={11} /> Download lyric SRT</button>}
    </div>}

    <div className="grid grid-cols-1 gap-2 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <label className="block space-y-1 text-xs text-text-secondary">
        Shot take
        <select
          className={selectClass}
          value={clipIndex ?? ''}
          onChange={event => setClipIndex(event.target.value === '' ? null : Number(event.target.value))}
          disabled={!upscalable.length || upscaling}
        >
          {!upscalable.length && <option value="">No current rendered shots</option>}
          {upscalable.map(clip => <option key={clip.index} value={clip.index}>Shot {clip.index + 1}</option>)}
        </select>
      </label>
      <label className="block space-y-1 text-xs text-text-secondary">
        Spatial finishing
        <select className={selectClass} value={spatial} onChange={event => setSpatial(event.target.value)} disabled={upscaling}>
          <option value="">Choose an upscaler</option>
          {dlssSpatialOptions.map(option => <option key={option.value} value={option.value} disabled={dlss.disabled(option.value)}>{dlss.label(option.value, option.label)}</option>)}
          {relayvsrOptions.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          {[1.5, 2, 3, 4].map(scale => <option key={scale} value={`lanczos${scale}`}>Lanczos ×{scale}</option>)}
        </select>
      </label>
    </div>
    {spatial.startsWith('dlss5*') && <MediaFinishingControls
      spatial={spatial}
      temporal=""
      onTemporal={() => undefined}
      options={options}
      onOptions={setOptions}
      image={false}
    />}
    <button
      type="button"
      disabled={!upscalable.length || clipIndex == null || !spatial || upscaling}
      onClick={() => void queueUpscale()}
      className="inline-flex items-center gap-2 rounded-lg bg-cta px-3 py-2 text-xs text-white disabled:opacity-40"
    >{upscaling && <Loader2 size={12} className="animate-spin" />}{upscaling ? 'Queueing…' : 'Queue new upscaled take'}</button>
    {joinedCuts.length > 0 && <div className="rounded border border-border bg-bg-tertiary p-2 space-y-2">
      <label className="block space-y-1 text-xs text-text-secondary">
        Finished joined cut
        <select
          className={selectClass}
          value={joinedCutFilename}
          onChange={event => setJoinedCutFilename(event.target.value)}
          disabled={upscaling}
        >
          {joinedCuts.map(cut => <option key={cut.filename} value={cut.filename}>{cut.filename}</option>)}
        </select>
      </label>
      <button
        type="button"
        disabled={!joinedCutFilename || !spatial || upscaling}
        onClick={() => void queueJoinedCutUpscale()}
        className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-2 text-xs text-text-primary disabled:opacity-40"
      >{upscaling && <Loader2 size={12} className="animate-spin" />}{upscaling ? 'Queueing…' : 'Queue joined cut upscale'}</button>
    </div>}
    {(message || error) && <p className={`break-words text-[10px] ${error ? 'text-indicator-warning' : 'text-text-muted'}`}>{error || message}</p>}
  </section>
}
