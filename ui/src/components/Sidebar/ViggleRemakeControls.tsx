import { useCallback, useMemo, useRef, useState } from 'react'
import { Film, ImagePlus, Loader2, Upload, WandSparkles, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { loadMediaInput } from '../../lib/mediaInput'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import { GalleryInput } from '../shared/GalleryInput'
import * as api from '../../api/client'

const MAX_FRAMES = 4
const MAX_TRIM_DURATION = 345 / 24
const MIN_TRIM_DURATION = 3.75
const DEFAULT_TRIM_START = 0
const DEFAULT_TRIM_END = MAX_TRIM_DURATION

function basename(path: string): string {
  return path.replace(/\\/g, '/').split('/').pop() || path
}

function sourceUrl(path: string): string {
  if (path.includes('/api/v1/file/') || path.includes('/api/v1/uploads/')) return path
  if (/(?:^|[/\\])uploads(?:[/\\])/i.test(path)) return api.getUploadUrl(basename(path))
  return api.getFileUrl(basename(path))
}

function defaultTrim(duration: number): { start: number; end: number } {
  return { start: DEFAULT_TRIM_START, end: Math.min(DEFAULT_TRIM_END, Math.max(0, duration)) }
}

function nextFrameTime(times: number[], start: number, end: number): number {
  const ordered = times.filter(time => Number.isFinite(time) && time > start && time < end)
    .sort((left, right) => left - right)
  const points = [start, ...ordered, end]
  let bestStart = start
  let bestEnd = end
  for (let index = 0; index < points.length - 1; index += 1) {
    if (points[index + 1] - points[index] > bestEnd - bestStart) {
      bestStart = points[index]
      bestEnd = points[index + 1]
    }
  }
  return Math.round(((bestStart + bestEnd) / 2) * 100) / 100
}

function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${minutes}:${remainder.toFixed(3).padStart(6, '0')}`
}

export function ViggleRemakeControls({ aligned = false }: { aligned?: boolean }) {
  const editVideoFile = useStore(state => state.editVideoFile)
  const editVideoPath = useStore(state => state.editVideoPath)
  const editVideoUrl = useStore(state => state.editVideoUrl)
  const editVideoDuration = useStore(state => state.editVideoDuration)
  const params = useStore(state => state.params)
  const setParam = useStore(state => state.setParam)
  const setEditVideo = useStore(state => state.setEditVideo)
  const clearEditVideo = useStore(state => state.clearEditVideo)
  const setGuideVideoFps = useStore(state => state.setGuideVideoFps)
  const sendFrameToImageMode = useStore(state => state.sendFrameToImageMode)
  const workspace = useStore(state => state.activeWorkspace)
  const [uploadingSource, setUploadingSource] = useState(false)
  const [uploadingFrame, setUploadingFrame] = useState(false)
  const [selectedFrameTime, setSelectedFrameTime] = useState(0)
  const [frameEditError, setFrameEditError] = useState('')
  const [framePreviewUrls, setFramePreviewUrls] = useState<Record<string, string>>({})
  const frameLimit = aligned ? 1 : MAX_FRAMES
  const sourceFileRef = useRef<HTMLInputElement>(null)
  const frameFileRef = useRef<HTMLInputElement>(null)

  const paths = useMemo(() => Array.isArray(params.image_refs)
    ? params.image_refs.filter((path): path is string => typeof path === 'string' && path.trim().length > 0)
    : [], [params.image_refs])
  const savedTimes = useMemo(() => Array.isArray(params.viggle_remake_frame_times)
    ? params.viggle_remake_frame_times.map(value => Number(value))
    : [], [params.viggle_remake_frame_times])
  const duration = editVideoDuration
  const fallbackRange = defaultTrim(duration || DEFAULT_TRIM_END)
  const trimStart = Number.isFinite(Number(params.viggle_trim_start))
    ? Number(params.viggle_trim_start)
    : duration > 0 ? fallbackRange.start : DEFAULT_TRIM_START
  const trimEnd = Number.isFinite(Number(params.viggle_trim_end))
    ? Number(params.viggle_trim_end)
    : duration > 0 ? fallbackRange.end : DEFAULT_TRIM_END
  const videoUrl = editVideoUrl || (editVideoPath ? sourceUrl(editVideoPath) : '')
  const frames = useMemo(() => paths.map((path, index) => ({
    path,
    time: Number.isFinite(savedTimes[index]) ? savedTimes[index] : nextFrameTime(savedTimes, trimStart, trimEnd),
    url: framePreviewUrls[path] || sourceUrl(path),
    name: basename(path),
  })).sort((left, right) => left.time - right.time), [paths, savedTimes, trimStart, trimEnd, framePreviewUrls])

  const writeFrames = useCallback((nextPaths: string[], nextTimes: number[]) => {
    const ordered = nextPaths.map((path, index) => ({ path, time: nextTimes[index] }))
      .sort((left, right) => left.time - right.time)
    setParam('image_refs', ordered.map(frame => frame.path))
    setParam('viggle_remake_frame_times', aligned ? undefined : ordered.map(frame => frame.time))
  }, [aligned, setParam])

  const handleSourceUpload = useCallback(async (file: File) => {
    setUploadingSource(true)
    try {
      const [uploaded, media] = await Promise.all([api.uploadImage(file), loadMediaInput(file)])
      setEditVideo(file, uploaded.path, media.url, media.duration, media.resolution)
      setParam('video_guide', uploaded.path)
      setGuideVideoFps(uploaded.fps && uploaded.fps > 0 ? uploaded.fps : null)
      const range = defaultTrim(media.duration || DEFAULT_TRIM_END)
      setParam('viggle_trim_start', range.start)
      setParam('viggle_trim_end', range.end)
      setSelectedFrameTime(range.start)
      useStore.setState({ editStartTime: range.start, editEndTime: range.end })
    } catch (error) {
      console.error('Remake source upload failed:', error)
      return false
    } finally {
      setUploadingSource(false)
    }
  }, [setEditVideo, setGuideVideoFps, setParam])

  const handleFrameUpload = useCallback(async (file: File) => {
    if (frames.length >= frameLimit) return false
    setUploadingFrame(true)
    try {
      const uploaded = await api.uploadImage(file)
      const framePaths = frames.map(frame => frame.path)
      const times = frames.map(frame => frame.time)
      const time = nextFrameTime(times, trimStart, trimEnd)
      writeFrames([...framePaths, uploaded.path], [...times, time])
      setFramePreviewUrls(previous => ({ ...previous, [uploaded.path]: URL.createObjectURL(file) }))
    } catch (error) {
      console.error('Remake reference frame upload failed:', error)
      return false
    } finally {
      setUploadingFrame(false)
    }
  }, [frameLimit, frames, trimStart, trimEnd, writeFrames])

  const handleTrimChange = useCallback((start: number, end: number) => {
    if (!Number.isFinite(start) || !Number.isFinite(end)) return
    const maxEnd = Math.min(MAX_TRIM_DURATION, duration > 0 ? duration : MAX_TRIM_DURATION)
    const minSpan = Math.min(MIN_TRIM_DURATION, maxEnd)
    const safeStart = Math.max(0, Math.min(start, maxEnd - minSpan))
    const safeEnd = Math.min(maxEnd, Math.max(safeStart + minSpan, end))
    setParam('viggle_trim_start', safeStart)
    setParam('viggle_trim_end', safeEnd)
    setSelectedFrameTime(current => Math.max(safeStart, Math.min(safeEnd - 1 / 24, current)))
    useStore.setState({ editStartTime: safeStart, editEndTime: safeEnd })
    const framePaths = frames.map(frame => frame.path)
    const nextTimes = frames.map(frame => Math.max(safeStart, Math.min(safeEnd - 1 / 24, frame.time)))
    writeFrames(framePaths, nextTimes)
  }, [duration, frames, setParam, writeFrames])

  const handleFrameTime = (index: number, value: number) => {
    if (!Number.isFinite(value)) return
    const framePaths = frames.map(frame => frame.path)
    const nextTimes = frames.map(frame => frame.time)
    nextTimes[index] = Math.max(trimStart, Math.min(trimEnd - 1 / 24, value))
    writeFrames(framePaths, nextTimes)
  }

  const removeFrame = (index: number) => {
    writeFrames(
      frames.filter((_, current) => current !== index).map(frame => frame.path),
      frames.filter((_, current) => current !== index).map(frame => frame.time),
    )
  }

  const clearSource = () => {
    clearEditVideo()
    setParam('video_guide', undefined)
    setParam('viggle_trim_start', undefined)
    setParam('viggle_trim_end', undefined)
    setSelectedFrameTime(0)
    setGuideVideoFps(null)
  }

  const extractSourceFrame = async (time: number) => {
    const response = await fetch('/api/v1/extract-frames', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: editVideoPath, start_time: time, workspace }),
    })
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Could not extract the source frame.' }))
      throw new Error(error.detail || 'Could not extract the source frame.')
    }
    const result = await response.json() as { start_path?: string; start_url?: string }
    if (!result.start_path) throw new Error('The source frame could not be saved.')
    return { path: result.start_path, url: result.start_url || sourceUrl(result.start_path) }
  }

  const addSourceFrame = async (time: number) => {
    if (!editVideoPath || frames.length >= MAX_FRAMES) return
    const existing = frames.find(frame => Math.abs(frame.time - time) < 1 / 24)
    if (existing) {
      setSelectedFrameTime(existing.time)
      return
    }
    setUploadingFrame(true)
    setFrameEditError('')
    try {
      const extracted = await extractSourceFrame(time)
      writeFrames([...frames.map(frame => frame.path), extracted.path], [...frames.map(frame => frame.time), time])
      setFramePreviewUrls(previous => ({ ...previous, [extracted.path]: extracted.url }))
    } catch (error) {
      setFrameEditError(error instanceof Error ? error.message : 'Could not extract the source frame.')
    } finally {
      setUploadingFrame(false)
    }
  }

  const suggestSourceFrames = async () => {
    if (!editVideoPath || frames.length >= MAX_FRAMES) return
    const lastFrame = Math.max(trimStart, trimEnd - 1 / 24)
    const candidates = [trimStart, (trimStart + trimEnd) / 2, lastFrame]
      .map(time => Math.round(time * 1000) / 1000)
      .filter((time, index, all) => all.findIndex(candidate => Math.abs(candidate - time) < 1 / 24) === index)
      .filter(time => !frames.some(frame => Math.abs(frame.time - time) < 1 / 24))
      .slice(0, MAX_FRAMES - frames.length)
    if (!candidates.length) return
    setUploadingFrame(true)
    setFrameEditError('')
    try {
      const extracted = await Promise.all(candidates.map(async time => ({ ...await extractSourceFrame(time), time })))
      writeFrames([...frames.map(frame => frame.path), ...extracted.map(frame => frame.path)],
        [...frames.map(frame => frame.time), ...extracted.map(frame => frame.time)])
      setFramePreviewUrls(previous => ({ ...previous,
        ...Object.fromEntries(extracted.map(frame => [frame.path, frame.url])) }))
    } catch (error) {
      setFrameEditError(error instanceof Error ? error.message : 'Could not prepare the suggested source frames.')
    } finally {
      setUploadingFrame(false)
    }
  }

  const editKeyframe = async (time: number) => {
    setFrameEditError('')
    try {
      await sendFrameToImageMode('remake', time)
    } catch (error) {
      setFrameEditError(error instanceof Error ? error.message : 'Could not open the keyframe in Maestro Image Mode.')
    }
  }

  return (
    <div className="space-y-3">
      <GalleryInput kind="video" label="Remake source video" onFile={handleSourceUpload}
        getImages={() => videoUrl ? [{ url: videoUrl, name: editVideoFile?.name || basename(editVideoPath) }] : []}
        disabledReason={uploadingSource ? 'Uploading source video…' : undefined} />
              <div>
        <label className="mb-1.5 block text-[11px] uppercase tracking-wider text-text-muted">Source video</label>
        {editVideoPath ? (
          <div className="relative space-y-2 rounded-lg border border-border bg-bg-tertiary p-2">
            <button type="button" onClick={clearSource} title="Remove source video"
              className="absolute right-2 top-2 z-20 rounded-full bg-black/65 p-1 text-white/80 hover:text-white">
              <X size={13} />
            </button>
            {videoUrl && (
              <VideoTimelineSelector
                videoUrl={videoUrl}
                duration={duration}
                startTime={trimStart}
                endTime={trimEnd}
                playheadTime={selectedFrameTime}
                onStartChange={start => handleTrimChange(start, trimEnd)}
                onEndChange={end => handleTrimChange(trimStart, end)}
                onPlayheadChange={setSelectedFrameTime}
                onMetadata={metadataDuration => {
                  if (!Number.isFinite(metadataDuration) || metadataDuration <= 0) return
                  useStore.setState({ editVideoDuration: metadataDuration })
                  const current = useStore.getState().params
                  const fallback = defaultTrim(metadataDuration)
                  const maxEnd = Math.min(MAX_TRIM_DURATION, metadataDuration)
                  const minSpan = Math.min(MIN_TRIM_DURATION, maxEnd)
                  const savedStart = Number(current.viggle_trim_start)
                  const savedEnd = Number(current.viggle_trim_end)
                  const start = Number.isFinite(savedStart)
                    ? Math.max(0, Math.min(savedStart, maxEnd - minSpan))
                    : fallback.start
                  const end = Number.isFinite(savedEnd)
                    ? Math.min(maxEnd, Math.max(start + minSpan, savedEnd))
                    : Math.min(maxEnd, Math.max(start + minSpan, fallback.end))
                  setParam('viggle_trim_start', start)
                  setParam('viggle_trim_end', end)
                  setSelectedFrameTime(current => Math.max(start, Math.min(end - 1 / 24, current)))
                  useStore.setState({ editStartTime: start, editEndTime: end })
                  const paths = Array.isArray(current.image_refs) ? current.image_refs : []
                  const times = Array.isArray(current.viggle_remake_frame_times)
                    ? current.viggle_remake_frame_times.map(Number) : []
                  writeFrames(paths, paths.map((_, index) => Number.isFinite(times[index])
                    ? Math.max(start, Math.min(end - 1 / 24, times[index]))
                    : nextFrameTime(times, start, end)))
                }}
              />
            )}
            <p className="truncate pr-6 text-[9px] text-text-muted">{editVideoFile?.name || basename(editVideoPath)}</p>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-[9px] text-text-muted">
                Trim start
                <input type="number" min="0" max={Math.max(0, trimEnd - MIN_TRIM_DURATION)} step="0.001" value={trimStart}
                  onChange={event => handleTrimChange(event.currentTarget.valueAsNumber, trimEnd)}
                  className="mt-1 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary" />
              </label>
              <label className="text-[9px] text-text-muted">
                Trim end
                <input type="number" min={trimStart + MIN_TRIM_DURATION} max={Math.min(MAX_TRIM_DURATION, duration || MAX_TRIM_DURATION)} step="0.001" value={trimEnd}
                  onChange={event => handleTrimChange(trimStart, event.currentTarget.valueAsNumber)}
                  className="mt-1 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary" />
              </label>
            </div>
            {!aligned && <label className="block text-[9px] text-text-muted">
              Keyframe to add (seconds)
              <input type="number" min={trimStart} max={Math.max(trimStart, trimEnd - 1 / 24)} step="0.001"
                value={selectedFrameTime} onChange={event => {
                  const time = event.currentTarget.valueAsNumber
                  if (Number.isFinite(time)) setSelectedFrameTime(Math.max(trimStart, Math.min(trimEnd - 1 / 24, time)))
                }}
                className="mt-1 w-full rounded-md border border-border bg-bg-primary px-2 py-1.5 text-[10px] text-text-primary" />
            </label>}
          </div>
        ) : (
          <div className="space-y-2">
            <button type="button" disabled={uploadingSource} onClick={() => sourceFileRef.current?.click()}
              className="w-full rounded-lg border-2 border-dashed border-border p-4 text-center transition-colors hover:border-accent-blue/60 hover:bg-bg-hover/30 disabled:opacity-50">
              <Upload size={18} className="mx-auto mb-1.5 text-text-muted" />
              <span className="text-[11px] text-text-secondary">{uploadingSource ? 'Uploading…' : 'Upload a source video'}</span>
            </button>
            <input ref={sourceFileRef} type="file" accept="video/*" className="hidden"
              aria-label="Upload Remake source video"
              onChange={event => {
                const file = event.currentTarget.files?.[0]
                if (file) void handleSourceUpload(file)
                event.currentTarget.value = ''
              }} />
          </div>
        )}
      </div>

      <div className="space-y-2 rounded-lg border border-border p-2.5">
        <div className="flex items-center justify-between gap-2">
          <label className="text-[11px] uppercase tracking-wider text-text-muted">
            {aligned ? 'Replacement character' : 'Reference frames'}
          </label>
          {!aligned && <span className="text-[9px] text-text-muted">{frames.length}/{frameLimit}</span>}
        </div>
        <p className="text-[10px] leading-snug text-text-muted">
          {aligned
            ? 'Choose one image of the replacement character. The source video supplies the timing, scene and performance.'
            : 'Choose up to four source moments, edit each frame in Maestro Image Mode, then review the replacements before generating.'}
        </p>
        {!aligned && <div className="grid grid-cols-2 gap-2">
          <button type="button" disabled={!editVideoPath || uploadingFrame || frames.length >= MAX_FRAMES}
            onClick={() => void addSourceFrame(selectedFrameTime)}
            className="flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-border text-[10px] text-text-secondary hover:border-accent-blue/60 hover:text-text-primary disabled:opacity-50">
            <ImagePlus size={13} /> Add selected source frame
          </button>
          <button type="button" disabled={!editVideoPath || uploadingFrame || frames.length >= MAX_FRAMES}
            onClick={() => void suggestSourceFrames()}
            className="flex min-h-9 items-center justify-center gap-1.5 rounded-lg border border-border text-[10px] text-text-secondary hover:border-accent-blue/60 hover:text-text-primary disabled:opacity-50">
            {uploadingFrame ? <Loader2 size={13} className="animate-spin" /> : <Film size={13} />}
            Suggest keyframes
          </button>
        </div>}
        <GalleryInput kind="image" label={aligned ? 'Replacement character image' : 'Remake frame'} onFile={handleFrameUpload}
          getImages={() => frames.map(frame => ({ url: frame.url, name: aligned ? frame.name : `Frame at ${formatTime(frame.time)}` }))}
          disabledReason={uploadingFrame ? 'Uploading image…' : frames.length >= frameLimit
            ? aligned ? 'Remove the selected image to choose another.' : 'Four reference frames is the limit.'
            : undefined} />
        {frames.length > 0 && (
          <div className="space-y-2">
            {frames.map((frame, index) => (
              <div key={`${frame.path}-${index}`} className="flex items-center gap-2 rounded-lg border border-border bg-bg-tertiary p-1.5">
                <img src={frame.url} alt={aligned ? 'Replacement character' : `Remake reference frame ${index + 1}`} className="h-12 w-16 rounded object-cover" />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[10px] text-text-secondary">{aligned ? frame.name : `Frame ${index + 1} · ${frame.name}`}</div>
                  {!aligned && <label className="mt-1 flex items-center gap-1 text-[9px] text-text-muted">
                    Time (seconds)
                    <input type="number" min={trimStart} max={trimEnd} step="0.001" value={frame.time}
                      onChange={event => handleFrameTime(index, event.currentTarget.valueAsNumber)}
                      disabled={!editVideoPath}
                      className="w-20 rounded border border-border bg-bg-primary px-1.5 py-1 text-[10px] text-text-primary disabled:opacity-50" />
                  </label>}
                  {!aligned && <button type="button" disabled={!editVideoPath} onClick={() => void editKeyframe(frame.time)}
                    className="mt-1 flex min-h-7 items-center gap-1 text-[9px] text-accent-blue hover:text-text-primary disabled:opacity-50">
                    <WandSparkles size={11} /> Edit this source moment in Maestro
                  </button>}
                </div>
                <button type="button" aria-label={aligned ? 'Remove replacement character image' : `Remove frame ${index + 1}`} onClick={() => removeFrame(index)}
                  className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary">
                  <X size={13} />
                </button>
              </div>
            ))}
          </div>
        )}
        {frames.length < frameLimit && (
          <button type="button" disabled={uploadingFrame} onClick={() => frameFileRef.current?.click()}
            className="flex min-h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-border text-[10px] text-text-secondary hover:border-accent-blue/60 hover:text-text-primary disabled:opacity-50">
            <ImagePlus size={13} /> {uploadingFrame ? 'Uploading image…' : aligned ? 'Upload character image' : 'Add uploaded frame'}
          </button>
        )}
        {frameEditError && <p role="alert" className="text-[10px] text-red-400">{frameEditError}</p>}
        <input ref={frameFileRef} type="file" accept="image/*" className="hidden"
          aria-label={aligned ? 'Upload replacement character image' : 'Upload Remake reference frame'}
          onChange={event => {
            const file = event.currentTarget.files?.[0]
            if (file) void handleFrameUpload(file)
            event.currentTarget.value = ''
          }} />
      </div>

      <div className="flex items-start gap-2 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-2.5 py-2">
        <Film size={13} className="mt-0.5 shrink-0 text-emerald-400" />
        <p className="text-[10px] leading-snug text-text-secondary">The source soundtrack is retained in the generated clip.</p>
      </div>
    </div>
  )
}
