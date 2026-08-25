import { useEffect, useMemo, useRef } from 'react'
import { Maximize2, Pause, Play, RotateCcw, SkipBack, SkipForward } from 'lucide-react'
import type { EditorAsset, EditorTimelineItem, EditorTrack } from '../types'
import { activeEditorItems, editorProjectDuration, formatEditorTime } from './editorUtils'
import { useEditorStore } from './useEditorStore'

function syncMediaElement(
  element: HTMLMediaElement,
  item: EditorTimelineItem,
  playhead: number,
  playing: boolean,
  volume: number,
): void {
  const wantedTime = Math.max(0, item.source_in + (playhead - item.start) * item.speed)
  if (Number.isFinite(element.duration)) {
    const bounded = Math.min(Math.max(0, element.duration - 0.03), wantedTime)
    if (Math.abs(element.currentTime - bounded) > (playing ? 0.2 : 0.045)) element.currentTime = bounded
  } else if (Math.abs(element.currentTime - wantedTime) > 0.045) {
    element.currentTime = wantedTime
  }
  element.playbackRate = Math.max(0.1, Math.min(8, item.speed))
  element.volume = Math.max(0, Math.min(1, volume))
  if (playing) void element.play().catch(() => {})
  else element.pause()
}

function PreviewVisual({
  asset,
  item,
  track,
  playhead,
  playing,
  canvasWidth,
  canvasHeight,
}: {
  asset: EditorAsset
  item: EditorTimelineItem
  track: EditorTrack
  playhead: number
  playing: boolean
  canvasWidth: number
  canvasHeight: number
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const translateX = (item.transform?.x || 0) / Math.max(1, canvasWidth) * 100
  const translateY = (item.transform?.y || 0) / Math.max(1, canvasHeight) * 100
  const transform = `translate(${translateX}%, ${translateY}%) scale(${item.transform?.scale || 1}) rotate(${item.transform?.rotation || 0}deg)`

  useEffect(() => {
    const video = videoRef.current
    if (!video || asset.type !== 'video') return
    // Preview the same mix FFmpeg exports: an added music/voice track does not
    // implicitly silence audio embedded in visible video clips.
    video.muted = Boolean(item.muted) || track.muted
    syncMediaElement(video, item, playhead, playing, item.volume * (track.volume ?? 1))
  }, [asset.type, item, playhead, playing, track.muted, track.volume])

  const commonStyle = {
    objectFit: item.fit,
    opacity: item.opacity,
    transform,
  } as const

  if (asset.type === 'image') {
    return <img src={asset.url} alt="" className="absolute inset-0 h-full w-full" style={commonStyle} />
  }
  return (
    <video
      ref={videoRef}
      src={asset.url}
      className="absolute inset-0 h-full w-full"
      style={commonStyle}
      playsInline
      preload="auto"
    />
  )
}

function PreviewAudio({
  asset,
  item,
  track,
  playhead,
  playing,
}: {
  asset: EditorAsset
  item: EditorTimelineItem
  track: EditorTrack
  playhead: number
  playing: boolean
}) {
  const ref = useRef<HTMLAudioElement>(null)
  useEffect(() => {
    if (!ref.current) return
    syncMediaElement(ref.current, item, playhead, playing, item.volume * (track.volume ?? 1))
  }, [item, playhead, playing, track.volume])
  return <audio ref={ref} src={asset.url} preload="auto" />
}

export function EditorPreview() {
  const project = useEditorStore(state => state.project)
  const playhead = useEditorStore(state => state.playhead)
  const playing = useEditorStore(state => state.playing)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
  const setPlaying = useEditorStore(state => state.setPlaying)
  const canvasRef = useRef<HTMLDivElement>(null)
  const playheadRef = useRef(playhead)
  const duration = editorProjectDuration(project)
  const active = useMemo(
    () => project ? activeEditorItems(project, playhead) : [],
    [playhead, project],
  )
  const activeAudio = active.filter(entry => entry.track.type === 'audio')
  const activeVisual = active.filter(entry => entry.track.type === 'video')
  const activeText = active.filter(entry => entry.track.type === 'text')

  useEffect(() => {
    playheadRef.current = playhead
  }, [playhead])

  useEffect(() => {
    if (!playing || !project) return
    const startedAt = performance.now()
    const originalPlayhead = playheadRef.current
    let frame = 0
    const tick = (now: number) => {
      const next = originalPlayhead + (now - startedAt) / 1000
      if (next >= duration) {
        setPlayhead(duration)
        setPlaying(false)
        return
      }
      setPlayhead(next)
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [duration, playing, project, setPlayhead, setPlaying])

  if (!project) return <div className="flex min-h-0 flex-1 items-center justify-center text-xs text-text-muted">Opening Editor…</div>

  const seek = (value: number) => setPlayhead(Math.max(0, Math.min(duration, value)))
  const frameDuration = 1 / project.canvas.fps

  return (
    <section className="flex min-h-0 flex-1 flex-col bg-bg-primary">
      <div className="flex min-h-0 flex-1 items-center justify-center overflow-hidden p-3 md:p-5">
        <div
          ref={canvasRef}
          className="relative max-h-full max-w-full overflow-hidden rounded-md bg-black shadow-2xl ring-1 ring-white/5"
          style={{
            aspectRatio: `${project.canvas.width} / ${project.canvas.height}`,
            width: project.canvas.width >= project.canvas.height ? 'min(100%, 980px)' : 'min(52%, 520px)',
            height: project.canvas.width < project.canvas.height ? '100%' : 'auto',
            background: project.canvas.background,
          }}
          onDoubleClick={() => void canvasRef.current?.requestFullscreen?.()}
        >
          {activeVisual.map(({ item, track }) => {
            const asset = item.asset_id ? project.assets[item.asset_id] : null
            return asset ? (
              <PreviewVisual
                key={item.id}
                asset={asset}
                item={item}
                track={track}
                playhead={playhead}
                playing={playing}
                canvasWidth={project.canvas.width}
                canvasHeight={project.canvas.height}
              />
            ) : null
          })}
          {activeText.map(({ item }) => {
            const style = item.style || { x: 0, y: 0, font_size: 64, color: '#ffffff' }
            return (
              <div
                key={item.id}
                className="pointer-events-none absolute left-1/2 top-1/2 max-w-[88%] -translate-x-1/2 -translate-y-1/2 whitespace-pre-wrap rounded bg-black/30 px-3 py-1.5 text-center font-semibold leading-tight drop-shadow-lg"
                style={{
                  color: style.color,
                  fontSize: `clamp(12px, ${Math.max(1.1, style.font_size / project.canvas.height * 75)}vw, 92px)`,
                  marginLeft: `${style.x / project.canvas.width * 100}%`,
                  marginTop: `${style.y / project.canvas.height * 100}%`,
                  opacity: item.opacity,
                }}
              >
                {item.text || 'Title'}
              </div>
            )
          })}
          {activeAudio.map(({ item, track }) => {
            const asset = item.asset_id ? project.assets[item.asset_id] : null
            return asset ? <PreviewAudio key={item.id} asset={asset} item={item} track={track} playhead={playhead} playing={playing} /> : null
          })}
          {activeVisual.length === 0 && activeText.length === 0 && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center text-white/25">
              <Play size={28} />
              <span className="text-[10px]">Drop media onto the timeline</span>
            </div>
          )}
        </div>
      </div>

      <div className="flex h-11 shrink-0 items-center gap-2 border-t border-border bg-bg-secondary px-3 md:px-4">
        <button type="button" onClick={() => seek(0)} className="rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Go to start">
          <SkipBack size={14} />
        </button>
        <button type="button" onClick={() => seek(playhead - frameDuration)} className="hidden rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary sm:block" title="Previous frame">
          <RotateCcw size={13} />
        </button>
        <button
          type="button"
          onClick={() => {
            if (!playing && playhead >= duration && duration > 0) seek(0)
            setPlaying(!playing)
          }}
          disabled={duration <= 0}
          className="flex h-7 w-7 items-center justify-center rounded-full bg-text-primary text-bg-primary hover:opacity-90 disabled:opacity-30"
          title={playing ? 'Pause (Space)' : 'Play (Space)'}
        >
          {playing ? <Pause size={13} fill="currentColor" /> : <Play size={13} fill="currentColor" className="ml-0.5" />}
        </button>
        <button type="button" onClick={() => seek(playhead + frameDuration)} className="hidden rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary sm:block" title="Next frame">
          <SkipForward size={13} />
        </button>
        <span className="w-[88px] font-mono text-[10px] text-text-secondary">
          {formatEditorTime(playhead, true, project.canvas.fps)}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(frameDuration, duration)}
          step={frameDuration}
          value={Math.min(playhead, Math.max(frameDuration, duration))}
          onChange={event => seek(Number(event.target.value))}
          className="min-w-0 flex-1 accent-[var(--color-accent-blue)]"
          aria-label="Playhead"
        />
        <span className="hidden w-11 text-right font-mono text-[9px] text-text-muted sm:block">{formatEditorTime(duration)}</span>
        <button type="button" onClick={() => void canvasRef.current?.requestFullscreen?.()} className="rounded p-1.5 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Fullscreen preview">
          <Maximize2 size={13} />
        </button>
      </div>
    </section>
  )
}
