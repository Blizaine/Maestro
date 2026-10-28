import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Check,
  Film,
  Image as ImageIcon,
  Pause,
  Play,
  Volume2,
  X,
} from 'lucide-react'
import type { EditorAsset, EditorTimelineItem } from '../types'
import { useEditorMediaPreview } from './editorMediaPreview'
import { useEditorStore } from './useEditorStore'
import { useEditorDialog } from './useEditorDialog'

interface EditorTakeComparisonProps {
  item: EditorTimelineItem
  assets: Record<string, EditorAsset>
  activeAssetId: string
  takeAssetIds: string[]
  workspace: string
  canSelectActive: boolean
  onSetActiveTake: (assetId: string) => void
}

interface TakePlaybackState {
  sourceIn: number
  speed: number
}

interface ComparisonTake {
  assetId: string
  asset: EditorAsset
  number: number
  state: TakePlaybackState
}

function playbackStateFor(item: EditorTimelineItem, assetId: string): TakePlaybackState {
  const savedState = item.take_states?.[assetId]
  const sourceIn = savedState?.source_in ?? (assetId === item.asset_id ? item.source_in : 0)
  const speed = savedState?.speed ?? (assetId === item.asset_id ? item.speed : 1)
  return {
    sourceIn: Number.isFinite(sourceIn) ? Math.max(0, sourceIn) : 0,
    speed: Number.isFinite(speed) && speed > 0 ? speed : 1,
  }
}

function TakePlaceholder({ type, name }: { type: EditorAsset['type']; name: string }) {
  const Icon = type === 'image' ? ImageIcon : type === 'audio' ? AudioLines : Film
  return (
    <span className="grid h-full w-full place-items-center bg-gradient-to-br from-accent-blue/15 to-bg-tertiary" aria-label={`No preview for ${name}`}>
      <Icon size={20} className="text-text-muted" />
    </span>
  )
}

function VideoTakeThumbnail({ asset, workspace }: { asset: EditorAsset; workspace: string }) {
  const preview = useEditorMediaPreview(asset, workspace)
  if (asset.missing || !preview.data?.thumbnail_url) {
    return <TakePlaceholder type="video" name={asset.name} />
  }
  return (
    <span
      className="block h-full w-full bg-cover bg-left bg-no-repeat"
      role="img"
      aria-label={`Preview of ${asset.name}`}
      style={{ backgroundImage: `url(${preview.data.thumbnail_url})`, backgroundSize: '800% 100%' }}
    />
  )
}

function TakeThumbnail({ asset, workspace }: { asset: EditorAsset; workspace: string }) {
  const [imageFailed, setImageFailed] = useState(false)
  if (asset.type === 'video') return <VideoTakeThumbnail asset={asset} workspace={workspace} />
  if (asset.type === 'image' && !asset.missing && !imageFailed) {
    return <img src={asset.url} alt={`Preview of ${asset.name}`} loading="lazy" onError={() => setImageFailed(true)} className="h-full w-full object-cover" draggable={false} />
  }
  return <TakePlaceholder type={asset.type} name={asset.name} />
}

function formatSeconds(value: number): string {
  return `${value.toFixed(2)} s`
}

function mediaTimeAt(take: ComparisonTake, timelineTime: number): number {
  const mediaTime = take.state.sourceIn + timelineTime * take.state.speed
  return Number.isFinite(take.asset.duration) && take.asset.duration > 0
    ? Math.min(mediaTime, take.asset.duration)
    : mediaTime
}

function ComparisonMedia({
  take,
  side,
  muted,
  videoRef,
  onLoadedMetadata,
  onTimeUpdate,
  onEnded,
}: {
  take: ComparisonTake
  side: 'A' | 'B'
  muted: boolean
  videoRef: React.RefObject<HTMLVideoElement | null>
  onLoadedMetadata: () => void
  onTimeUpdate: () => void
  onEnded: () => void
}) {
  if (take.asset.type === 'image') {
    return (
      <img
        src={take.asset.url}
        alt={`${side}: ${take.asset.name}`}
        className="h-full w-full bg-black object-contain"
        draggable={false}
      />
    )
  }
  if (take.asset.type === 'video') {
    return (
      <video
        ref={videoRef}
        src={take.asset.url}
        aria-label={`${side}: ${take.asset.name}`}
        muted={muted || !take.asset.has_audio}
        playsInline
        preload="metadata"
        onLoadedMetadata={onLoadedMetadata}
        onTimeUpdate={onTimeUpdate}
        onEnded={onEnded}
        className="h-full w-full bg-black object-contain"
      />
    )
  }
  return (
    <div className="grid h-full w-full place-items-center bg-black/80 px-4 text-center text-[12px] text-text-muted">
      <span><AudioLines size={24} className="mx-auto mb-2" />Audio takes can be selected on the timeline, but this comparison supports video and image sources.</span>
    </div>
  )
}

function TakeComparisonDialog({
  item,
  current,
  alternative,
  canSelectActive,
  onSetActiveTake,
  onClose,
}: {
  item: EditorTimelineItem
  current: ComparisonTake
  alternative: ComparisonTake
  canSelectActive: boolean
  onSetActiveTake: (assetId: string) => void
  onClose: () => void
}) {
  const dialogId = useId()
  const dialogRef = useEditorDialog(true, onClose)
  const labelId = useId()
  const currentVideoRef = useRef<HTMLVideoElement>(null)
  const alternativeVideoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [position, setPosition] = useState(0)
  const videos = useMemo(() => [current, alternative], [alternative, current])
  const videoRefs = useMemo(() => [currentVideoRef, alternativeVideoRef], [])
  const masterAudioIndex = current.asset.type === 'video' && current.asset.has_audio
    ? 0
    : alternative.asset.type === 'video' && alternative.asset.has_audio ? 1 : -1
  const hasVideo = videos.some(take => take.asset.type === 'video')
  const availableDurations = videos
    .filter(take => take.asset.type === 'video' && take.asset.duration > 0)
    .map(take => Math.max(0, (take.asset.duration - take.state.sourceIn) / take.state.speed))
  const seekMax = Math.max(0, Math.min(item.duration, ...(availableDurations.length ? availableDurations : [item.duration])))
  const rangeMax = Math.max(0.01, seekMax)

  const setVideoTime = useCallback((index: number, timelineTime: number) => {
    const node = videoRefs[index].current
    if (!node) return
    const take = videos[index]
    const targetTime = mediaTimeAt(take, timelineTime)
    try {
      // Comparison previews synchronize native media element state imperatively;
      // playbackRate is not React-owned state and this runs only in an event/effect callback.
      // eslint-disable-next-line react-hooks/immutability -- Native video playback speed is an external media side effect.
      node.playbackRate = take.state.speed
      if (!Number.isFinite(node.currentTime) || Math.abs(node.currentTime - targetTime) > 0.04) {
        node.currentTime = targetTime
      }
    } catch {
      // The browser applies the source position once metadata is ready.
    }
  }, [videoRefs, videos])

  const seekTo = useCallback((timelineTime: number) => {
    const clamped = Math.max(0, Math.min(seekMax, timelineTime))
    setPosition(clamped)
    videos.forEach((take, index) => {
      if (take.asset.type === 'video') setVideoTime(index, clamped)
    })
  }, [seekMax, setVideoTime, videos])

  const pauseVideos = useCallback(() => {
    videoRefs.forEach(ref => ref.current?.pause())
    setPlaying(false)
  }, [videoRefs])

  useEffect(() => {
    seekTo(0)
    videoRefs.forEach(ref => ref.current?.pause())
    setPlaying(false)
    return () => {
      videoRefs.forEach(ref => ref.current?.pause())
    }
  }, [seekTo, videoRefs])

  const playVideos = () => {
    const playable = videos
      .map((take, index) => take.asset.type === 'video' ? videoRefs[index].current : null)
      .filter((video): video is HTMLVideoElement => Boolean(video))
    if (!playable.length || seekMax <= 0) return
    seekTo(position)
    setPlaying(true)
    playable.forEach(video => { void video.play().catch(() => setPlaying(false)) })
  }

  const clockIndex = videos[0].asset.type === 'video' ? 0 : 1
  const updateFromClock = () => {
    const clock = videoRefs[clockIndex].current
    if (!clock) return
    const take = videos[clockIndex]
    const timelineTime = Math.max(0, Math.min(seekMax, (clock.currentTime - take.state.sourceIn) / take.state.speed))
    setPosition(timelineTime)
    const otherIndex = 1 - clockIndex
    if (videos[otherIndex].asset.type === 'video') setVideoTime(otherIndex, timelineTime)
    if (seekMax > 0 && timelineTime >= seekMax - 0.04) pauseVideos()
  }

  return (
    <div
      ref={dialogRef}
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-3 backdrop-blur-sm sm:p-6"
      onMouseDown={event => { if (event.target === event.currentTarget) onClose() }}
    >
      <section
        id={dialogId}
        role="dialog"
        aria-modal="true"
        aria-labelledby={labelId}
        className="flex max-h-full w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-border bg-bg-secondary shadow-2xl"
      >
        <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
          <div className="min-w-0">
            <h2 id={labelId} className="text-[14px] font-semibold text-text-primary">Compare takes</h2>
            <p className="mt-0.5 truncate text-[11px] text-text-muted">{item.name}</p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close take comparison" title="Close comparison" className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border text-text-secondary hover:bg-bg-hover hover:text-text-primary">
            <X size={16} />
          </button>
        </header>

        <div className="grid min-h-0 grid-cols-1 gap-3 overflow-y-auto p-3 sm:grid-cols-2 sm:p-4">
          {videos.map((take, index) => {
            const side = index === 0 ? 'A' : 'B'
            const isAudioMaster = masterAudioIndex === index
            return (
              <article key={take.assetId} className="min-w-0 overflow-hidden rounded-xl border border-border bg-bg-tertiary">
                <div className="flex items-center justify-between gap-2 px-3 py-2">
                  <div className="min-w-0">
                    <div className="text-[11px] font-semibold text-text-primary">{side} · {index === 0 ? 'Current take' : `Take ${take.number}`}</div>
                    <div className="truncate text-[10px] text-text-muted" title={take.asset.name}>{take.asset.display_name || take.asset.name}</div>
                  </div>
                  <span className="shrink-0 rounded-md bg-bg-secondary px-2 py-1 font-mono text-[10px] text-text-secondary">
                    In {formatSeconds(take.state.sourceIn)} · {take.state.speed.toFixed(2)}×
                  </span>
                </div>
                <div className="aspect-video overflow-hidden bg-black">
                  <ComparisonMedia
                    take={take}
                    side={side}
                    muted={!isAudioMaster}
                    videoRef={videoRefs[index]}
                    onLoadedMetadata={() => setVideoTime(index, position)}
                    onTimeUpdate={index === clockIndex ? updateFromClock : () => undefined}
                    onEnded={pauseVideos}
                  />
                </div>
                <div className="flex items-center justify-between gap-2 px-3 py-2 text-[10px] text-text-muted">
                  <span className="capitalize">{take.asset.type} · {take.asset.width || '—'}×{take.asset.height || '—'}</span>
                  {isAudioMaster && <span className="inline-flex items-center gap-1 text-accent-blue"><Volume2 size={11} /> Audio master</span>}
                </div>
              </article>
            )
          })}
        </div>

        <footer className="space-y-3 border-t border-border px-4 py-3">
          {hasVideo && (
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={playing ? pauseVideos : playVideos}
                disabled={seekMax <= 0}
                aria-label={playing ? 'Pause comparison' : 'Play comparison'}
                className="flex h-9 shrink-0 items-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 text-[11px] font-medium text-text-primary hover:bg-bg-hover disabled:opacity-45"
              >
                {playing ? <Pause size={14} /> : <Play size={14} />}
                {playing ? 'Pause' : 'Play'}
              </button>
              <label htmlFor={`${dialogId}-seek`} className="sr-only">Comparison position</label>
              <input
                id={`${dialogId}-seek`}
                type="range"
                min={0}
                max={rangeMax}
                step={0.01}
                value={Math.min(position, rangeMax)}
                disabled={seekMax <= 0}
                onChange={event => seekTo(Number(event.target.value))}
                className="min-w-0 flex-1 accent-[var(--color-accent-blue)]"
              />
              <span className="shrink-0 font-mono text-[10px] text-text-secondary">{formatSeconds(position)} / {formatSeconds(seekMax)}</span>
            </div>
          )}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="inline-flex items-center gap-1.5 text-[10px] text-text-muted">
              <Check size={12} className="text-accent-blue" />
              {masterAudioIndex >= 0 ? 'Only the marked audio master is audible.' : 'No audio track in these sources.'}
            </p>
            <div className="flex items-center gap-2">
              <button type="button" onClick={onClose} className="rounded-lg border border-border px-3 py-2 text-[11px] text-text-secondary hover:bg-bg-hover">Close</button>
              <button
                type="button"
                disabled={!canSelectActive}
                onClick={() => { onSetActiveTake(alternative.assetId); onClose() }}
                title={canSelectActive ? `Make Take ${alternative.number} active on the timeline` : 'Unlock this track to change its active take'}
                className="rounded-lg bg-accent-blue px-3 py-2 text-[11px] font-semibold text-white hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-45"
              >
                Use Take {alternative.number} on timeline
              </button>
            </div>
          </div>
        </footer>
      </section>
    </div>
  )
}

function EditorTakeComparisonContent({
  item,
  assets,
  activeAssetId,
  takeAssetIds,
  workspace,
  canSelectActive,
  onSetActiveTake,
}: EditorTakeComparisonProps) {
  const [comparisonAssetId, setComparisonAssetId] = useState<string | null>(null)
  const currentAsset = assets[activeAssetId]
  const comparisonAsset = comparisonAssetId ? assets[comparisonAssetId] : undefined
  const comparisonIndex = comparisonAssetId ? takeAssetIds.indexOf(comparisonAssetId) : -1
  const canCompare = (asset: EditorAsset | undefined): asset is EditorAsset => Boolean(
    asset && !asset.missing && (asset.type === 'video' || asset.type === 'image'),
  )

  const currentNumber = Math.max(1, takeAssetIds.indexOf(activeAssetId) + 1)
  const currentTake: ComparisonTake | null = currentAsset ? {
    assetId: activeAssetId,
    asset: currentAsset,
    number: currentNumber,
    state: playbackStateFor(item, activeAssetId),
  } : null
  const alternativeTake: ComparisonTake | null = comparisonAsset && comparisonIndex >= 0 ? {
    assetId: comparisonAsset.id,
    asset: comparisonAsset,
    number: comparisonIndex + 1,
    state: playbackStateFor(item, comparisonAsset.id),
  } : null
  const radioName = `active-take-${item.id}`

  return (
    <section className="space-y-2 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Alternate takes">
      <div className="flex items-center justify-between gap-2">
        <div>
          <h3 className="text-[12px] font-semibold text-text-primary">Takes</h3>
          <p className="mt-0.5 text-[10px] text-text-muted">Select the source used by this clip.</p>
        </div>
        <span className="shrink-0 rounded-md bg-bg-secondary px-2 py-1 text-[10px] text-text-secondary">
          {currentAsset ? `Take ${currentNumber} active` : 'No active source'}
        </span>
      </div>

      <div role="radiogroup" aria-label="Choose active take" className="space-y-2">
        {takeAssetIds.map((assetId, index) => {
          const asset = assets[assetId]
          if (!asset) return null
          const isActive = assetId === activeAssetId
          const compareEnabled = currentAsset
            && assetId !== activeAssetId
            && canCompare(currentAsset)
            && canCompare(asset)
          return (
            <div key={assetId} className={`overflow-hidden rounded-lg border ${isActive ? 'border-accent-blue/60 bg-accent-blue/5' : 'border-border bg-bg-secondary'}`}>
              <label className="flex cursor-pointer items-center gap-2 p-2 focus-within:ring-2 focus-within:ring-inset focus-within:ring-accent-blue/70">
                <input
                  type="radio"
                  name={radioName}
                  value={assetId}
                  checked={isActive}
                  disabled={!canSelectActive}
                  onChange={() => { if (!isActive) onSetActiveTake(assetId) }}
                  aria-label={`Take ${index + 1}: ${asset.display_name || asset.name}`}
                  className="h-4 w-4 shrink-0 accent-[var(--color-accent-blue)]"
                />
                <span className="h-14 w-24 shrink-0 overflow-hidden rounded-md bg-black/25">
                  <TakeThumbnail asset={asset} workspace={workspace} />
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5 text-[11px] font-medium text-text-primary">
                    <span>Take {index + 1}</span>
                    {isActive && <span className="rounded bg-accent-blue/15 px-1.5 py-0.5 text-[9px] text-accent-blue">Current</span>}
                    {asset.missing && <span className="rounded bg-red-500/10 px-1.5 py-0.5 text-[9px] text-red-300">Offline</span>}
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-text-muted" title={asset.display_name || asset.name}>{asset.display_name || asset.name}</span>
                  <span className="mt-0.5 block text-[10px] capitalize text-text-muted">{asset.type} · {asset.width || '—'}×{asset.height || '—'}</span>
                </span>
              </label>
              {assetId !== activeAssetId && (
                <div className="flex justify-end border-t border-border/70 px-2 py-1.5">
                  <button
                    type="button"
                    disabled={!compareEnabled}
                    onClick={() => { useEditorStore.getState().setPlaying(false); setComparisonAssetId(assetId) }}
                    title={compareEnabled ? `Compare current take with Take ${index + 1}` : 'Comparison is available for online video and image sources'}
                    aria-label={`Compare current take with Take ${index + 1}`}
                    className="rounded-md border border-border px-2.5 py-1.5 text-[10px] font-medium text-text-secondary hover:border-accent-blue/50 hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Compare A/B
                  </button>
                </div>
              )}
            </div>
          )
        })}
      </div>

      <p className="text-[10px] leading-relaxed text-text-muted">Each take keeps its own source in and speed. Switching takes preserves the clip edit and other sources.</p>

      {currentTake && alternativeTake && canCompare(currentTake.asset) && canCompare(alternativeTake.asset) && (
        <TakeComparisonDialog
          key={`${item.id}:${currentTake.assetId}:${alternativeTake.assetId}`}
          item={item}
          current={currentTake}
          alternative={alternativeTake}
          canSelectActive={canSelectActive}
          onSetActiveTake={onSetActiveTake}
          onClose={() => setComparisonAssetId(null)}
        />
      )}
    </section>
  )
}

export function EditorTakeComparison(props: EditorTakeComparisonProps) {
  const key = `${props.item.id}:${props.activeAssetId}`
  return <EditorTakeComparisonContent key={key} {...props} />
}
