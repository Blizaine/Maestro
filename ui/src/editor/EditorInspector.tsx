import { useId, useRef, useState } from 'react'
import {
  AlignCenter,
  AlignLeft,
  AlignRight,
  AudioLines,
  Clapperboard,
  Eye,
  EyeOff,
  FolderOpen,
  Lock,
  Loader2,
  RotateCcw,
  SlidersHorizontal,
  Trash2,
  TriangleAlert,
  Unlock,
  Volume2,
  VolumeX,
  WandSparkles,
} from 'lucide-react'
import type { EditorAIReturnMode, EditorAIRoundTripTool, EditorAnimatedProperty, EditorTextStyle, EditorTimelineItem, EditorTransitionType } from '../types'
import { editorCanvasLabel } from './editorUtils'
import { DEFAULT_EDITOR_FONT, EDITOR_FONT_OPTIONS, editorFontStack } from './editorFonts'
import { EditorAnimationControls } from './EditorAnimationControls'
import { EditorColorControls } from './EditorColorControls'
import { EditorTakeComparison } from './EditorTakeComparison'
import { getApplicableEditorProperties } from './editorAnimation'
import { useEditorStore } from './useEditorStore'

function FieldLabel({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  const className = 'text-[10px] font-medium uppercase tracking-wider text-text-muted'
  return htmlFor
    ? <label htmlFor={htmlFor} className={className}>{children}</label>
    : <div className={className}>{children}</div>
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 0.1,
  onChange,
}: {
  label: string
  value: number
  min?: number
  max?: number
  step?: number
  onChange: (value: number) => void
}) {
  const inputId = useId()
  return (
    <div className="space-y-1">
      <FieldLabel htmlFor={inputId}>{label}</FieldLabel>
      <input
        id={inputId}
        type="number"
        value={Number.isFinite(value) ? Number(value.toFixed(3)) : 0}
        min={min}
        max={max}
        step={step}
        onChange={event => onChange(Number(event.target.value))}
        className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
      />
    </div>
  )
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
}) {
  const inputId = useId()
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <FieldLabel htmlFor={inputId}>{label}</FieldLabel>
        <span className="font-mono text-[10px] text-text-secondary">{value.toFixed(step < 0.1 ? 2 : 1)}</span>
      </div>
      <input id={inputId} type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))} className="w-full accent-[var(--color-accent-blue)]" />
    </div>
  )
}

function DirectorRerunPanel({ item }: { item: EditorTimelineItem }) {
  const source = item.director
  const prompt = source?.draft_video_prompt ?? (
    source?.window_prompts?.length
      ? source.window_prompts.join('\n\n')
      : source?.video_prompt || ''
  )
  const rerunDirectorClip = useEditorStore(state => state.rerunDirectorClip)
  const setDirectorDraftPrompt = useEditorStore(state => state.setDirectorDraftPrompt)
  const clearDirectorRerunError = useEditorStore(state => state.clearDirectorRerunError)
  const directorRerunItemId = useEditorStore(state => state.directorRerunItemId)
  const rerunError = useEditorStore(state => {
    const error = state.directorRerunErrors[item.id]
    return error?.projectId === state.project?.id ? error.message : null
  })
  if (!source) return null
  return (
    <div className="overflow-hidden rounded-xl border border-accent-blue/25 bg-accent-blue/5">
      <div className="flex items-center justify-between gap-2 border-b border-accent-blue/15 px-3 py-2.5">
        <span className="flex items-center gap-1.5 text-[10px] font-medium text-accent-blue">
          <Clapperboard size={12} /> Director shot {source.clip_index + 1}
        </span>
        <span className="max-w-[92px] truncate font-mono text-[7px] text-text-muted" title={source.pipeline_id}>
          {source.pipeline_id}
        </span>
      </div>
      <div className="space-y-2 p-2.5">
        <div>
          <FieldLabel>Video prompt draft</FieldLabel>
          <textarea
            aria-label="Video prompt draft"
            value={prompt}
            onChange={event => setDirectorDraftPrompt(item.id, event.target.value)}
            rows={6}
            className="mt-1 w-full resize-y rounded-lg border border-border bg-bg-tertiary px-2 py-2 text-[9px] leading-relaxed text-text-primary outline-none focus:border-accent-blue/60"
          />
        </div>
        <button
          type="button"
          onClick={() => void rerunDirectorClip(item.id, prompt)}
          disabled={Boolean(directorRerunItemId)}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-accent-blue px-2 py-2 text-[9px] font-semibold text-white hover:brightness-110 disabled:cursor-wait disabled:opacity-45"
        >
          {directorRerunItemId === item.id
            ? <Loader2 size={11} className="animate-spin" />
            : <RotateCcw size={11} />}
          {directorRerunItemId === item.id ? 'Regenerating in Director…' : 'Re-run Director shot'}
        </button>
        {rerunError && (
          <div
            role="alert"
            onPointerDown={event => event.stopPropagation()}
            onMouseDown={event => event.stopPropagation()}
            onClick={event => event.stopPropagation()}
            className="flex items-start justify-between gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-2 text-[8px] leading-relaxed text-red-200"
          >
            <span className="min-w-0 break-words">Director rerun failed: {rerunError}</span>
            <button
              type="button"
              onClick={event => {
                event.stopPropagation()
                clearDirectorRerunError(item.id)
              }}
              className="shrink-0 underline underline-offset-2 opacity-80 hover:opacity-100"
            >
              Dismiss
            </button>
          </div>
        )}
        <p className="text-[8px] leading-relaxed text-text-muted">
          Uses the original Director model, references, soundtrack timing, LoRAs, and settings. The new render becomes the active take; the current take stays available.
        </p>
      </div>
    </div>
  )
}

function EditorInspectorContent({
  compact = false,
  aiReturnMode,
  setAiReturnMode,
}: {
  compact?: boolean
  aiReturnMode: EditorAIReturnMode
  setAiReturnMode: (mode: EditorAIReturnMode) => void
}) {
  const [aiExpanded, setAiExpanded] = useState(false)
  const [activeClipTab, setActiveClipTab] = useState<'basic' | 'audio' | 'motion' | 'ai'>('basic')
  const relinkInputRef = useRef<HTMLInputElement>(null)
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([])
  const inspectorId = useId()
  const project = useEditorStore(state => state.project)
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const selectedItemIds = useEditorStore(state => state.selectedItemIds)
  const selectedTrackId = useEditorStore(state => state.selectedTrackId)
  const playhead = useEditorStore(state => state.playhead)
  const setPlayhead = useEditorStore(state => state.setPlayhead)
  const updateItem = useEditorStore(state => state.updateItem)
  const detachSelectedAudio = useEditorStore(state => state.detachSelectedAudio)
  const deleteSelected = useEditorStore(state => state.deleteSelected)
  const renameTrack = useEditorStore(state => state.renameTrack)
  const setTrackVolume = useEditorStore(state => state.setTrackVolume)
  const setTrackZIndex = useEditorStore(state => state.setTrackZIndex)
  const toggleTrackMute = useEditorStore(state => state.toggleTrackMute)
  const toggleTrackLock = useEditorStore(state => state.toggleTrackLock)
  const removeTrack = useEditorStore(state => state.removeTrack)
  const setCanvas = useEditorStore(state => state.setCanvas)
  const setExportSettings = useEditorStore(state => state.setExportSettings)
  const setActiveTake = useEditorStore(state => state.setActiveTake)
  const relinkMedia = useEditorStore(state => state.relinkMedia)
  const beginAIRoundTrip = useEditorStore(state => state.beginAIRoundTrip)
  const cancelAIRoundTrip = useEditorStore(state => state.cancelAIRoundTrip)
  const roundTrip = useEditorStore(state => state.roundTrip)

  const location = project && selectedItemId
    ? project.tracks.flatMap(track => track.items.map(item => ({ track, item }))).find(entry => entry.item.id === selectedItemId)
    : null
  const item = location?.item
  const track = location?.track
  const selectedTrack = project?.tracks.find(candidate => candidate.id === selectedTrackId) || null
  const asset = item?.asset_id ? project?.assets[item.asset_id] : null
  const takeIds = item
    ? Array.from(new Set([...(item.take_asset_ids || []), item.asset_id || '']))
        .filter(id => Boolean(project?.assets[id]))
    : []
  const hasAudioControls = Boolean(track && (
    track.type === 'audio'
    || (track.type === 'video' && asset?.type === 'video' && asset.has_audio)
  ))
  const animatedProperties: EditorAnimatedProperty[] = track
    ? getApplicableEditorProperties(track.type, asset?.type, Boolean(asset?.has_audio))
    : []
  const hasAITab = Boolean(track?.type === 'video' && (asset?.type === 'video' || item?.director))
  const clipTabs: Array<{ id: 'basic' | 'audio' | 'motion' | 'ai'; label: string }> = [
    { id: 'basic', label: 'Basic' },
    ...(hasAudioControls ? [{ id: 'audio' as const, label: 'Audio' }] : []),
    ...(animatedProperties.length ? [{ id: 'motion' as const, label: 'Motion' }] : []),
    ...(hasAITab ? [{ id: 'ai' as const, label: 'AI' }] : []),
  ]
  const visibleClipTab = clipTabs.some(tab => tab.id === activeClipTab) ? activeClipTab : 'basic'
  const aiTools: Array<{ id: EditorAIRoundTripTool; label: string; description: string }> = [
    { id: 'retake', label: 'Retake', description: 'Regenerate the selected range.' },
    { id: 'edit_anything', label: 'Edit Anything', description: 'Prompt-driven changes.' },
    { id: 'recast', label: 'Recast', description: 'Replace a character.' },
    { id: 'repaint', label: 'Repaint', description: 'Restyle or replace regions.' },
    { id: 'outpaint', label: 'Outpaint', description: 'Extend the frame.' },
    { id: 'upscale', label: 'Upscale', description: 'Create a higher-quality take.' },
    { id: 'film_grain', label: 'Film Grain', description: 'Add a cinematic grain finish.' },
    { id: 'revoice', label: 'Revoice', description: 'Replace the spoken voice.' },
  ]
  const aiRoundTripActive = Boolean(roundTrip && !['completed', 'failed'].includes(roundTrip.status))

  const patchTransform = (key: keyof EditorTimelineItem['transform'], value: number) => {
    if (!item) return
    updateItem(item.id, { transform: { ...item.transform, [key]: value } })
  }
  const patchStyle = (key: keyof EditorTextStyle, value: number | string) => {
    if (!item) return
    updateItem(item.id, {
      style: {
        x: item.style?.x ?? 0,
        y: item.style?.y ?? 0,
        font_family: item.style?.font_family ?? DEFAULT_EDITOR_FONT,
        font_size: item.style?.font_size ?? 64,
        color: item.style?.color ?? '#ffffff',
        background_color: item.style?.background_color ?? '#000000',
        background_opacity: item.style?.background_opacity ?? 0.32,
        text_align: item.style?.text_align ?? 'center',
        [key]: value,
      },
    })
  }

  const confirmTrackRemoval = (trackId: string, trackName: string, itemCount: number) => {
    if (itemCount > 0 && !window.confirm(
      `Remove “${trackName}” and its ${itemCount} clip${itemCount === 1 ? '' : 's'}? This can be undone with Ctrl+Z.`,
    )) return
    removeTrack(trackId)
  }

  if (!project) return null

  return (
    <aside className={`min-h-0 overflow-y-auto bg-bg-secondary ${compact ? 'h-full' : 'w-[300px] shrink-0 border-l border-border'}`}>
      <div className="sticky top-0 z-10 flex h-10 items-center gap-2 border-b border-border bg-bg-secondary/95 px-3 backdrop-blur">
        <SlidersHorizontal size={12} className="text-accent-blue" />
        <h2 className="text-[12px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          {item ? 'Clip inspector' : selectedTrack ? 'Track inspector' : 'Project inspector'}
        </h2>
      </div>

      {item && track ? (
        <>
          <div role="tablist" aria-label="Clip inspector sections" className="sticky top-10 z-10 grid border-b border-border bg-bg-secondary/95 p-2 backdrop-blur" style={{ gridTemplateColumns: `repeat(${clipTabs.length}, minmax(0, 1fr))` }}>
            {clipTabs.map((tab, index) => (
              <button
                key={tab.id}
                ref={node => { tabRefs.current[index] = node }}
                type="button"
                role="tab"
                id={`${inspectorId}-tab-${tab.id}`}
                aria-selected={visibleClipTab === tab.id}
                aria-controls={`${inspectorId}-panel-${tab.id}`}
                tabIndex={visibleClipTab === tab.id ? 0 : -1}
                onClick={() => setActiveClipTab(tab.id)}
                onKeyDown={event => {
                  const nextIndex = event.key === 'ArrowRight'
                    ? (index + 1) % clipTabs.length
                    : event.key === 'ArrowLeft'
                      ? (index - 1 + clipTabs.length) % clipTabs.length
                      : event.key === 'Home' ? 0
                        : event.key === 'End' ? clipTabs.length - 1 : index
                  if (nextIndex === index) return
                  event.preventDefault()
                  setActiveClipTab(clipTabs[nextIndex].id)
                  tabRefs.current[nextIndex]?.focus()
                }}
                className={`rounded-lg px-2 py-2 text-[11px] font-medium transition-colors ${visibleClipTab === tab.id ? 'bg-bg-active text-text-primary' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="p-3">
            {visibleClipTab === 'basic' && (
              <div id={`${inspectorId}-panel-basic`} role="tabpanel" aria-labelledby={`${inspectorId}-tab-basic`} className="space-y-4">
                <div className="space-y-2">
                  <button type="button" aria-pressed={Boolean(item.pinned)} disabled={track.locked}
                    onClick={() => updateItem(item.id, { pinned: !item.pinned })}
                    className={`w-full rounded-lg border px-3 py-2 text-xs disabled:opacity-40 ${item.pinned ? 'border-accent-blue/50 bg-accent-blue/10 text-accent-blue' : 'border-border text-text-secondary hover:bg-bg-hover'}`}>
                    {item.pinned ? 'Pinned · protected from auto-edit' : 'Pin this clip for auto-edit'}
                  </button>
                  <FieldLabel>Timing</FieldLabel>
                  <div className="grid grid-cols-2 gap-2">
                    <NumberField label="Start" value={item.start} min={0} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { start: value })} />
                    <NumberField label="Duration" value={item.duration} min={1 / project.canvas.fps} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { duration: value })} />
                    {track.type !== 'text' && <NumberField label="Source in" value={item.source_in} min={0} step={1 / project.canvas.fps} onChange={value => updateItem(item.id, { source_in: value })} />}
                    {track.type !== 'text' && <NumberField label="Speed" value={item.speed} min={0.1} max={8} step={0.05} onChange={value => updateItem(item.id, { speed: value })} />}
                  </div>
                </div>

                {track.type !== 'audio' && (
                  <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Transform controls">
                    <div className="flex items-center justify-between gap-2">
                      <FieldLabel>Transform</FieldLabel>
                      <button
                        type="button"
                        onClick={() => updateItem(item.id, { transform: { x: 0, y: 0, scale: 1, rotation: 0 } })}
                        title="Reset position, scale, and rotation"
                        aria-label="Reset transform"
                        className="inline-flex items-center gap-1.5 rounded-md border border-border px-2 py-1.5 text-[10px] font-medium text-text-secondary hover:bg-bg-hover"
                      >
                        <RotateCcw size={11} /> Reset
                      </button>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <NumberField label="X" value={item.transform.x} step={1} onChange={value => patchTransform('x', value)} />
                      <NumberField label="Y" value={item.transform.y} step={1} onChange={value => patchTransform('y', value)} />
                    </div>
                    <SliderField label="Scale" value={item.transform.scale} min={0.1} max={3} step={0.01} onChange={value => patchTransform('scale', value)} />
                    {track.type !== 'text' && <SliderField label="Rotation" value={item.transform.rotation} min={-180} max={180} step={1} onChange={value => patchTransform('rotation', value)} />}
                    {track.type === 'video' && (
                      <div>
                        <FieldLabel>Source fit</FieldLabel>
                        <div className="mt-1 grid grid-cols-2 gap-1 rounded-lg bg-bg-secondary p-1">
                          {(['contain', 'cover'] as const).map(fit => (
                            <button
                              key={fit}
                              type="button"
                              onClick={() => updateItem(item.id, { fit })}
                              aria-pressed={item.fit === fit}
                              title={fit === 'contain' ? 'Show the full source inside the canvas' : 'Fill the canvas and crop overflow'}
                              className={`rounded-md py-2 text-[11px] capitalize ${item.fit === fit ? 'bg-bg-active text-text-primary' : 'text-text-muted hover:text-text-secondary'}`}
                            >
                              {fit}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </section>
                )}

                {track.type === 'video' && (asset?.type === 'video' || asset?.type === 'image') && (
                  <EditorColorControls
                    color={item.color}
                    disabled={track.locked}
                    onChange={color => updateItem(item.id, { color })}
                  />
                )}

                {takeIds.length > 1 && (
                  <EditorTakeComparison
                    item={item}
                    assets={project.assets}
                    activeAssetId={item.asset_id || ''}
                    takeAssetIds={takeIds}
                    workspace={project.workspace || 'default'}
                    canSelectActive={!track.locked}
                    onSetActiveTake={assetId => setActiveTake(item.id, assetId)}
                  />
                )}

                <div>
                  <FieldLabel htmlFor={`${inspectorId}-clip-name`}>Name</FieldLabel>
                  <input
                    id={`${inspectorId}-clip-name`}
                    value={item.name}
                    onChange={event => updateItem(item.id, { name: event.target.value.slice(0, 140) })}
                    className="mt-1 w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
                  />
                  <p className="mt-1 truncate text-[10px] text-text-muted">{track.name}{asset ? ` · ${asset.width || '—'}×${asset.height || '—'}` : ''}</p>
                </div>

                {selectedItemIds.length > 1 && (
                  <div className="rounded-lg border border-accent-warm/25 bg-accent-warm/5 px-2.5 py-2 text-[11px] leading-relaxed text-accent-warm">
                    {selectedItemIds.length} clips selected. Move, duplicate, copy, delete, or link them as a group; this inspector edits the primary clip.
                  </div>
                )}

                {asset?.missing && (
                  <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-3">
                    <div className="flex items-start gap-2">
                      <TriangleAlert size={14} className="mt-0.5 shrink-0 text-red-300" />
                      <div className="min-w-0 flex-1">
                        <div className="text-[12px] font-medium text-red-200">Source media is offline</div>
                        <p className="mt-1 break-all text-[10px] leading-relaxed text-red-200/70">{asset.name}</p>
                      </div>
                    </div>
                    <input
                      ref={relinkInputRef}
                      type="file"
                      accept={asset.type === 'audio' ? 'audio/*' : asset.type === 'image' ? 'image/*' : 'video/*'}
                      aria-label="Choose replacement source file"
                      className="hidden"
                      onChange={event => {
                        const file = event.target.files?.[0]
                        event.target.value = ''
                        if (file) void relinkMedia(asset.id, file)
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => relinkInputRef.current?.click()}
                      className="mt-2 flex w-full items-center justify-center gap-1.5 rounded-lg border border-red-400/25 bg-bg-secondary py-2.5 text-[11px] font-medium text-red-100 hover:border-red-300/45"
                    >
                      <FolderOpen size={13} /> Relink source file
                    </button>
                  </div>
                )}

                {track.type === 'text' && (
                  <div className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
                    <div className="space-y-1">
                      <FieldLabel htmlFor={`${inspectorId}-text`}>Text</FieldLabel>
                      <textarea
                        id={`${inspectorId}-text`}
                        value={item.text || ''}
                        onChange={event => updateItem(item.id, { text: event.target.value })}
                        rows={3}
                        className="w-full resize-y rounded-md border border-border bg-bg-secondary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
                      />
                    </div>
                    <div className="space-y-1">
                      <FieldLabel htmlFor={`${inspectorId}-font`}>Font</FieldLabel>
                      <select
                        id={`${inspectorId}-font`}
                        value={item.style?.font_family || DEFAULT_EDITOR_FONT}
                        onChange={event => patchStyle('font_family', event.target.value)}
                        className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
                        style={{ fontFamily: editorFontStack(item.style?.font_family) }}
                      >
                        {EDITOR_FONT_OPTIONS.map(option => (
                          <option key={option.value} value={option.value} style={{ fontFamily: option.stack }}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <NumberField label="Font size" value={item.style?.font_size ?? 64} min={8} max={400} step={1} onChange={value => patchStyle('font_size', value)} />
                      <div className="space-y-1">
                        <FieldLabel htmlFor={`${inspectorId}-text-color`}>Color</FieldLabel>
                        <input id={`${inspectorId}-text-color`} type="color" value={item.style?.color || '#ffffff'} onChange={event => patchStyle('color', event.target.value)} className="h-9 w-full rounded-md border border-border bg-bg-secondary p-1" />
                      </div>
                      <NumberField label="Text X" value={item.style?.x ?? 0} step={1} onChange={value => patchStyle('x', value)} />
                      <NumberField label="Text Y" value={item.style?.y ?? 0} step={1} onChange={value => patchStyle('y', value)} />
                    </div>
                    <div role="group" aria-label="Text alignment" className="grid grid-cols-3 gap-1 rounded-lg bg-bg-secondary p-1">
                      {([
                        ['left', AlignLeft],
                        ['center', AlignCenter],
                        ['right', AlignRight],
                      ] as const).map(([alignment, Icon]) => (
                        <button
                          key={alignment}
                          type="button"
                          onClick={() => patchStyle('text_align', alignment)}
                          aria-pressed={(item.style?.text_align || 'center') === alignment}
                          className={`flex justify-center rounded-md py-2 ${
                            (item.style?.text_align || 'center') === alignment
                              ? 'bg-bg-active text-text-primary'
                              : 'text-text-muted hover:text-text-secondary'
                          }`}
                          title={`${alignment} aligned`}
                          aria-label={`${alignment} aligned`}
                        >
                          <Icon size={14} />
                        </button>
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <FieldLabel htmlFor={`${inspectorId}-text-background`}>Background</FieldLabel>
                        <input id={`${inspectorId}-text-background`} type="color" value={item.style?.background_color || '#000000'} onChange={event => patchStyle('background_color', event.target.value)} className="h-9 w-full rounded-md border border-border bg-bg-secondary p-1" />
                      </div>
                      <SliderField label="Background opacity" value={item.style?.background_opacity ?? 0.32} min={0} max={1} step={0.01} onChange={value => patchStyle('background_opacity', value)} />
                    </div>
                  </div>
                )}

                <div className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
                  {track.type !== 'audio' && <SliderField label="Opacity" value={item.opacity} min={0} max={1} step={0.01} onChange={value => updateItem(item.id, { opacity: value })} />}
                  <div className="grid grid-cols-2 gap-2">
                    <div className="space-y-1">
                      <FieldLabel htmlFor={`${inspectorId}-transition-in`}>Transition in</FieldLabel>
                      <select
                        id={`${inspectorId}-transition-in`}
                        value={item.transition_in || ((item.fade_in || 0) > 0 ? 'dissolve' : 'none')}
                        onChange={event => {
                          const transition = event.target.value as EditorTransitionType
                          updateItem(item.id, {
                            transition_in: transition,
                            fade_in: transition === 'none' ? 0 : Math.max(item.fade_in || 0, Math.min(0.5, item.duration / 2)),
                          })
                        }}
                        className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-primary outline-none"
                      >
                        <option value="none">None</option>
                        <option value="dissolve">Dissolve</option>
                        {track.type === 'video' && <option value="fade_black">Fade through black</option>}
                      </select>
                    </div>
                    <div className="space-y-1">
                      <FieldLabel htmlFor={`${inspectorId}-transition-out`}>Transition out</FieldLabel>
                      <select
                        id={`${inspectorId}-transition-out`}
                        value={item.transition_out || ((item.fade_out || 0) > 0 ? 'dissolve' : 'none')}
                        onChange={event => {
                          const transition = event.target.value as EditorTransitionType
                          updateItem(item.id, {
                            transition_out: transition,
                            fade_out: transition === 'none' ? 0 : Math.max(item.fade_out || 0, Math.min(0.5, item.duration / 2)),
                          })
                        }}
                        className="w-full rounded-md border border-border bg-bg-secondary px-2 py-2 text-[11px] text-text-primary outline-none"
                      >
                        <option value="none">None</option>
                        <option value="dissolve">Dissolve</option>
                        {track.type === 'video' && <option value="fade_black">Fade through black</option>}
                      </select>
                    </div>
                  </div>
                  <SliderField label="Transition in duration" value={item.fade_in || 0} min={0} max={Math.max(0.1, item.duration)} step={0.05} onChange={value => updateItem(item.id, { fade_in: Math.min(item.duration, value), transition_in: value > 0 ? item.transition_in === 'fade_black' ? 'fade_black' : 'dissolve' : 'none' })} />
                  <SliderField label="Transition out duration" value={item.fade_out || 0} min={0} max={Math.max(0.1, item.duration)} step={0.05} onChange={value => updateItem(item.id, { fade_out: Math.min(item.duration, value), transition_out: value > 0 ? item.transition_out === 'fade_black' ? 'fade_black' : 'dissolve' : 'none' })} />
                </div>

                <button type="button" onClick={() => updateItem(item.id, { disabled: !item.disabled })} className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-border bg-bg-tertiary py-2.5 text-[11px] text-text-secondary hover:bg-bg-hover">
                  {item.disabled ? <Eye size={13} /> : <EyeOff size={13} />} {item.disabled ? 'Enable clip' : 'Disable clip'}
                </button>
                <button type="button" onClick={deleteSelected} className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-red-500/20 bg-red-500/5 py-2.5 text-[11px] text-red-400 hover:bg-red-500/10">
                  <Trash2 size={13} /> Remove clip
                </button>
              </div>
            )}

            {visibleClipTab === 'audio' && (
              <div id={`${inspectorId}-panel-audio`} role="tabpanel" aria-labelledby={`${inspectorId}-tab-audio`} className="space-y-4">
                <section className="space-y-3 rounded-xl border border-border bg-bg-tertiary p-3" aria-label="Clip audio controls">
                  <SliderField label="Clip volume" value={item.volume} min={0} max={4} step={0.01} onChange={value => updateItem(item.id, { volume: value })} />
                  <button
                    type="button"
                    onClick={() => updateItem(item.id, { muted: !item.muted })}
                    aria-pressed={Boolean(item.muted)}
                    className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-border bg-bg-secondary py-2.5 text-[11px] text-text-secondary hover:bg-bg-hover"
                  >
                    {item.muted ? <Volume2 size={13} /> : <VolumeX size={13} />} {item.muted ? 'Unmute clip audio' : 'Mute clip audio'}
                  </button>
                  <p className="text-[10px] leading-relaxed text-text-muted">Adjust this clip’s sound without changing the source file.</p>
                </section>
                {track.type === 'video' && asset?.has_audio && !item.muted && (
                  <button type="button" onClick={detachSelectedAudio} className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-accent-blue/20 bg-accent-blue/5 py-2.5 text-[11px] text-accent-blue hover:bg-accent-blue/10">
                    <AudioLines size={13} /> Detach audio to its own track
                  </button>
                )}
              </div>
            )}

            {visibleClipTab === 'motion' && (
              <div id={`${inspectorId}-panel-motion`} role="tabpanel" aria-labelledby={`${inspectorId}-tab-motion`} className="space-y-4">
                <EditorAnimationControls
                  key={item.id}
                  item={item}
                  properties={animatedProperties}
                  absolutePlayhead={playhead}
                  fps={project.canvas.fps}
                  disabled={track.locked}
                  onUpdate={patch => updateItem(item.id, patch)}
                  onNavigate={setPlayhead}
                />
              </div>
            )}

            {visibleClipTab === 'ai' && (
              <div id={`${inspectorId}-panel-ai`} role="tabpanel" aria-labelledby={`${inspectorId}-tab-ai`} className="space-y-4">
                {track.type === 'video' && item.director && (
                  <DirectorRerunPanel key={`${item.id}:${item.asset_id || ''}`} item={item} />
                )}

                {track.type === 'video' && asset?.type === 'video' && (
                  <div className="overflow-hidden rounded-xl border border-accent-warm/25 bg-accent-warm/5">
                    <button
                      type="button"
                      onClick={() => setAiExpanded(value => !value)}
                      aria-expanded={aiExpanded}
                      className="flex w-full items-center justify-between px-3 py-3 text-[12px] font-medium text-accent-warm hover:bg-accent-warm/10"
                    >
                      <span className="flex items-center gap-1.5"><WandSparkles size={14} /> Edit clip with Maestro AI</span>
                      <span className="text-[11px]">{aiExpanded ? 'Hide tools' : 'Choose a tool'}</span>
                    </button>
                    {aiExpanded && (
                      <div className="space-y-3 border-t border-accent-warm/15 p-3">
                        <div role="group" aria-label="AI result placement" className="grid grid-cols-2 gap-1 rounded-lg bg-bg-tertiary p-1">
                          {(['alternate', 'replace'] as EditorAIReturnMode[]).map(mode => (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => setAiReturnMode(mode)}
                              aria-pressed={aiReturnMode === mode}
                              className={`rounded-md px-2 py-2 text-[11px] ${aiReturnMode === mode ? 'bg-bg-active text-text-primary' : 'text-text-muted hover:text-text-secondary'}`}
                            >
                              {mode === 'alternate' ? 'Add alternate take' : 'Replace on timeline'}
                            </button>
                          ))}
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          {aiTools.map(tool => (
                            <button
                              key={tool.id}
                              type="button"
                              onClick={() => void beginAIRoundTrip(tool.id, aiReturnMode)}
                              disabled={aiRoundTripActive}
                              className="rounded-lg border border-border bg-bg-secondary px-2.5 py-2.5 text-left hover:border-accent-warm/40 hover:bg-bg-hover disabled:cursor-not-allowed disabled:opacity-35"
                              title={aiRoundTripActive ? 'Finish or cancel the current AI round trip first.' : tool.description}
                            >
                              <span className="block text-[11px] font-medium text-text-primary">{tool.label}</span>
                              <span className="mt-0.5 block text-[10px] leading-snug text-text-muted">{tool.description}</span>
                            </button>
                          ))}
                        </div>
                        <p className="text-[10px] leading-relaxed text-text-muted">Maestro tracks the next queued generation and returns its output to this exact clip automatically.</p>
                      </div>
                    )}
                    {roundTrip?.itemId === item.id && (
                      <div className={`flex items-center justify-between gap-2 border-t px-3 py-2.5 text-[10px] ${roundTrip.status === 'failed' ? 'border-red-500/20 text-red-300' : roundTrip.status === 'completed' ? 'border-emerald-500/20 text-emerald-300' : 'border-accent-blue/20 text-accent-blue'}`}>
                        <span className="truncate">{roundTrip.status === 'armed' ? 'Waiting for generation…' : roundTrip.status === 'completed' ? 'AI result returned to Editor' : roundTrip.status === 'failed' ? roundTrip.error : `${roundTrip.status} in universal queue…`}</span>
                        <button type="button" onClick={cancelAIRoundTrip} className="shrink-0 underline opacity-75 hover:opacity-100">Dismiss</button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      ) : selectedTrack ? (
        <div className="space-y-4 p-3">
          <div>
            <FieldLabel htmlFor={`${inspectorId}-track-name`}>Track name</FieldLabel>
            <input
              id={`${inspectorId}-track-name`}
              value={selectedTrack.name}
              onChange={event => renameTrack(selectedTrack.id, event.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[12px] text-text-primary outline-none focus:border-accent-blue/60"
            />
            <p className="mt-1 text-[10px] capitalize text-text-muted">{selectedTrack.type} · {selectedTrack.items.length} clips</p>
          </div>
          {selectedTrack.type !== 'text' && (
            <SliderField label="Track volume" value={selectedTrack.volume ?? 1} min={0} max={2} step={0.01} onChange={value => setTrackVolume(selectedTrack.id, value)} />
          )}
          {selectedTrack.type !== 'audio' && (
            <NumberField label="Layer order" value={selectedTrack.z_index} min={-100} max={100} step={1} onChange={value => setTrackZIndex(selectedTrack.id, value)} />
          )}
          <div className="grid grid-cols-2 gap-2">
            <button type="button" onClick={() => toggleTrackMute(selectedTrack.id)} className={`flex items-center justify-center gap-1.5 rounded-lg border py-2 text-[9px] ${selectedTrack.muted ? 'border-red-500/30 bg-red-500/10 text-red-300' : 'border-border bg-bg-tertiary text-text-secondary hover:bg-bg-hover'}`}>
              {selectedTrack.muted ? <VolumeX size={11} /> : <Volume2 size={11} />} {selectedTrack.muted ? 'Muted' : 'Audible'}
            </button>
            <button type="button" onClick={() => toggleTrackLock(selectedTrack.id)} className={`flex items-center justify-center gap-1.5 rounded-lg border py-2 text-[9px] ${selectedTrack.locked ? 'border-accent-blue/30 bg-accent-blue/10 text-accent-blue' : 'border-border bg-bg-tertiary text-text-secondary hover:bg-bg-hover'}`}>
              {selectedTrack.locked ? <Lock size={11} /> : <Unlock size={11} />} {selectedTrack.locked ? 'Locked' : 'Unlocked'}
            </button>
          </div>
          <button
            type="button"
            onClick={() => confirmTrackRemoval(selectedTrack.id, selectedTrack.name, selectedTrack.items.length)}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-red-500/20 bg-red-500/5 py-2 text-[10px] text-red-400 hover:bg-red-500/10"
            title={selectedTrack.items.length ? 'Remove track and its clips' : 'Remove track'}
          >
            <Trash2 size={11} /> Remove track{selectedTrack.items.length ? ` + ${selectedTrack.items.length} clip${selectedTrack.items.length === 1 ? '' : 's'}` : ''}
          </button>
        </div>
      ) : (
        <div className="space-y-4 p-3">
          <div className="rounded-lg border border-border bg-bg-tertiary p-2.5">
            <div className="text-[10px] font-medium text-text-primary">{editorCanvasLabel(project.canvas.width, project.canvas.height)} canvas</div>
            <div className="mt-0.5 text-[8px] text-text-muted">{project.canvas.width} × {project.canvas.height} · {project.canvas.fps} fps</div>
          </div>
          <div className="space-y-2">
            <FieldLabel>Canvas</FieldLabel>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Width" value={project.canvas.width} min={64} max={7680} step={2} onChange={width => setCanvas({ width: Math.round(width / 2) * 2 })} />
              <NumberField label="Height" value={project.canvas.height} min={64} max={4320} step={2} onChange={height => setCanvas({ height: Math.round(height / 2) * 2 })} />
              <NumberField label="FPS" value={project.canvas.fps} min={1} max={120} step={1} onChange={fps => setCanvas({ fps })} />
              <div className="space-y-1">
                <FieldLabel htmlFor={`${inspectorId}-canvas-background`}>Background</FieldLabel>
                <input id={`${inspectorId}-canvas-background`} type="color" value={project.canvas.background} onChange={event => setCanvas({ background: event.target.value })} className="h-9 w-full rounded-md border border-border bg-bg-tertiary p-1" />
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <FieldLabel>Export</FieldLabel>
            <label htmlFor={`${inspectorId}-export-quality`} className="block text-[10px] font-medium uppercase tracking-wider text-text-muted">Export quality</label>
            <select id={`${inspectorId}-export-quality`} value={project.export.quality} onChange={event => setExportSettings({ quality: event.target.value as 'draft' | 'balanced' | 'high' })} className="w-full rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[12px] text-text-primary outline-none">
              <option value="draft">Draft · fastest</option>
              <option value="balanced">Balanced</option>
              <option value="high">High quality</option>
            </select>
            <label className="flex items-center justify-between rounded-md border border-border bg-bg-tertiary px-2 py-2 text-[9px] text-text-secondary">
              Include audio
              <input type="checkbox" checked={project.export.include_audio} onChange={event => setExportSettings({ include_audio: event.target.checked })} className="accent-[var(--color-accent-blue)]" />
            </label>
          </div>
          <p className="text-[9px] leading-relaxed text-text-muted">Select a clip to edit timing, framing, speed, volume, opacity, and text. All edits remain non-destructive until export.</p>
        </div>
      )}
    </aside>
  )
}

export function EditorInspector({ compact = false }: { compact?: boolean }) {
  const selectedItemId = useEditorStore(state => state.selectedItemId)
  const [aiReturnMode, setAiReturnMode] = useState<EditorAIReturnMode>('alternate')
  return (
    <EditorInspectorContent
      key={selectedItemId ?? 'no-selected-clip'}
      compact={compact}
      aiReturnMode={aiReturnMode}
      setAiReturnMode={setAiReturnMode}
    />
  )
}
