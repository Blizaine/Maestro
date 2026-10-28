import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Clapperboard,
  Film,
  FolderOpen,
  Heart,
  Image as ImageIcon,
  Loader2,
  Plus,
  List,
  Grid2X2,
  Eye,
  RefreshCw,
  Search,
  Sparkles,
  Type,
  Upload,
} from 'lucide-react'
import type { EditorAsset, EditorMediaType, PipelineListItem } from '../types'
import { useEditorMediaPreview } from './editorMediaPreview'
import { useEditorStore } from './useEditorStore'
import { EditorSourceViewer } from './EditorSourceViewer'
import { EditorAnimatedTitleDialog } from './EditorAnimatedTitleDialog'

type Filter = 'all' | 'favorites' | 'director' | EditorMediaType
const MEDIA_BATCH_SIZE = 80

function VideoThumbnail({ asset, workspace }: { asset: EditorAsset; workspace: string }) {
  const preview = useEditorMediaPreview(asset, workspace)
  if (preview.data?.thumbnail_url) {
    return (
      <span
        className="block h-full w-full bg-cover bg-left bg-no-repeat"
        style={{
          backgroundImage: `url(${preview.data.thumbnail_url})`,
          // The cached thumbnail is an eight-frame horizontal contact sheet.
          // Show its first frame without creating another video decoder.
          backgroundSize: '800% 100%',
        }}
      />
    )
  }
  return (
    <span className="grid h-full w-full place-items-center bg-gradient-to-br from-accent-blue/15 to-bg-tertiary">
      {preview.loading
        ? <Loader2 size={15} className="animate-spin text-text-muted" />
        : <Film size={16} className="text-text-muted" />}
    </span>
  )
}

function LazyVideoThumbnail({ asset, workspace }: { asset: EditorAsset; workspace: string }) {
  const rootRef = useRef<HTMLSpanElement>(null)
  const [visible, setVisible] = useState(() => typeof IntersectionObserver === 'undefined')
  useEffect(() => {
    const root = rootRef.current
    if (!root || visible) return
    if (typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setVisible(true)
        observer.disconnect()
      }
    }, { rootMargin: '160px' })
    observer.observe(root)
    return () => observer.disconnect()
  }, [visible])
  return (
    <span ref={rootRef} className="block h-full w-full">
      {visible
        ? <VideoThumbnail asset={asset} workspace={workspace} />
        : <span className="grid h-full w-full place-items-center bg-bg-tertiary"><Film size={16} className="text-text-muted/70" /></span>}
    </span>
  )
}

function MediaPreview({ asset, workspace }: { asset: EditorAsset; workspace: string }) {
  if (asset.type === 'image') {
    return <img src={asset.url} alt="" className="h-full w-full object-cover" loading="lazy" />
  }
  if (asset.type === 'video') {
    return <LazyVideoThumbnail asset={asset} workspace={workspace} />
  }
  return (
    <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-accent-blue/20 to-accent-cool-to/10">
      <AudioLines size={20} className="text-accent-blue" />
    </div>
  )
}

function DirectorRunPreview({ run, compact }: { run: PipelineListItem; compact: boolean }) {
  const [failed, setFailed] = useState(false)
  return (
    <div className={`${compact ? 'aspect-video' : 'h-16'} relative flex w-full items-center justify-center overflow-hidden bg-black/35`}>
      {run.thumbnail_url && !failed ? (
        <img
          src={run.thumbnail_url}
          alt=""
          className="h-full w-full object-cover"
          loading="lazy"
          onError={() => setFailed(true)}
        />
      ) : (
        <Clapperboard size={compact ? 28 : 22} className="text-accent-warm/80" />
      )}
      <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-white/75">
        {run.pipeline_type === 'music_video' ? 'Music video' : run.pipeline_type === 'short_film_story' ? 'Short film' : 'Director'}
      </span>
      <span className="absolute bottom-1.5 right-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white/75">
        {run.clip_count} shot{run.clip_count === 1 ? '' : 's'}
      </span>
    </div>
  )
}

export function EditorMediaBin({ compact = false }: { compact?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState<Filter>('all')
  const [workspaceFilter, setWorkspaceFilter] = useState('all')
  const [visibleLimit, setVisibleLimit] = useState(MEDIA_BATCH_SIZE)
  const [uploading, setUploading] = useState(false)
  const [view, setView] = useState<'grid' | 'list'>('grid')
  const [thumbnailSize, setThumbnailSize] = useState(120)
  const [usedOnly, setUsedOnly] = useState(false)
  const [sourceAsset, setSourceAsset] = useState<EditorAsset | null>(null)
  const [titleOpen, setTitleOpen] = useState(false)
  const project = useEditorStore(state => state.project)
  const cards = compact && view === 'grid'
  const library = useEditorStore(state => state.library)
  const projectWorkspace = useEditorStore(state => state.project?.workspace || 'default')
  const libraryWorkspaces = useEditorStore(state => state.libraryWorkspaces)
  const directorRuns = useEditorStore(state => state.directorRuns)
  const directorImportingId = useEditorStore(state => state.directorImportingId)
  const selectedTrackId = useEditorStore(state => state.selectedTrackId)
  const refreshLibrary = useEditorStore(state => state.refreshLibrary)
  const uploadMedia = useEditorStore(state => state.uploadMedia)
  const addMedia = useEditorStore(state => state.addMedia)
  const importDirectorRun = useEditorStore(state => state.importDirectorRun)
  const addTitle = useEditorStore(state => state.addTitle)
  const usedAssets = useMemo(() => new Set((project?.tracks || []).flatMap(track => track.items)
    .map(item => project?.assets[item.asset_id || '']).filter((asset): asset is EditorAsset => Boolean(asset))
    .map(asset => `${asset.origin}:${asset.workspace || ''}:${asset.name}`)), [project])
  const isUsed = (asset: EditorAsset) => usedAssets.has(`${asset.origin}:${asset.workspace || ''}:${asset.name}`)

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (filter === 'director') return []
    return library.filter(asset => (
      (workspaceFilter === 'all' || asset.workspace === workspaceFilter)
      && (!usedOnly || usedAssets.has(`${asset.origin}:${asset.workspace || ''}:${asset.name}`))
      && (filter === 'all' || (filter === 'favorites' ? asset.favorite : asset.type === filter))
      && (!query || asset.name.toLowerCase().includes(query))
    ))
  }, [filter, library, search, workspaceFilter, usedOnly, usedAssets])
  const visibleDirectorRuns = useMemo(() => {
    if (filter !== 'director') return []
    const query = search.trim().toLowerCase()
    return directorRuns.filter(run => (
      (workspaceFilter === 'all' || run.workspace === workspaceFilter)
      && (!query || run.scene_description.toLowerCase().includes(query))
    ))
  }, [directorRuns, filter, search, workspaceFilter])
  const visibleItems = filter === 'director' ? visibleDirectorRuns : visible
  const rendered = visible.slice(0, visibleLimit)
  const renderedDirectorRuns = visibleDirectorRuns.slice(0, visibleLimit)

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return
    setUploading(true)
    try {
      for (const file of Array.from(files)) {
        await uploadMedia(file, selectedTrackId || undefined)
      }
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <section className={`flex min-h-0 flex-col bg-bg-secondary ${compact ? 'h-full' : 'w-[260px] shrink-0 border-r border-border'}`}>
      <div data-testid="editor-media-bin-header" className="shrink-0 space-y-1.5 border-b border-border p-2">
        <div className="flex min-w-0 items-center gap-1">
          <h2 className="min-w-0 flex-1 whitespace-nowrap text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Media</h2>
          <button type="button" onClick={() => setUsedOnly(value => !value)} aria-pressed={usedOnly}
            aria-label="Used in project" title="Used in project"
            className={`min-h-9 shrink-0 rounded-md border px-1 text-[10px] font-medium ${usedOnly ? 'border-accent-blue/50 bg-accent-blue/10 text-accent-blue' : 'border-border text-text-secondary'}`}>Used</button>
          {/* Global range styling outranks width utilities, so inline dimensions keep this slider inside the compact header and touch-sized. */}
          {cards && <input type="range" min={100} max={220} step={10} value={thumbnailSize} aria-label="Media thumbnail size"
            title="Thumbnail size" onChange={event => setThumbnailSize(Number(event.target.value))}
            className="shrink-0 accent-[var(--color-accent-blue)]"
            style={{
              width: 44,
              height: 36,
              background: 'linear-gradient(to bottom, transparent calc(50% - 2px), var(--color-border) calc(50% - 2px), var(--color-border) calc(50% + 2px), transparent calc(50% + 2px))',
            }} />}
          <div className="flex shrink-0 items-center gap-0.5">
            <button type="button" aria-label="Grid view" aria-pressed={view === 'grid'} onClick={() => setView('grid')}
              className={`grid min-h-9 min-w-8 place-items-center rounded-md ${view === 'grid' ? 'bg-bg-active text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`}><Grid2X2 size={14} /></button>
            <button type="button" aria-label="List view" aria-pressed={view === 'list'} onClick={() => setView('list')}
              className={`grid min-h-9 min-w-8 place-items-center rounded-md ${view === 'list' ? 'bg-bg-active text-accent-blue' : 'text-text-muted hover:bg-bg-hover'}`}><List size={14} /></button>
          </div>
          <button type="button" aria-label="Refresh media" onClick={() => void refreshLibrary()}
            className="grid min-h-9 min-w-8 shrink-0 place-items-center rounded-md text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Refresh media">
            <RefreshCw size={13} />
          </button>
        </div>
        <div className="grid grid-cols-2 gap-1">
          <div className="relative min-w-0">
            <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
            <input
              aria-label="Search media"
              value={search}
              onChange={event => {
                setSearch(event.target.value)
                setVisibleLimit(MEDIA_BATCH_SIZE)
              }}
              placeholder="Search media"
              className="h-8 w-full min-w-0 rounded-lg border border-border bg-bg-tertiary pl-7 pr-2 text-[11px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue/60"
            />
          </div>
          <label className="relative flex min-w-0 items-center">
            <FolderOpen size={11} className="pointer-events-none absolute left-2 text-text-muted" />
            <select
              value={workspaceFilter}
              onChange={event => {
                setWorkspaceFilter(event.target.value)
                setVisibleLimit(MEDIA_BATCH_SIZE)
              }}
              className="h-8 w-full min-w-0 appearance-none rounded-lg border border-border bg-bg-tertiary pl-6 pr-5 text-[10px] text-text-secondary outline-none focus:border-accent-blue/60"
              aria-label="Filter media by workspace"
            >
              <option value="all">All workspaces</option>
              {libraryWorkspaces.map(workspace => (
                <option key={workspace} value={workspace}>{workspace === 'default' ? 'Default workspace' : workspace}</option>
              ))}
              <option value="__uploads__">Uploads</option>
            </select>
            <span className="pointer-events-none absolute right-2 text-[10px] text-text-muted">▾</span>
          </label>
        </div>
        <div data-testid="editor-media-bin-toolbar" className="grid grid-cols-3 gap-1">
          <button type="button" aria-label="Upload media" title="Upload media" onClick={() => inputRef.current?.click()}
            disabled={uploading}
            className="flex min-h-9 min-w-0 items-center justify-center gap-1 rounded-lg border border-dashed border-border-light bg-bg-tertiary px-1.5 text-[11px] font-medium text-text-secondary hover:border-accent-blue/50 hover:text-accent-blue disabled:opacity-50">
            {uploading ? <Loader2 size={12} className="shrink-0 animate-spin" /> : <Upload size={12} className="shrink-0" />}<span className="truncate">Upload</span>
          </button>
          <button type="button" aria-label="Add text title" title="Add text title" onClick={() => addTitle(undefined, selectedTrackId || undefined)}
            className="flex min-h-9 min-w-0 items-center justify-center gap-1 rounded-lg border border-border bg-bg-tertiary px-1.5 text-[11px] font-medium text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue">
            <Type size={12} className="shrink-0" /><span className="truncate">Text</span>
          </button>
          <button type="button" aria-label="Create animated title" title="Create animated title" disabled={!project} onClick={() => setTitleOpen(true)}
            className="flex min-h-9 min-w-0 items-center justify-center gap-1 rounded-lg border border-accent-blue/30 bg-accent-blue/5 px-1.5 text-[11px] font-medium text-accent-blue disabled:opacity-40">
            <Sparkles size={12} className="shrink-0" /><span className="truncate">Animated</span>
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="video/*,image/*,audio/*,.mkv,.webm,.mov,.mp4,.wav,.mp3,.m4a,.flac,.ogg"
            multiple
            className="hidden"
            onChange={event => void handleFiles(event.target.files)}
          />
        </div>
        <div data-testid="editor-media-bin-filters" className="grid grid-cols-6 gap-1">
          {(['all', 'video', 'image', 'audio', 'director', 'favorites'] as Filter[]).map(value => {
            const label = value === 'all' ? 'All media' : value[0].toUpperCase() + value.slice(1)
            const Icon = value === 'video'
              ? Film
              : value === 'image'
                ? ImageIcon
                : value === 'audio'
                  ? AudioLines
                  : value === 'director'
                    ? Clapperboard
                  : value === 'favorites'
                    ? Heart
                    : null
            return (
              <button
                key={value}
                type="button"
                aria-label={label}
                aria-pressed={filter === value}
                onClick={() => {
                  setFilter(value)
                  setVisibleLimit(MEDIA_BATCH_SIZE)
                }}
                className={`grid min-h-9 min-w-0 place-items-center rounded-md px-1 text-[11px] capitalize transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-blue/60 ${filter === value ? value === 'favorites' ? 'bg-red-500/15 text-red-400' : 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
                title={label}
              >
                {Icon ? <Icon size={12} className="mx-auto" fill={value === 'favorites' && filter === 'favorites' ? 'currentColor' : 'none'} /> : 'All'}
              </button>
            )
          })}
        </div>
      </div>

      <div className={`min-h-0 flex-1 overflow-y-auto p-2.5 ${cards ? 'grid auto-rows-max gap-2.5' : 'space-y-1.5'}`}
        data-testid="editor-media-bin-scroller"
        style={cards ? { gridTemplateColumns: `repeat(auto-fill, minmax(${thumbnailSize}px, 1fr))` } : undefined}>
        {renderedDirectorRuns.map(run => (
          <article
            key={run.id}
            data-testid="editor-media-director-run"
            className={`group overflow-hidden rounded-xl border border-accent-warm/20 bg-gradient-to-br from-accent-warm/10 via-bg-tertiary to-bg-tertiary transition-all hover:border-accent-warm/40 ${compact ? 'shadow-sm' : 'p-2'}`}
          >
            <DirectorRunPreview
              key={run.thumbnail_url || 'director-thumbnail-pending'}
              run={run}
              compact={compact}
            />
            <div className={compact ? 'p-2' : 'pt-2'}>
              <div className="line-clamp-2 text-[11px] font-medium leading-snug text-text-primary" title={run.scene_description}>
                {run.scene_description || `Director run ${run.id}`}
              </div>
              <div className="mt-1 flex items-center justify-between gap-1 text-[10px] text-text-muted">
                <span className="truncate">{run.workspace} · {run.status}</span>
                <button
                  type="button"
                  onClick={() => void importDirectorRun(run.id)}
                  disabled={Boolean(directorImportingId)}
                  className="flex shrink-0 items-center gap-1 rounded-md bg-accent-warm/15 px-1.5 py-1 text-[11px] font-medium text-accent-warm hover:bg-accent-warm/25 disabled:cursor-wait disabled:opacity-45"
                  title="Add every Director shot and its master soundtrack at the playhead"
                >
                  {directorImportingId === run.id ? <Loader2 size={9} className="animate-spin" /> : <Plus size={9} />}
                  Add run
                </button>
              </div>
            </div>
          </article>
        ))}
        {rendered.map(asset => (
          <article
            key={`${asset.origin}:${asset.name}`}
            data-testid="editor-media-asset"
            draggable
            onDragStart={event => {
              event.dataTransfer.effectAllowed = 'copy'
              const payload = JSON.stringify(asset)
              event.dataTransfer.setData('application/x-maestro-editor-asset', payload)
              // Safari may omit custom MIME payloads across complex drag targets.
              // Keep an identifiable plain-text fallback for remote Mac sessions.
              event.dataTransfer.setData('text/plain', `maestro-editor-asset:${payload}`)
            }}
            onDoubleClick={() => void addMedia(asset, undefined, selectedTrackId || undefined)}
            className={`group overflow-hidden rounded-xl border border-border bg-bg-tertiary transition-colors hover:border-border-light hover:bg-bg-hover ${cards ? 'shadow-sm' : 'flex items-center gap-2 p-1.5'}`}
          >
            <div className={`relative shrink-0 overflow-hidden bg-media-canvas ${cards ? 'aspect-video w-full' : 'h-12 w-16 rounded-md'}`}>
              <MediaPreview asset={asset} workspace={projectWorkspace} />
              <span className="absolute bottom-1 left-1 max-w-[80%] truncate rounded bg-black/65 px-1 py-0.5 text-[10px] text-white/80" title={asset.workspace || asset.origin}>
                {asset.workspace === '__uploads__' ? 'Uploads' : asset.workspace || asset.origin}
              </span>
              {asset.favorite && <Heart size={9} fill="currentColor" className="absolute right-1 top-1 text-red-400 drop-shadow" />}
              {isUsed(asset) && <span className="absolute right-1 bottom-1 rounded bg-black/70 px-1 text-[10px] text-white">Used</span>}
            </div>
            <div className={`min-w-0 ${cards ? 'p-2' : 'flex-1'}`}>
              <div className="truncate text-[11px] font-medium text-text-secondary" title={asset.display_name || asset.name}>{asset.display_name || asset.name}</div>
              <div className="mt-0.5 flex items-center justify-between gap-1 text-[11px] text-text-muted">
                <span className="capitalize">{asset.type}</span>
                {asset.duration > 0 && <span>{asset.duration.toFixed(1)}s</span>}
                <button type="button" onClick={() => setSourceAsset(asset)} title="Preview and mark In/Out" aria-label={`Preview ${asset.display_name || asset.name}`}
                  className="ml-auto rounded p-1.5 hover:bg-bg-hover hover:text-text-primary"><Eye size={13} /></button>
                <button
                  type="button"
                  onClick={() => void addMedia(asset, undefined, selectedTrackId || undefined)}
                  className="rounded bg-accent-blue/10 p-1 text-accent-blue opacity-80 hover:bg-accent-blue/20 group-hover:opacity-100"
                  title="Add at playhead"
                >
                  <Plus size={10} />
                </button>
              </div>
            </div>
          </article>
        ))}
        {visibleItems.length > (filter === 'director' ? renderedDirectorRuns.length : rendered.length) && (
          <button
            type="button"
            onClick={() => setVisibleLimit(limit => limit + MEDIA_BATCH_SIZE)}
            className="col-span-full w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-[11px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue"
          >
            Show more · {visibleItems.length - (filter === 'director' ? renderedDirectorRuns.length : rendered.length)} remaining
          </button>
        )}
        {visibleItems.length === 0 && (
          <div className="col-span-full flex min-h-40 flex-col items-center justify-center gap-2 px-4 text-center text-text-muted">
            <Film size={24} className="opacity-50" />
            <p className="text-[10px] leading-relaxed">
              {filter === 'favorites'
                ? 'No favorite media matches these filters.'
                : filter === 'director'
                  ? 'No saved Director runs match these filters. Completed Director projects appear here as editable shot packages.'
                : 'Upload media or generate it in Studio. Workspace media appears here automatically.'}
            </p>
          </div>
        )}
      </div>
      {sourceAsset && <EditorSourceViewer key={sourceAsset.id} asset={sourceAsset} onClose={() => setSourceAsset(null)} />}
      <EditorAnimatedTitleDialog open={titleOpen} onClose={() => setTitleOpen(false)} />
    </section>
  )
}
