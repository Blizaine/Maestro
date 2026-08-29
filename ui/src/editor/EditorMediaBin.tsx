import { useMemo, useRef, useState } from 'react'
import {
  AudioLines,
  Clapperboard,
  Film,
  FolderOpen,
  Heart,
  Image as ImageIcon,
  Loader2,
  Plus,
  RefreshCw,
  Search,
  Type,
  Upload,
} from 'lucide-react'
import type { EditorAsset, EditorMediaType, PipelineListItem } from '../types'
import { useEditorStore } from './useEditorStore'

type Filter = 'all' | 'favorites' | 'director' | EditorMediaType
const MEDIA_BATCH_SIZE = 80

function MediaPreview({ asset }: { asset: EditorAsset }) {
  if (asset.type === 'image') {
    return <img src={asset.url} alt="" className="h-full w-full object-cover" loading="lazy" />
  }
  if (asset.type === 'video') {
    return <video src={asset.url} className="h-full w-full object-cover" muted preload="metadata" />
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
      <span className="absolute left-1.5 top-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[7px] uppercase tracking-wide text-white/75">
        {run.pipeline_type === 'music_video' ? 'Music video' : run.pipeline_type === 'short_film_story' ? 'Short film' : 'Director'}
      </span>
      <span className="absolute bottom-1.5 right-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[7px] text-white/75">
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
  const library = useEditorStore(state => state.library)
  const libraryWorkspaces = useEditorStore(state => state.libraryWorkspaces)
  const directorRuns = useEditorStore(state => state.directorRuns)
  const directorImportingId = useEditorStore(state => state.directorImportingId)
  const selectedTrackId = useEditorStore(state => state.selectedTrackId)
  const refreshLibrary = useEditorStore(state => state.refreshLibrary)
  const uploadMedia = useEditorStore(state => state.uploadMedia)
  const addMedia = useEditorStore(state => state.addMedia)
  const importDirectorRun = useEditorStore(state => state.importDirectorRun)
  const addTitle = useEditorStore(state => state.addTitle)

  const visible = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (filter === 'director') return []
    return library.filter(asset => (
      (workspaceFilter === 'all' || asset.workspace === workspaceFilter)
      && (filter === 'all' || (filter === 'favorites' ? asset.favorite : asset.type === filter))
      && (!query || asset.name.toLowerCase().includes(query))
    ))
  }, [filter, library, search, workspaceFilter])
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
      <div className="space-y-2 border-b border-border p-2.5">
        <div className="flex items-center justify-between">
          <h2 className="text-[11px] font-semibold uppercase tracking-[0.12em] text-text-secondary">Media</h2>
          <button type="button" onClick={() => void refreshLibrary()} className="rounded p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary" title="Refresh media">
            <RefreshCw size={12} />
          </button>
        </div>
        <div className="relative">
          <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-text-muted" />
          <input
            value={search}
            onChange={event => {
              setSearch(event.target.value)
              setVisibleLimit(MEDIA_BATCH_SIZE)
            }}
            placeholder="Search media"
            className="w-full rounded-lg border border-border bg-bg-tertiary py-1.5 pl-7 pr-2 text-[10px] text-text-primary outline-none placeholder:text-text-muted focus:border-accent-blue/60"
          />
        </div>
        <label className="relative flex items-center">
          <FolderOpen size={11} className="pointer-events-none absolute left-2.5 text-text-muted" />
          <select
            value={workspaceFilter}
            onChange={event => {
              setWorkspaceFilter(event.target.value)
              setVisibleLimit(MEDIA_BATCH_SIZE)
            }}
            className="w-full appearance-none rounded-lg border border-border bg-bg-tertiary py-1.5 pl-7 pr-7 text-[9px] text-text-secondary outline-none focus:border-accent-blue/60"
            aria-label="Filter media by workspace"
          >
            <option value="all">All workspaces</option>
            {libraryWorkspaces.map(workspace => (
              <option key={workspace} value={workspace}>{workspace === 'default' ? 'Default workspace' : workspace}</option>
            ))}
            <option value="__uploads__">Uploads</option>
          </select>
          <span className="pointer-events-none absolute right-2.5 text-[8px] text-text-muted">▾</span>
        </label>
        <div className="grid grid-cols-6 gap-1">
          {(['all', 'video', 'image', 'audio', 'director', 'favorites'] as Filter[]).map(value => {
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
                onClick={() => {
                  setFilter(value)
                  setVisibleLimit(MEDIA_BATCH_SIZE)
                }}
                className={`rounded-md px-1 py-1 text-[9px] capitalize transition-colors ${filter === value ? value === 'favorites' ? 'bg-red-500/15 text-red-400' : 'bg-accent-blue/15 text-accent-blue' : 'text-text-muted hover:bg-bg-hover hover:text-text-secondary'}`}
                title={value === 'favorites' ? 'Favorites' : value}
              >
                {Icon ? <Icon size={10} className="mx-auto" fill={value === 'favorites' && filter === 'favorites' ? 'currentColor' : 'none'} /> : 'All'}
              </button>
            )
          })}
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            className="flex items-center justify-center gap-1 rounded-lg border border-dashed border-border-light bg-bg-tertiary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/50 hover:text-accent-blue disabled:opacity-50"
          >
            {uploading ? <Loader2 size={11} className="animate-spin" /> : <Upload size={11} />} Upload
          </button>
          <button
            type="button"
            onClick={() => addTitle(undefined, selectedTrackId || undefined)}
            className="flex items-center justify-center gap-1 rounded-lg border border-border bg-bg-tertiary px-2 py-1.5 text-[9px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue"
          >
            <Type size={11} /> Title
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
      </div>

      <div className={`min-h-0 flex-1 overflow-y-auto p-2.5 ${compact ? 'grid auto-rows-max grid-cols-2 gap-2.5' : 'space-y-1.5'}`}>
        {renderedDirectorRuns.map(run => (
          <article
            key={run.id}
            className={`group overflow-hidden rounded-xl border border-accent-warm/20 bg-gradient-to-br from-accent-warm/10 via-bg-tertiary to-bg-tertiary transition-all hover:border-accent-warm/40 ${compact ? 'shadow-sm' : 'p-2'}`}
          >
            <DirectorRunPreview
              key={run.thumbnail_url || 'director-thumbnail-pending'}
              run={run}
              compact={compact}
            />
            <div className={compact ? 'p-2' : 'pt-2'}>
              <div className="line-clamp-2 text-[9px] font-medium leading-snug text-text-primary" title={run.scene_description}>
                {run.scene_description || `Director run ${run.id}`}
              </div>
              <div className="mt-1 flex items-center justify-between gap-1 text-[7px] text-text-muted">
                <span className="truncate">{run.workspace} · {run.status}</span>
                <button
                  type="button"
                  onClick={() => void importDirectorRun(run.id)}
                  disabled={Boolean(directorImportingId)}
                  className="flex shrink-0 items-center gap-1 rounded-md bg-accent-warm/15 px-1.5 py-1 text-[8px] font-medium text-accent-warm hover:bg-accent-warm/25 disabled:cursor-wait disabled:opacity-45"
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
            className={`group overflow-hidden rounded-xl border border-border bg-bg-tertiary transition-all hover:border-border-light hover:bg-bg-hover ${compact ? 'shadow-sm' : 'flex items-center gap-2 p-1.5'}`}
          >
            <div className={`relative shrink-0 overflow-hidden bg-media-canvas ${compact ? 'aspect-video w-full' : 'h-12 w-16 rounded-md'}`}>
              <MediaPreview asset={asset} />
              <span className="absolute bottom-1 left-1 max-w-[80%] truncate rounded bg-black/65 px-1 py-0.5 text-[7px] text-white/80" title={asset.workspace || asset.origin}>
                {asset.workspace === '__uploads__' ? 'Uploads' : asset.workspace || asset.origin}
              </span>
              {asset.favorite && <Heart size={9} fill="currentColor" className="absolute right-1 top-1 text-red-400 drop-shadow" />}
            </div>
            <div className={`min-w-0 ${compact ? 'p-2' : 'flex-1'}`}>
              <div className="truncate text-[9px] font-medium text-text-secondary" title={asset.name}>{asset.name}</div>
              <div className="mt-0.5 flex items-center justify-between gap-1 text-[8px] text-text-muted">
                <span className="capitalize">{asset.type}</span>
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
            className="col-span-full w-full rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-[9px] text-text-secondary hover:border-accent-blue/40 hover:text-accent-blue"
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
    </section>
  )
}
