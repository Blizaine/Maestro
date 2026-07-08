import { useState, useRef } from 'react'
import { Pencil, RefreshCw, Copy, Trash2, Check, Combine, Loader2, Sparkles, Mic } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { getUploadUrl } from '../../api/client'
import { modelDisplayName } from '../../lib/modelDisplay'

export function VideoInfoBar() {
  const outputs = useStore(s => s.filteredOutputs())
  const selectedOutput = useStore(s => s.selectedOutput)
  const meta = useStore(s => s.selectedOutputMeta)
  const metadataLoading = useStore(s => s.metadataLoading)
  const loadSettingsFromOutput = useStore(s => s.loadSettingsFromOutput)
  const rerollGeneration = useStore(s => s.rerollGeneration)
  const deleteSelectedOutput = useStore(s => s.deleteSelectedOutput)
  const rejoinClipGroup = useStore(s => s.rejoinClipGroup)
  const quickUpscaleClip = useStore(s => s.quickUpscaleClip)
  const sendClipToTools = useStore(s => s.sendClipToTools)
  // For translating the raw model_type slug into the display name
  // (see modelDisplayName helper).
  const models = useStore(s => s.models)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const confirmRef = useRef(false)
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const [copied, setCopied] = useState(false)
  const [rejoining, setRejoining] = useState(false)
  const [upscaling, setUpscaling] = useState(false)

  const selected = outputs[selectedOutput]
  if (!selected) return null

  // While loading, show a subtle bar
  if (metadataLoading) {
    return (
      <div className="shrink-0 px-4 py-2 border-b border-border min-h-[44px] flex items-center">
        <div className="text-[11px] text-text-muted animate-pulse">Loading info...</div>
      </div>
    )
  }

  const params = meta?.params as Record<string, unknown> | null
  const uploadFilenames = meta?.upload_filenames as Record<string, string> | undefined

  // Extract display values
  const prompt = (params?.prompt as string) || ''
  const modelType = (params?.model_type as string) || ''
  const modelLabel = modelDisplayName(modelType, models)
  const resolution = (params?.resolution as string) || ''
  const seed = params?.seed as number | undefined
  const generationTime = meta?.generation_time

  // Multi-clip group info
  const multiClipInfo = params?.multi_clip_info as { group_id: string; index: number; total: number } | undefined
  const groupId = multiClipInfo?.group_id
  const clipIndex = multiClipInfo?.index
  const clipTotal = multiClipInfo?.total

  // Image thumbnails from upload filenames (may be string or array for multi-clip)
  const rawStart = uploadFilenames?.image_start
  const rawEnd = uploadFilenames?.image_end
  const imageStartFile = Array.isArray(rawStart) ? (rawStart.find(f => f) || null) : rawStart
  const imageEndFile = Array.isArray(rawEnd) ? (rawEnd.find(f => f) || null) : rawEnd

  const handleCopyPrompt = () => {
    if (!prompt) return
    navigator.clipboard.writeText(prompt)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  const handleDelete = async () => {
    if (!confirmRef.current) {
      confirmRef.current = true
      setConfirmDelete(true)
      clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(() => {
        confirmRef.current = false
        setConfirmDelete(false)
      }, 3000)
      return
    }
    clearTimeout(timeoutRef.current)
    confirmRef.current = false
    setConfirmDelete(false)
    await deleteSelectedOutput()
  }

  const handleRejoin = async () => {
    if (!groupId) return
    setRejoining(true)
    try {
      await rejoinClipGroup(groupId)
    } finally {
      setRejoining(false)
    }
  }

  const handleUpscale = async () => {
    if (upscaling) return
    setUpscaling(true)
    try {
      await quickUpscaleClip(selected.name, selected.url)
    } catch {
      /* the job tile surfaces any error */
    } finally {
      // Brief submit cooldown; the job tile tracks real progress from here.
      setTimeout(() => setUpscaling(false), 1500)
    }
  }

  return (
    <div className="shrink-0 px-4 py-2 border-b border-border flex items-center gap-3 min-h-[44px]">
      {/* Start/End image thumbnails */}
      {imageStartFile && (
        <img
          src={getUploadUrl(imageStartFile)}
          alt="Start"
          className="w-8 h-8 rounded border border-border object-cover shrink-0"
          title="Start image"
        />
      )}
      {imageEndFile && (
        <img
          src={getUploadUrl(imageEndFile)}
          alt="End"
          className="w-8 h-8 rounded border border-border object-cover shrink-0"
          title="End image"
        />
      )}

      {/* Info text */}
      <div className="flex-1 min-w-0">
        {params ? (
          <>
            <div className="text-xs text-text-secondary truncate">
              {modelLabel && <span className="font-medium" title={modelType}>{modelLabel}</span>}
              {resolution && <span className="text-text-muted"> &middot; {resolution}</span>}
              {seed != null && seed >= 0 && <span className="text-text-muted"> &middot; seed {seed}</span>}
              {generationTime != null && <span className="text-text-muted"> &middot; {generationTime}s</span>}
              {clipIndex != null && clipTotal != null && (
                <span className="text-accent-blue"> &middot; clip {clipIndex + 1}/{clipTotal}</span>
              )}
            </div>
            {prompt && (
              <div className="text-[11px] text-text-muted truncate mt-0.5" title={prompt}>
                {prompt}
              </div>
            )}
          </>
        ) : (
          <div className="text-[11px] text-text-muted">{selected.name}</div>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-0.5 shrink-0">
        {params && (
          <>
            <button
              onClick={loadSettingsFromOutput}
              className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Load settings"
            >
              <Pencil size={14} />
            </button>
            <button
              onClick={rerollGeneration}
              className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Re-generate with same settings"
            >
              <RefreshCw size={14} />
            </button>
            {groupId && (
              <button
                onClick={handleRejoin}
                disabled={rejoining}
                className="p-1.5 rounded-lg hover:bg-bg-hover text-accent-blue hover:text-accent-blue-hover transition-colors disabled:opacity-50"
                title={`Rejoin all ${clipTotal} clips in this group`}
              >
                {rejoining ? <Loader2 size={14} className="animate-spin" /> : <Combine size={14} />}
              </button>
            )}
            <button
              onClick={handleCopyPrompt}
              className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
              title="Copy prompt"
            >
              {copied ? <Check size={14} className="text-accent-green" /> : <Copy size={14} />}
            </button>
          </>
        )}
        {/* Tools: one-click post-processing on this clip. Upscale fires now;
            Replace voice opens the Tools panel (it needs voice references). */}
        {selected.type === 'video' && (
          <>
            <button
              onClick={handleUpscale}
              disabled={upscaling}
              className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-accent-blue transition-colors disabled:opacity-50"
              title="Upscale this clip (FlashVSR / configured method)"
            >
              {upscaling ? <Loader2 size={14} className="animate-spin" /> : <Sparkles size={14} />}
            </button>
            <button
              onClick={() => sendClipToTools(selected.name, selected.url, 'revoice')}
              className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-accent-blue transition-colors"
              title="Replace the voice in this clip (SeedVC)"
            >
              <Mic size={14} />
            </button>
          </>
        )}
        <button
          onClick={handleDelete}
          className={`p-1.5 rounded-lg transition-colors flex items-center gap-1 ${
            confirmDelete
              ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
              : 'hover:bg-bg-hover text-text-secondary hover:text-red-400'
          }`}
          title={confirmDelete ? 'Click again to confirm delete' : 'Delete output'}
        >
          <Trash2 size={14} />
          {confirmDelete && <span className="text-[11px] font-medium">Delete?</span>}
        </button>
      </div>
    </div>
  )
}
