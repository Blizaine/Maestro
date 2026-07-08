import { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { ArrowLeft, Download, Tag, Loader2, Check, ExternalLink, KeyRound, Boxes } from 'lucide-react'
import DOMPurify from 'dompurify'
import { useStore } from '../../stores/useStore'
import { fetchLoraDirectories, fetchCheckpointArchitectures, reloadModels } from '../../api/client'
import type { CheckpointArchitecture } from '../../api/client'
import type { CivitAIModel, CivitAIModelVersion, CivitAIFile } from '../../types'

function formatBytes(bytes: number): string {
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(0)} MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${bytes} B`
}

interface Props {
  model: CivitAIModel
  onBack: () => void
  // 'checkpoint' imports a full model (registers a finetune variant); 'lora'
  // (default) is the original adapter-download flow.
  kind?: 'lora' | 'checkpoint'
}

export function ModelDetail({ model, onBack, kind = 'lora' }: Props) {
  const isCheckpoint = kind === 'checkpoint'
  const startDownload = useStore(s => s.startCivitAIDownload)
  const loadModels = useStore(s => s.loadModels)
  const downloads = useStore(s => s.civitDownloads)
  // API-key gate. Downloads still attempt without a key (some public
  // LoRAs work fine), but we warn for the much larger set that won't
  // — and we point the user at the right place to fix it.
  const civitaiKeySet = !!useStore(s => s.servicesConfig?.civitai_api_key_set)
  const setLoraBrowserOpen = useStore(s => s.setLoraBrowserOpen)
  const setSettingsOpen = useStore(s => s.setSettingsOpen)
  const setSettingsTab = useStore(s => s.setSettingsTab)
  const goToCivitaiKeySettings = useCallback(() => {
    setLoraBrowserOpen(false)
    setSettingsTab('integrations')
    setSettingsOpen(true)
  }, [setLoraBrowserOpen, setSettingsTab, setSettingsOpen])

  const versions = model.modelVersions || []
  const [selectedVersionIdx, setSelectedVersionIdx] = useState(0)
  const version: CivitAIModelVersion | undefined = versions[selectedVersionIdx]
  const files = version?.files || []
  const [selectedFileIdx, setSelectedFileIdx] = useState(0)
  const file: CivitAIFile | undefined = files[selectedFileIdx]

  const images = version?.images || []
  // CivitAI descriptions are user-supplied HTML — sanitize before rendering.
  const sanitizedDescription = useMemo(
    () => (model.description ? DOMPurify.sanitize(model.description, { USE_PROFILES: { html: true } }) : ''),
    [model.description]
  )
  const trainedWords = version?.trainedWords || []
  const localArch = version?.localArch

  // Load available LoRA directories for target selection
  const browserDefaultDir = useStore(s => s.loraBrowserDefaultDir)
  const [loraDirs, setLoraDirs] = useState<string[]>([])
  const [targetDirOverride, setTargetDirOverride] = useState(browserDefaultDir || '')
  useEffect(() => {
    fetchLoraDirectories().then(r => setLoraDirs(r.directories)).catch(() => {})
  }, [])

  // Check if this file is already downloading/completed
  const activeDownload = file ? downloads.find(d => d.filename === file.name) : null

  // ── Checkpoint import: target-architecture picker ──────────────────
  // For checkpoints we don't pick a loras directory — we pick which supported
  // base architecture to register the full model under.
  const baseModel = version?.baseModel || ''
  const [architectures, setArchitectures] = useState<CheckpointArchitecture[]>([])
  const [targetArchitecture, setTargetArchitecture] = useState('')
  useEffect(() => {
    if (!isCheckpoint) return
    let cancelled = false
    fetchCheckpointArchitectures(baseModel)
      .then(r => {
        if (cancelled) return
        setArchitectures(r.architectures)
        setTargetArchitecture(prev => r.suggested_architecture || prev || r.architectures[0]?.architecture || '')
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [isCheckpoint, baseModel])

  // Group architectures by family for an <optgroup> picker.
  const groupedArchs = useMemo(() => {
    const groups: Record<string, CheckpointArchitecture[]> = {}
    for (const a of architectures) (groups[a.family || 'other'] ||= []).push(a)
    return groups
  }, [architectures])

  // Ask-per-download int8 (Phase 2): default ON for large checkpoints (a bf16
  // 22B is huge), OFF for small ones. mmgp quantizes at load, so one bf16 file
  // runs at ~int8 VRAM with no pre-quantized file.
  const fileBytes = (file?.sizeKB || 0) * 1024
  const LARGE_CKPT_BYTES = 12 * 1024 * 1024 * 1024 // 12 GB
  const [autoQuantize, setAutoQuantize] = useState(false)
  useEffect(() => {
    if (isCheckpoint) setAutoQuantize(fileBytes > LARGE_CKPT_BYTES)
  }, [isCheckpoint, fileBytes, LARGE_CKPT_BYTES])

  // After a checkpoint import completes, hot-reload the server model list +
  // refresh the UI so the new model appears in the dropdown without a restart.
  const reloadedRef = useRef(false)
  useEffect(() => {
    if (!isCheckpoint) return
    if (activeDownload?.status === 'completed' && !reloadedRef.current) {
      reloadedRef.current = true
      reloadModels().then(() => loadModels()).catch(() => {})
    } else if (!activeDownload || activeDownload.status === 'downloading') {
      reloadedRef.current = false
    }
  }, [isCheckpoint, activeDownload?.status]) // eslint-disable-line react-hooks/exhaustive-deps

  const handleDownload = () => {
    if (!file || !version) return

    // Extract example prompts from image metadata
    const examplePrompts: string[] = []
    for (const img of images) {
      const meta = (img as unknown as Record<string, unknown>).meta as Record<string, unknown> | undefined
      if (meta?.prompt && typeof meta.prompt === 'string') {
        examplePrompts.push(meta.prompt)
      }
    }

    // Strip HTML tags from descriptions
    const stripHtml = (html: string) => html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()

    const common = {
      download_url: file.downloadUrl,
      filename: file.name,
      model_id: model.id,
      version_id: version.id,
      trained_words: trainedWords,
      model_name: model.name,
      images: images.slice(0, 4).map(img => ({ url: img.url })),
      description: stripHtml(model.description || ''),
      version_description: stripHtml(version.description || ''),
      base_model: version.baseModel || '',
      example_prompts: examplePrompts.slice(0, 5),
      tags: model.tags || [],
      nsfw: model.nsfw || false,
    }
    if (isCheckpoint) {
      startDownload({ ...common, target_arch: '', kind: 'checkpoint', target_architecture: targetArchitecture, auto_quantize: autoQuantize })
    } else {
      startDownload({ ...common, target_arch: localArch || '', target_dir_name: targetDirOverride || undefined })
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border shrink-0">
        <button
          onClick={onBack}
          className="p-1.5 rounded-lg hover:bg-bg-hover text-text-secondary hover:text-text-primary transition-colors"
        >
          <ArrowLeft size={16} />
        </button>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-text-primary truncate">{model.name}</h2>
          <div className="text-[10px] text-text-muted">
            by {model.creator?.username || 'Unknown'}
            {model.type && <span className="ml-1.5 text-accent-blue">{model.type}</span>}
          </div>
        </div>
        <a
          href={`https://civitai.com/models/${model.id}`}
          target="_blank"
          rel="noopener noreferrer"
          className="p-1.5 rounded-lg hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors"
          title="Open on CivitAI"
        >
          <ExternalLink size={14} />
        </a>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Image gallery */}
        {images.length > 0 && (
          <div className="flex gap-2 overflow-x-auto p-4 pb-2">
            {images.map((img, i) => (
              <div key={i} className="shrink-0 w-40 aspect-[3/4] rounded-lg overflow-hidden border border-border bg-bg-active">
                {img.type === 'video' ? (
                  <video src={img.url} className="w-full h-full object-cover" muted loop autoPlay playsInline />
                ) : (
                  <img src={img.url} alt="" className="w-full h-full object-cover" loading="lazy" referrerPolicy="no-referrer" />
                )}
              </div>
            ))}
          </div>
        )}

        <div className="px-4 py-3 space-y-4">
          {/* Version selector */}
          {versions.length > 1 && (
            <div>
              <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1 block">Version</label>
              <select
                value={selectedVersionIdx}
                onChange={e => { setSelectedVersionIdx(Number(e.target.value)); setSelectedFileIdx(0) }}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
              >
                {versions.map((v, i) => (
                  <option key={v.id} value={i}>
                    {v.name} ({v.baseModel})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* File selector */}
          {files.length > 1 && (
            <div>
              <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1 block">File</label>
              <select
                value={selectedFileIdx}
                onChange={e => setSelectedFileIdx(Number(e.target.value))}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
              >
                {files.map((f, i) => (
                  <option key={f.id} value={i}>
                    {f.name} ({formatBytes(f.sizeKB * 1024)})
                    {f.metadata?.fp ? ` - ${f.metadata.fp}` : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Info badges */}
          <div className="flex flex-wrap gap-2">
            {version && (
              <span className="text-[10px] px-2 py-1 rounded bg-bg-active text-text-secondary">
                {version.baseModel}
              </span>
            )}
            {!isCheckpoint && localArch && (
              <span className="text-[10px] px-2 py-1 rounded bg-accent-blue/10 text-accent-blue">
                Target: {localArch}
              </span>
            )}
            {!isCheckpoint && !localArch && version && (
              <span className="text-[10px] px-2 py-1 rounded bg-amber-500/10 text-amber-400">
                Unknown architecture
              </span>
            )}
            {file && (
              <span className="text-[10px] px-2 py-1 rounded bg-bg-active text-text-muted">
                {formatBytes(file.sizeKB * 1024)}
              </span>
            )}
          </div>

          {/* Trained words */}
          {trainedWords.length > 0 && (
            <div>
              <div className="flex items-center gap-1 mb-1.5">
                <Tag size={11} className="text-text-muted" />
                <span className="text-[11px] text-text-muted uppercase tracking-wider">Trigger Words</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {trainedWords.map(word => (
                  <span
                    key={word}
                    className="text-[11px] px-2 py-0.5 rounded-full bg-bg-active text-text-secondary border border-border"
                  >
                    {word}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Description */}
          {sanitizedDescription && (
            <div>
              <div className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5">Description</div>
              <div
                className="text-xs text-text-secondary leading-relaxed max-h-[200px] overflow-y-auto bg-bg-active rounded-lg px-3 py-2 border border-border prose-sm"
                dangerouslySetInnerHTML={{ __html: sanitizedDescription }}
              />
            </div>
          )}

          {/* Target: checkpoint → base-architecture picker; lora → directory */}
          {isCheckpoint ? (
            <div>
              <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1 flex items-center gap-1.5">
                <Boxes size={12} /> Import as model
              </label>
              <select
                value={targetArchitecture}
                onChange={e => setTargetArchitecture(e.target.value)}
                className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
              >
                <option value="">Select base architecture…</option>
                {Object.entries(groupedArchs).map(([family, list]) => (
                  <optgroup key={family} label={family}>
                    {list.map(a => (
                      <option key={a.architecture} value={a.architecture}>{a.name}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
              <p className="text-[10px] text-text-muted mt-1 leading-snug">
                The base model this checkpoint was trained for{baseModel ? ` (CivitAI base: ${baseModel})` : ''}.
                It'll be registered as a new selectable model.
              </p>
            </div>
          ) : (
            loraDirs.length > 0 && (
              <div>
                <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1 block">Save to Directory</label>
                <select
                  value={targetDirOverride}
                  onChange={e => setTargetDirOverride(e.target.value)}
                  className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent-blue"
                >
                  <option value="">Auto ({localArch || 'loras'})</option>
                  {loraDirs.map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
            )
          )}

          {/* Ask-per-download int8 (checkpoint only) */}
          {isCheckpoint && (
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={autoQuantize}
                onChange={e => setAutoQuantize(e.target.checked)}
                className="w-3.5 h-3.5 rounded accent-accent-blue mt-0.5"
              />
              <span className="text-[11px] text-text-secondary leading-snug">
                Optimize VRAM (load as int8)
                <span className="block text-[10px] text-text-muted">
                  Recommended for large checkpoints — runs at roughly half the VRAM with minimal quality loss.
                  {fileBytes > 0 ? ` File: ${formatBytes(fileBytes)}.` : ''}
                </span>
              </span>
            </label>
          )}

          {/* Download button (with optional API-key advisory above) */}
          <div className="pt-2 space-y-2">
            {activeDownload ? (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-text-secondary">
                    {activeDownload.status === 'completed' ? (
                      <span className="flex items-center gap-1 text-accent-green"><Check size={12} /> {isCheckpoint ? 'Imported — added to models' : 'Downloaded'}</span>
                    ) : activeDownload.status === 'failed' ? (
                      <span className="text-red-400">Failed: {activeDownload.error}</span>
                    ) : (
                      <span className="flex items-center gap-1"><Loader2 size={12} className="animate-spin" /> Downloading...</span>
                    )}
                  </span>
                  {activeDownload.status === 'downloading' && (
                    <span className="text-text-muted">{activeDownload.progress}%</span>
                  )}
                </div>
                {activeDownload.status === 'downloading' && (
                  <div className="w-full bg-bg-active rounded-full h-1.5 overflow-hidden">
                    <div
                      className="h-full bg-accent-blue rounded-full transition-all duration-300"
                      style={{ width: `${activeDownload.progress}%` }}
                    />
                  </div>
                )}
                {/* Architecture-mismatch warnings emitted by the backend
                    after the file lands. The download itself completed
                    fine; the warning means the LoRA won't load against
                    the destination model and the user should move/replace
                    it before trying to use it. */}
                {activeDownload.status === 'completed' && activeDownload.warnings && activeDownload.warnings.length > 0 && (
                  <div className="space-y-1 mt-1">
                    {activeDownload.warnings.map((w, i) => (
                      <div key={i} className="text-[11px] text-amber-300 bg-amber-500/10 border border-amber-500/30 rounded px-2 py-1.5 leading-snug">
                        {w}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <>
                {/* Inline API-key advisory shown only when no key is set.
                    Doesn't block the download (some public LoRAs are
                    available anonymously and we want to allow optimism)
                    but makes it obvious why a download might fail and
                    where to fix it. The download endpoint surfaces a
                    crisp error if the response turns out to be bogus,
                    so the user has a complete loop. */}
                {!civitaiKeySet && (
                  <div className="flex items-start gap-2 px-3 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-[11px] text-amber-100 leading-snug">
                    <KeyRound size={12} className="text-amber-400 shrink-0 mt-0.5" />
                    <div className="flex-1 min-w-0">
                      No CivitAI API key set. Most NSFW or restricted LoRAs
                      will fail to download.{' '}
                      <a
                        href="https://civitai.com/user/account"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="underline decoration-amber-400/40 hover:decoration-amber-300 hover:text-amber-300"
                      >
                        Get a key
                      </a>
                      {' '}then{' '}
                      <button
                        onClick={goToCivitaiKeySettings}
                        className="underline decoration-amber-400/40 hover:decoration-amber-300 hover:text-amber-300"
                      >
                        paste it in Settings
                      </button>.
                    </div>
                  </div>
                )}
                <button
                  onClick={handleDownload}
                  disabled={!file || (isCheckpoint && !targetArchitecture)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-accent-blue text-white text-sm rounded-lg hover:bg-accent-blue-hover transition-colors disabled:opacity-50"
                >
                  <Download size={14} />
                  {isCheckpoint ? `Import ${file?.name || 'checkpoint'}` : `Download ${file?.name || 'LoRA'}`}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
