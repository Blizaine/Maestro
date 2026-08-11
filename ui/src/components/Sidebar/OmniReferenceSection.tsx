import { useMemo, useRef, useState } from 'react'
import { FileAudio, GripVertical, Image as ImageIcon, Info, Loader2, Plus, Video, X } from 'lucide-react'
import * as api from '../../api/client'
import { useStore } from '../../stores/useStore'
import type { MiniMaxH3AudioIntent, MiniMaxH3Reference, MiniMaxH3ReferenceType } from '../../types'

const IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i
const VIDEO_RE = /\.(mp4|mov|mkv|webm|avi|m4v)$/i
const AUDIO_RE = /\.(wav|mp3|flac|ogg|m4a|aac)$/i

function mediaType(file: File): MiniMaxH3ReferenceType | null {
  // Prefer a recognized extension. Some iOS document providers expose M4A
  // files with a generic or video/mp4 MIME type even though they are audio.
  if (IMAGE_RE.test(file.name)) return 'image'
  if (VIDEO_RE.test(file.name)) return 'video'
  if (AUDIO_RE.test(file.name)) return 'audio'
  if (file.type.startsWith('image/')) return 'image'
  if (file.type.startsWith('video/')) return 'video'
  if (file.type.startsWith('audio/')) return 'audio'
  return null
}

function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function referenceLabels(references: MiniMaxH3Reference[]): string[] {
  let pictures = 0
  let videos = 0
  let audios = 0
  return references.map(reference => {
    const labels: string[] = []
    if (reference.type === 'audio' || (reference.type === 'video' && (reference.has_audio || reference.audio_path) && reference.include_audio !== false)) {
      labels.push(`Audio ${++audios}`)
    }
    if (reference.type === 'image') labels.push(`Picture ${++pictures}`)
    if (reference.type === 'video') labels.push(`Video ${++videos}`)
    return labels.join(' + ')
  })
}

export function OmniReferenceSection() {
  const params = useStore(s => s.params)
  const modelOptions = useStore(s => s.modelOptions)
  const setParam = useStore(s => s.setParam)
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [dragIndex, setDragIndex] = useState<number | null>(null)

  const references = params.minimax_h3_references ?? []
  const limits = modelOptions?.omni_reference_limits ?? {
    image: 9, video: 3, audio: 3, total: 12,
  }
  const labels = useMemo(() => referenceLabels(references), [references])

  const update = (next: MiniMaxH3Reference[]) => setParam('minimax_h3_references', next)

  const addFiles = async (files: File[]) => {
    if (uploading || files.length === 0) return
    setUploading(true)
    setError('')
    try {
      const next = [...references]
      const counts = {
        image: next.filter(item => item.type === 'image').length,
        video: next.filter(item => item.type === 'video').length,
        audio: next.filter(item => item.type === 'audio').length,
      }
      for (const file of files) {
        const type = mediaType(file)
        if (!type) {
          setError(`${file.name} is not a supported image, video, or audio file.`)
          continue
        }
        if (next.length >= limits.total || counts[type] >= limits[type]) {
          setError(`Reference limit reached (${limits.image} images, ${limits.video} videos, ${limits.audio} audio; ${limits.total} total).`)
          break
        }
        const uploaded = type === 'audio'
          ? await api.uploadAudio(file)
          : await api.uploadImage(file)
        next.push({
          id: newId(),
          type,
          path: uploaded.path,
          filename: file.name,
          url: uploaded.url,
          duration_seconds: uploaded.duration_seconds ?? null,
          has_audio: type === 'video' ? Boolean('has_audio' in uploaded && uploaded.has_audio) : type === 'audio',
          include_audio: type === 'video' ? Boolean('has_audio' in uploaded && uploaded.has_audio) : undefined,
          audio_intent: type === 'audio' ? 'voice' : undefined,
          role: '',
        })
        counts[type] += 1
      }
      update(next)
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Reference upload failed.')
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  const patchReference = (index: number, patch: Partial<MiniMaxH3Reference>) => {
    update(references.map((reference, itemIndex) => itemIndex === index ? { ...reference, ...patch } : reference))
  }

  const attachAudio = async (referenceId: string, file: File | undefined) => {
    if (!file || uploading) return
    const type = mediaType(file)
    if (type !== 'audio' && type !== 'video') {
      setError(`${file.name} is not a supported audio file or a video with an audio track.`)
      return
    }
    setUploading(true)
    setError('')
    try {
      const uploaded = await api.uploadAudio(file)
      const current = useStore.getState().params.minimax_h3_references ?? []
      update(current.map(reference => reference.id === referenceId ? {
        ...reference,
        audio_path: uploaded.path,
        audio_filename: file.name,
        audio_duration_seconds: uploaded.duration_seconds ?? null,
        include_audio: true,
      } : reference))
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Soundtrack upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const reorder = (from: number, to: number) => {
    if (from === to) return
    const next = [...references]
    const [item] = next.splice(from, 1)
    next.splice(to, 0, item)
    update(next)
  }

  const detail = params.minimax_h3_reference_detail
    ?? modelOptions?.omni_reference_detail_default
    ?? 'match'

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">Omni References</label>
          <span
            title="Order matters. Picture, Video, and Audio labels are assigned from top to bottom and can be named in your prompt. Video soundtracks stay attached to their video."
            className="text-text-muted cursor-help"
          >
            <Info size={12} />
          </span>
        </div>
        <span className="text-[9px] text-text-muted">{references.length}/{limits.total}</span>
      </div>

      <div
        className="rounded-lg border border-dashed border-border hover:border-border-light px-3 py-2.5 flex items-center justify-center gap-2 cursor-pointer transition-colors"
        onClick={() => inputRef.current?.click()}
        onDragOver={event => event.preventDefault()}
        onDrop={event => {
          event.preventDefault()
          void addFiles(Array.from(event.dataTransfer.files))
        }}
      >
        {uploading ? <Loader2 size={14} className="animate-spin text-accent-blue" /> : <Plus size={14} className="text-text-muted" />}
        <span className="text-[10px] text-text-secondary">{uploading ? 'Uploading references…' : 'Add images, videos, or audio'}</span>
        {/* Do not add a mixed-media `accept` filter here. iOS/WebKit can
            grey out valid audio files when audio, image, and video types are
            combined. Maestro validates the selected files in addFiles(). */}
        <input
          ref={inputRef}
          type="file"
          multiple
          className="hidden"
          onChange={event => void addFiles(Array.from(event.target.files ?? []))}
        />
      </div>

      {references.length > 0 && (
        <div className="space-y-1.5">
          {references.map((reference, index) => (
            <div
              key={reference.id || `${reference.path}-${index}`}
              draggable
              onDragStart={() => setDragIndex(index)}
              onDragOver={event => event.preventDefault()}
              onDrop={event => {
                event.preventDefault()
                if (dragIndex !== null) reorder(dragIndex, index)
                setDragIndex(null)
              }}
              onDragEnd={() => setDragIndex(null)}
              className={`rounded-lg border bg-bg-tertiary p-2 flex gap-2 transition-colors ${dragIndex === index ? 'border-accent-blue' : 'border-border'}`}
            >
              <GripVertical size={14} className="mt-2 text-text-muted cursor-grab shrink-0" />
              <div className="w-12 h-12 rounded-md border border-border overflow-hidden bg-bg-primary flex items-center justify-center shrink-0">
                {reference.type === 'image' && reference.url ? (
                  <img src={reference.url} alt="" className="w-full h-full object-cover" />
                ) : reference.type === 'video' && reference.url ? (
                  <video src={reference.url} muted preload="metadata" className="w-full h-full object-cover" />
                ) : reference.type === 'audio' ? (
                  <FileAudio size={18} className="text-accent-blue" />
                ) : reference.type === 'video' ? (
                  <Video size={18} className="text-accent-blue" />
                ) : (
                  <ImageIcon size={18} className="text-accent-blue" />
                )}
              </div>
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] font-medium text-text-primary">{labels[index]}</span>
                  <span className="text-[9px] text-text-muted truncate">{reference.filename}</span>
                </div>
                <input
                  value={reference.role ?? ''}
                  onChange={event => patchReference(index, { role: event.target.value })}
                  placeholder="Who or what is this? (helps Enhance)"
                  className="w-full bg-bg-primary border border-border rounded px-2 py-1 text-[10px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                />
                {reference.type === 'audio' && (
                  <select
                    value={reference.audio_intent ?? 'voice'}
                    onChange={event => patchReference(index, {
                      audio_intent: event.target.value as MiniMaxH3AudioIntent,
                    })}
                    title="Voice reference is reused for identity in every clip. Music/performance timeline advances through the recording across sequence clips. Style-only advances through long references but borrows character rather than exact audio or timing."
                    className="w-full bg-bg-primary border border-border rounded px-2 py-1 text-[10px] text-text-secondary focus:outline-none focus:border-accent-blue"
                  >
                    <option value="voice">Voice reference</option>
                    <option value="drive">Music / performance timeline</option>
                    <option value="style">Music / sound style only</option>
                  </select>
                )}
                {reference.type === 'video' && (
                  <div className="flex items-center gap-1.5 text-[9px] text-text-secondary">
                    <label className="cursor-pointer hover:text-text-primary">
                      {reference.audio_path ? 'Replace audio' : 'Attach audio'}
                      {/* iOS has also shipped audio-only picker regressions.
                          Browse freely, then validate in attachAudio(). */}
                      <input
                        type="file"
                        className="hidden"
                        onChange={event => {
                          void attachAudio(reference.id, event.target.files?.[0])
                          event.currentTarget.value = ''
                        }}
                      />
                    </label>
                    {reference.audio_path && (
                      <button
                        type="button"
                        title="Remove attached soundtrack"
                        onClick={() => patchReference(index, {
                          audio_path: undefined,
                          audio_filename: undefined,
                          audio_duration_seconds: undefined,
                          include_audio: reference.has_audio === true,
                        })}
                        className="truncate text-text-muted hover:text-indicator-error"
                      >
                        × {reference.audio_filename || 'attached audio'}
                      </button>
                    )}
                  </div>
                )}
                {reference.type === 'video' && (reference.has_audio || reference.audio_path) && (
                  <label className="flex items-center gap-1.5 text-[9px] text-text-secondary cursor-pointer">
                    <input
                      type="checkbox"
                      checked={reference.include_audio !== false}
                      onChange={event => patchReference(index, { include_audio: event.target.checked })}
                      className="w-3 h-3 accent-accent-blue"
                    />
                    Include soundtrack
                  </label>
                )}
              </div>
              <button
                onClick={() => update(references.filter((_, itemIndex) => itemIndex !== index))}
                title="Remove reference"
                className="p-1 self-start text-text-muted hover:text-indicator-error"
              >
                <X size={13} />
              </button>
            </div>
          ))}
        </div>
      )}

      {references.length > 0 && (
        <div className="flex items-center justify-end gap-2">
          <select
            value={detail}
            onChange={event => setParam('minimax_h3_reference_detail', event.target.value as 'match' | 'max')}
            title="Match output uses less memory and is recommended. Maximum detail follows the official 2048px-short-edge reference preparation and can require substantially more memory."
            className="bg-bg-tertiary border border-border rounded px-2 py-1 text-[9px] text-text-secondary focus:outline-none focus:border-accent-blue"
          >
            {(modelOptions?.omni_reference_detail_choices ?? [
              ['Match output', 'match'],
              ['Maximum detail', 'max'],
            ]).map(([label, value]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
      )}

      {error && <p className="text-[9px] text-indicator-error">{error}</p>}
    </section>
  )
}
