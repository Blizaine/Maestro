import { useEffect, useRef, useState } from 'react'
import { BookUser, ChevronDown, FileAudio, GripVertical, Image as ImageIcon, Info, Loader2, Plus, Trash2, UserPlus, Video, X } from 'lucide-react'
import * as api from '../../api/client'
import { useStore } from '../../stores/useStore'
import type { MiniMaxH3AudioIntent, MiniMaxH3Reference, MiniMaxH3ReferenceType, ModelOptions, SavedOmniCharacter } from '../../types'

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

export function OmniReferenceSection({
  scope = 'studio',
  disabled = false,
}: {
  scope?: 'studio' | 'director'
  disabled?: boolean
}) {
  const params = useStore(s => s.params)
  const studioModelOptions = useStore(s => s.modelOptions)
  const setParam = useStore(s => s.setParam)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)
  const slidingWindowSeconds = useStore(s => s.slidingWindowSeconds)
  const directorReferences = useStore(s => s.directorH3References)
  const setDirectorReferences = useStore(s => s.setDirectorH3References)
  const directorDetail = useStore(s => s.directorH3ReferenceDetail)
  const setDirectorDetail = useStore(s => s.setDirectorH3ReferenceDetail)
  const directorVideoModel = useStore(s => s.selectedModelPerMode.video || '')
  const setDirectorTargetDuration = useStore(s => s.shortFilmSetTargetDuration)
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [dragIndex, setDragIndex] = useState<number | null>(null)
  const [directorModelOptions, setDirectorModelOptions] = useState<ModelOptions | null>(null)
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [characterFormOpen, setCharacterFormOpen] = useState(false)
  const [characterName, setCharacterName] = useState('')
  const [characterVisual, setCharacterVisual] = useState<File | null>(null)
  const [characterVoice, setCharacterVoice] = useState<File | null>(null)
  const [useVideoVoice, setUseVideoVoice] = useState(false)
  const [savingCharacter, setSavingCharacter] = useState(false)

  useEffect(() => {
    if (scope !== 'director' || !directorVideoModel) return
    let cancelled = false
    setDirectorModelOptions(null)
    void api.fetchModelOptions(directorVideoModel)
      .then(options => {
        if (!cancelled) setDirectorModelOptions(options)
      })
      .catch(() => {
        if (!cancelled) setDirectorModelOptions(null)
      })
    return () => { cancelled = true }
  }, [directorVideoModel, scope])

  useEffect(() => {
    let cancelled = false
    void api.fetchCharacters()
      .then(items => { if (!cancelled) setCharacters(items) })
      .catch(() => { if (!cancelled) setCharacters([]) })
    return () => { cancelled = true }
  }, [])

  const modelOptions = scope === 'director' ? directorModelOptions : studioModelOptions
  const references = scope === 'director'
    ? directorReferences
    : (params.minimax_h3_references ?? [])
  const limits = modelOptions?.omni_reference_limits ?? {
    image: 9, video: 3, audio: 3, total: 12,
  }
  const labels = referenceLabels(references)

  const update = (next: MiniMaxH3Reference[]) => {
    if (scope === 'director') setDirectorReferences(next)
    else setParam('minimax_h3_references', next)
  }

  const currentReferences = (): MiniMaxH3Reference[] => (
    scope === 'director'
      ? useStore.getState().directorH3References
      : (useStore.getState().params.minimax_h3_references ?? [])
  )

  const addFiles = async (files: File[]) => {
    if (disabled || uploading || files.length === 0) return
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

  const addCharacter = (character: SavedOmniCharacter) => {
    const current = currentReferences()
    const currentCharacterReferences = current.filter(
      reference => reference.library_character_id === character.id,
    )
    const additions: MiniMaxH3Reference[] = []
    if (!currentCharacterReferences.some(reference => reference.type !== 'audio')) additions.push({
      id: newId(),
      type: character.visual.type,
      path: character.visual.path,
      filename: character.visual.filename,
      url: character.visual.url,
      role: character.name,
      character_name: character.name,
      library_character_id: character.id,
      image_intent: character.visual.type === 'image' ? 'identity' : undefined,
      video_intent: character.visual.type === 'video' ? 'character' : undefined,
      duration_seconds: character.visual.duration_seconds ?? null,
      has_audio: character.visual.type === 'video' ? Boolean(character.visual.has_audio) : undefined,
      include_audio: character.visual.type === 'video' ? false : undefined,
    })
    if (character.voice && !currentCharacterReferences.some(reference => reference.type === 'audio')) {
      additions.push({
        id: newId(),
        type: 'audio',
        path: character.voice.path,
        filename: character.voice.filename,
        url: character.voice.url,
        role: character.name,
        character_name: character.name,
        library_character_id: character.id,
        audio_intent: 'voice',
        duration_seconds: character.voice.duration_seconds ?? null,
        has_audio: true,
      })
    }
    if (additions.length === 0) return
    const counts = {
      image: current.filter(item => item.type === 'image').length,
      video: current.filter(item => item.type === 'video').length,
      audio: current.filter(item => item.type === 'audio').length,
    }
    for (const addition of additions) counts[addition.type] += 1
    if (
      current.length + additions.length > limits.total
      || counts.image > limits.image
      || counts.video > limits.video
      || counts.audio > limits.audio
    ) {
      setError(`Adding ${character.name} would exceed this model's Omni reference limits.`)
      return
    }
    setError('')
    update([...current, ...additions])
  }

  const saveCharacter = async () => {
    const visualType = characterVisual ? mediaType(characterVisual) : null
    if (!characterName.trim()) {
      setError('Give this character a name.')
      return
    }
    if (!characterVisual || (visualType !== 'image' && visualType !== 'video')) {
      setError('Choose one character image or video.')
      return
    }
    if (characterVoice && mediaType(characterVoice) !== 'audio' && mediaType(characterVoice) !== 'video') {
      setError('The optional voice reference must be audio, or a video containing audio.')
      return
    }
    setSavingCharacter(true)
    setError('')
    try {
      const visualUpload = await api.uploadImage(characterVisual)
      const voiceUpload = characterVoice ? await api.uploadAudio(characterVoice) : null
      const character = await api.createCharacter({
        name: characterName.trim(),
        visual_path: visualUpload.path,
        visual_type: visualType,
        voice_path: voiceUpload?.path,
        use_video_voice: visualType === 'video' && !voiceUpload && useVideoVoice,
      })
      setCharacters(current => [...current, character])
      setCharacterName('')
      setCharacterVisual(null)
      setCharacterVoice(null)
      setUseVideoVoice(false)
      setCharacterFormOpen(false)
      addCharacter(character)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Character save failed.')
    } finally {
      setSavingCharacter(false)
    }
  }

  const removeCharacter = async (character: SavedOmniCharacter) => {
    if (references.some(reference => reference.library_character_id === character.id)) {
      setError(`Remove ${character.name} from the current Omni references before deleting it.`)
      return
    }
    if (!window.confirm(`Delete saved character “${character.name}”?`)) return
    try {
      await api.deleteCharacter(character.id)
      setCharacters(current => current.filter(item => item.id !== character.id))
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : 'Character delete failed.')
    }
  }

  const setAudioIntent = (index: number, intent: MiniMaxH3AudioIntent) => {
    const reference = references[index]
    patchReference(index, { audio_intent: intent })

    // Voice and style references are reusable conditioning, not timelines.
    // An exact music/performance driver, however, defines the output length.
    if (intent !== 'drive') return
    const audioDuration = Number(reference?.duration_seconds)
    if (!Number.isFinite(audioDuration) || audioDuration <= 0) return

    if (scope === 'director') {
      // Story-driven Director projects do not have a separately analyzed
      // source track, so the exact performance reference owns their target
      // duration. Audio/music Director projects validate this duration
      // against the analyzed project timeline when submitted.
      setDirectorTargetDuration(Math.max(1, Math.round(audioDuration * 10) / 10))
      return
    }

    const fps = Math.max(1, Number(modelOptions?.fps) || 24)
    const exceedsNativeWindow = audioDuration > slidingWindowSeconds + (1 / fps)
    if (exceedsNativeWindow) {
      // Enable sequence mode before setting Duration so the store does not
      // clamp a long soundtrack back to Omni's single-pass frame lattice.
      setParam('minimax_h3_reference_sequence', true)
    }
    setDurationSeconds(audioDuration)
  }

  const attachAudio = async (referenceId: string, file: File | undefined) => {
    if (!file || disabled || uploading) return
    const type = mediaType(file)
    if (type !== 'audio' && type !== 'video') {
      setError(`${file.name} is not a supported audio file or a video with an audio track.`)
      return
    }
    setUploading(true)
    setError('')
    try {
      const uploaded = await api.uploadAudio(file)
      const current = currentReferences()
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

  const detail = scope === 'director'
    ? directorDetail
    : (params.minimax_h3_reference_detail
      ?? modelOptions?.omni_reference_detail_default
      ?? 'match')

  const setDetail = (next: 'match' | 'max') => {
    if (scope === 'director') setDirectorDetail(next)
    else setParam('minimax_h3_reference_detail', next)
  }

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">Omni References</label>
          <span
            title={scope === 'director'
              ? 'Order matters. These references are attached to every H3 Omni shot. Picture, Video, and Audio labels are assigned from top to bottom and can be named in the project description. A Music / performance timeline becomes Director’s exact audio driver.'
              : 'Order matters. Picture, Video, and Audio labels are assigned from top to bottom and can be named in your prompt. Video soundtracks stay attached to their video.'}
            className="text-text-muted cursor-help"
          >
            <Info size={12} />
          </span>
        </div>
        <span className="text-[9px] text-text-muted">{references.length}/{limits.total}</span>
      </div>

      <div
        className={`rounded-lg border border-dashed border-border px-3 py-2.5 flex items-center justify-center gap-2 transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-border-light cursor-pointer'}`}
        onClick={() => { if (!disabled) inputRef.current?.click() }}
        onDragOver={event => { if (!disabled) event.preventDefault() }}
        onDrop={event => {
          event.preventDefault()
          if (disabled) return
          void addFiles(Array.from(event.dataTransfer.files))
        }}
        aria-disabled={disabled}
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
          disabled={disabled}
          className="hidden"
          onChange={event => void addFiles(Array.from(event.target.files ?? []))}
        />
      </div>

      <div className="rounded-lg border border-border bg-bg-tertiary/50 overflow-hidden">
        <button
          type="button"
          disabled={disabled}
          aria-expanded={libraryOpen}
          onClick={() => setLibraryOpen(open => !open)}
          className="w-full flex items-center justify-between gap-2 px-2.5 py-2 text-left hover:bg-bg-tertiary disabled:opacity-50"
        >
          <span className="flex items-center gap-1.5 text-[10px] font-medium text-text-primary">
            <BookUser size={13} className="text-accent-blue" />
            Character library
            <span className="text-[9px] font-normal text-text-muted">{characters.length}</span>
          </span>
          <ChevronDown size={13} className={`text-text-muted transition-transform ${libraryOpen ? 'rotate-180' : ''}`} />
        </button>

        {libraryOpen && (
          <div className="border-t border-border p-2 space-y-2">
            <p className="text-[9px] leading-relaxed text-text-muted">
              Add a saved name to your prompt normally. Maestro binds its picture or video and voice to one H3 Subject automatically.
            </p>

            {characters.length > 0 && (
              <div className="grid grid-cols-2 gap-1.5">
                {characters.map(character => {
                  const characterReferences = references.filter(
                    reference => reference.library_character_id === character.id,
                  )
                  const added = (
                    characterReferences.some(reference => reference.type !== 'audio')
                    && (!character.voice || characterReferences.some(reference => reference.type === 'audio'))
                  )
                  return (
                    <div key={character.id} className="rounded-md border border-border bg-bg-primary p-1.5 flex items-center gap-1.5 min-w-0">
                      <div className="w-9 h-9 rounded border border-border overflow-hidden bg-bg-tertiary shrink-0 flex items-center justify-center">
                        {character.visual.type === 'image' ? (
                          <img src={character.visual.url} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <video src={character.visual.url} muted preload="metadata" className="w-full h-full object-cover" />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-[10px] text-text-primary truncate" title={character.name}>{character.name}</p>
                        <p className="text-[8px] text-text-muted truncate">
                          {character.visual.type === 'video' ? 'video' : 'image'}{character.voice ? ' + voice' : ''}
                        </p>
                        <button
                          type="button"
                          disabled={disabled || added}
                          onClick={() => addCharacter(character)}
                          className={`text-[9px] ${added ? 'text-indicator-success' : 'text-accent-blue hover:text-text-primary'}`}
                        >
                          {added ? 'Added' : 'Add to run'}
                        </button>
                      </div>
                      <button
                        type="button"
                        disabled={disabled}
                        onClick={() => void removeCharacter(character)}
                        title="Delete saved character"
                        className="self-start p-0.5 text-text-muted hover:text-indicator-error"
                      >
                        <Trash2 size={11} />
                      </button>
                    </div>
                  )
                })}
              </div>
            )}

            <button
              type="button"
              disabled={disabled}
              onClick={() => setCharacterFormOpen(open => !open)}
              className="flex items-center gap-1 text-[9px] text-accent-blue hover:text-text-primary"
            >
              <UserPlus size={11} /> {characterFormOpen ? 'Close new character' : 'Save a new character'}
            </button>

            {characterFormOpen && (
              <div className="rounded-md border border-border bg-bg-primary p-2 space-y-1.5">
                <input
                  value={characterName}
                  disabled={disabled || savingCharacter}
                  onChange={event => setCharacterName(event.target.value)}
                  placeholder="Character name"
                  className="w-full bg-bg-tertiary border border-border rounded px-2 py-1 text-[10px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                />
                <label className="block rounded border border-dashed border-border px-2 py-1.5 text-[9px] text-text-secondary cursor-pointer hover:border-border-light">
                  <span className="font-medium">Image or video:</span> {characterVisual?.name || 'Choose visual reference'}
                  <input
                    type="file"
                    disabled={disabled || savingCharacter}
                    className="hidden"
                    onChange={event => {
                      const file = event.target.files?.[0] ?? null
                      setCharacterVisual(file)
                      if (file && mediaType(file) !== 'video') setUseVideoVoice(false)
                      event.currentTarget.value = ''
                    }}
                  />
                </label>
                <label className="block rounded border border-dashed border-border px-2 py-1.5 text-[9px] text-text-secondary cursor-pointer hover:border-border-light">
                  <span className="font-medium">Voice (optional):</span> {characterVoice?.name || 'Choose audio or video'}
                  <input
                    type="file"
                    disabled={disabled || savingCharacter}
                    className="hidden"
                    onChange={event => {
                      setCharacterVoice(event.target.files?.[0] ?? null)
                      event.currentTarget.value = ''
                    }}
                  />
                </label>
                {characterVisual && mediaType(characterVisual) === 'video' && !characterVoice && (
                  <label className="flex items-center gap-1.5 text-[9px] text-text-secondary cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useVideoVoice}
                      disabled={disabled || savingCharacter}
                      onChange={event => setUseVideoVoice(event.target.checked)}
                      className="w-3 h-3 accent-accent-blue"
                    />
                    Use this video's audio as the voice reference
                  </label>
                )}
                <p className="text-[8px] leading-relaxed text-text-muted">
                  Videos remain saved at full length. For each run Maestro makes H3-ready cached copies: 2–15 seconds each and 15 seconds total (three 10s clips become 5s each).
                </p>
                <button
                  type="button"
                  disabled={disabled || savingCharacter}
                  onClick={() => void saveCharacter()}
                  className="w-full rounded bg-accent-blue px-2 py-1.5 text-[9px] font-medium text-white disabled:opacity-50 flex items-center justify-center gap-1"
                >
                  {savingCharacter ? <Loader2 size={11} className="animate-spin" /> : <UserPlus size={11} />}
                  {savingCharacter ? 'Saving…' : 'Save and add'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {references.length > 0 && (
        <div className="space-y-1.5">
          {references.map((reference, index) => (
            <div
              key={reference.id || `${reference.path}-${index}`}
              draggable={!disabled}
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
                  disabled={disabled}
                  onChange={event => patchReference(index, { role: event.target.value })}
                  placeholder="Who or what is this? (helps Enhance)"
                  className="w-full bg-bg-primary border border-border rounded px-2 py-1 text-[10px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                />
                {reference.type === 'audio' && (
                  <select
                    value={reference.audio_intent ?? 'voice'}
                    disabled={disabled}
                    onChange={event => setAudioIntent(
                      index,
                      event.target.value as MiniMaxH3AudioIntent,
                    )}
                    title="Voice reference is reused for identity in every clip. Music / performance timeline adopts the track duration, preserves the exact soundtrack and advances through it across sequence clips. It automatically enables a multi-window sequence when needed. Style-only borrows musical character rather than exact audio or timing."
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
                        disabled={disabled}
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
                        disabled={disabled}
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
                      disabled={disabled}
                      checked={reference.include_audio !== false}
                      onChange={event => patchReference(index, { include_audio: event.target.checked })}
                      className="w-3 h-3 accent-accent-blue"
                    />
                    Include soundtrack
                  </label>
                )}
              </div>
              <button
                disabled={disabled}
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
            disabled={disabled}
            onChange={event => setDetail(event.target.value as 'match' | 'max')}
            title="Match output preserves the selected output-sized preparation and avoids reference upscaling. High detail follows the official Ref2VA PDD 2048px-short-edge recipe, but can use substantially more memory and time."
            className="bg-bg-tertiary border border-border rounded px-2 py-1 text-[9px] text-text-secondary focus:outline-none focus:border-accent-blue"
          >
            {(modelOptions?.omni_reference_detail_choices ?? [
              ['Match output (faster)', 'match'],
              ['High detail (official PDD recipe)', 'max'],
            ]).map(([label, value]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
      )}

      {references.filter(reference => reference.type === 'video').reduce((sum, reference) => sum + (Number(reference.duration_seconds) || 0), 0) > 15 && (
        <p className="text-[8px] leading-relaxed text-text-muted">
          These video references exceed H3's 15-second combined limit. Maestro will balance cached trimmed copies across them; your originals and saved characters remain unchanged.
        </p>
      )}

      {error && <p className="text-[9px] text-indicator-error">{error}</p>}
    </section>
  )
}
