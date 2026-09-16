import { useEffect, useId, useRef, useState } from 'react'
import { ChevronDown, FileAudio, Info, Loader2, UserPlus, UserRound, X } from 'lucide-react'
import * as api from '../../api/client'
import { useStore } from '../../stores/useStore'
import { readPersistentDisclosure, writePersistentDisclosure } from '../../lib/persistentDisclosure'
import { ImportCharacterButton } from '../Characters/CharacterFileActions'
import { ReferenceCharacterPicker } from '../Characters/ReferenceCharacterPicker'
import { characterDisplayName } from '../../lib/characters'
import { CharacterToolbarItem, SidebarDialog } from './SidebarPanels'
import { MediaAddTile, MediaInputCard } from './MediaInputCard'
import type { MiniMaxH3AudioIntent, MiniMaxH3Reference, MiniMaxH3ReferenceType, ModelOptions, SavedOmniCharacter } from '../../types'

const IMAGE_RE = /\.(png|jpe?g|webp|bmp|tiff?)$/i
const VIDEO_RE = /\.(mp4|mov|mkv|webm|avi|m4v)$/i
const AUDIO_RE = /\.(wav|mp3|flac|ogg|m4a|aac)$/i
const EMPTY_REFERENCES: MiniMaxH3Reference[] = []
const CHARACTER_LIBRARY_EXPANDED_KEY = 'maestro-omni-characters-expanded'

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

type ActiveReferenceItem =
  | {
      kind: 'character'
      characterId: string
      entries: Array<{ reference: MiniMaxH3Reference; index: number }>
    }
  | {
      kind: 'reference'
      reference: MiniMaxH3Reference
      index: number
    }

function referenceItemKey(item: ActiveReferenceItem): string {
  return item.kind === 'character' ? `character-${item.characterId}` : item.reference.id || item.reference.path
}

function groupActiveReferences(references: MiniMaxH3Reference[]): ActiveReferenceItem[] {
  const items: ActiveReferenceItem[] = []
  const characterItems = new Map<string, Extract<ActiveReferenceItem, { kind: 'character' }>>()
  references.forEach((reference, index) => {
    const characterId = reference.library_character_id?.trim()
    if (!characterId) {
      items.push({ kind: 'reference', reference, index })
      return
    }
    let item = characterItems.get(characterId)
    if (!item) {
      item = { kind: 'character', characterId, entries: [] }
      characterItems.set(characterId, item)
      items.push(item)
    }
    item.entries.push({ reference, index })
  })
  return items
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
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [dragIndices, setDragIndices] = useState<number[] | null>(null)
  const [editingKey, setEditingKey] = useState<string | null>(null)
  const editorId = useId()
  const referenceGrid = useRef<HTMLDivElement>(null)
  const [directorModelOptions, setDirectorModelOptions] = useState<ModelOptions | null>(null)
  const [characters, setCharacters] = useState<SavedOmniCharacter[]>([])
  const [libraryOpen, setLibraryOpen] = useState(() => (
    readPersistentDisclosure(CHARACTER_LIBRARY_EXPANDED_KEY, false)
  ))
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
    const refresh = () => { void api.fetchCharacters()
      .then(items => { if (!cancelled) setCharacters(items) })
      .catch(() => { if (!cancelled) setCharacters([]) }) }
    refresh()
    window.addEventListener('maestro-characters-changed', refresh)
    window.addEventListener('focus', refresh)
    return () => {
      cancelled = true
      window.removeEventListener('maestro-characters-changed', refresh)
      window.removeEventListener('focus', refresh)
    }
  }, [])

  useEffect(() => {
    writePersistentDisclosure(CHARACTER_LIBRARY_EXPANDED_KEY, libraryOpen)
  }, [libraryOpen])

  const modelOptions = scope === 'director' ? directorModelOptions : studioModelOptions
  const references = scope === 'director'
    ? directorReferences
    : (params.minimax_h3_references ?? EMPTY_REFERENCES)
  const limits = modelOptions?.omni_reference_limits ?? {
    image: 9, video: 3, audio: 3, total: 12,
  }
  const labels = referenceLabels(references)

  useEffect(() => {
    const needsVoiceLock = references.some(reference => (
      Boolean(reference.library_character_id)
      && reference.type === 'audio'
      && reference.audio_intent !== 'voice'
    ))
    if (!needsVoiceLock) return
    const normalized = references.map(reference => (
      reference.library_character_id && reference.type === 'audio'
        ? { ...reference, audio_intent: 'voice' as const }
        : reference
    ))
    if (scope === 'director') setDirectorReferences(normalized)
    else setParam('minimax_h3_references', normalized)
  }, [references, scope, setDirectorReferences, setParam])

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
    }
  }

  const patchReference = (index: number, patch: Partial<MiniMaxH3Reference>) => {
    update(references.map((reference, itemIndex) => itemIndex === index ? { ...reference, ...patch } : reference))
  }

  const replaceReference = async (reference: MiniMaxH3Reference, file?: File) => {
    if (!file || disabled || uploading) return
    if (mediaType(file) !== reference.type) { setError(`Choose a ${reference.type} to replace this reference.`); return }
    setUploading(true)
    setError('')
    try {
      const uploaded = reference.type === 'audio' ? await api.uploadAudio(file) : await api.uploadImage(file)
      update(currentReferences().map(item => item.id === reference.id ? {
        ...item, path: uploaded.path, filename: file.name, url: uploaded.url,
        duration_seconds: uploaded.duration_seconds ?? null,
        has_audio: reference.type === 'video' ? Boolean('has_audio' in uploaded && uploaded.has_audio) : reference.type === 'audio',
      } : item))
    } catch (error) { setError(error instanceof Error ? error.message : 'Could not replace reference.') }
    finally { setUploading(false) }
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
      refmod_path: character.refmod?.path,
      image_intent: character.visual.type === 'image' ? 'identity' : undefined,
      remove_background: character.visual.type === 'image' ? false : undefined,
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
      setError(`Adding ${characterDisplayName(character.name)} would exceed this model's Omni reference limits.`)
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
      setError(`Remove ${characterDisplayName(character.name)} from the current Omni references before deleting it.`)
      return
    }
    if (!window.confirm(`Delete saved character “${characterDisplayName(character.name)}”?`)) return
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

  const reorder = (fromIndices: number[], to: number) => {
    const ordered = groupActiveReferences(references)
    const indices = (item: ActiveReferenceItem) => item.kind === 'character' ? item.entries.map(entry => entry.index) : [item.index]
    const from = ordered.findIndex(item => indices(item).some(index => fromIndices.includes(index)))
    const target = ordered.findIndex(item => indices(item).includes(to))
    if (from < 0 || target < 0 || from === target) return
    // Move to the target tile's position in either direction, keeping a saved
    // character's visual and voice together. Removing the source first used to
    // turn a drop on the very next tile into a no-op.
    const [moving] = ordered.splice(from, 1)
    ordered.splice(target, 0, moving)
    update(ordered.flatMap(item => item.kind === 'character' ? item.entries.map(entry => entry.reference) : [item.reference]))
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

  const activeItems = groupActiveReferences(references)
  const canAddMore = references.length < limits.total && (['image', 'video', 'audio'] as const)
    .some(type => references.filter(reference => reference.type === type).length < limits[type])
  // Keep all three thumbnails in their row. The selected editor participates
  // in normal document flow, so it never overlays the prompt or other inputs.
  const referenceRows = Array.from({ length: Math.ceil((activeItems.length + Number(canAddMore)) / 3) }, (_, row) => activeItems.slice(row * 3, row * 3 + 3))
  const closeEditor = () => {
    referenceGrid.current?.querySelector<HTMLButtonElement>('button[aria-expanded="true"]')?.focus({ preventScroll: true })
    setEditingKey(null)
  }
  const moveItem = (itemIndex: number, direction: -1 | 1) => {
    const ordered = [...activeItems]
    const target = itemIndex + direction
    if (target < 0 || target >= ordered.length) return
    ;[ordered[itemIndex], ordered[target]] = [ordered[target], ordered[itemIndex]]
    update(ordered.flatMap(item => item.kind === 'character' ? item.entries.map(entry => entry.reference) : [item.reference]))
  }
  const addedCharacterIds = characters.filter(character => {
    const bound = references.filter(reference => reference.library_character_id === character.id)
    return bound.some(reference => reference.type !== 'audio')
      && (!character.voice || bound.some(reference => reference.type === 'audio'))
  }).map(character => character.id)

  const referenceFields = (item: ActiveReferenceItem) => {
    const fieldClass = 'min-h-9 w-full min-w-0 rounded-lg border border-border bg-bg-primary px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue disabled:opacity-50'
    if (item.kind === 'character') {
      const visualEntry = item.entries.find(entry => entry.reference.type !== 'audio')
      const voiceEntry = item.entries.find(entry => entry.reference.type === 'audio')
      return <div className="space-y-2 text-xs text-text-secondary">
        <div className="flex flex-wrap items-center gap-2">
          <span>{visualEntry?.reference.type === 'video' ? 'Video identity' : 'Image identity'}</span>
          {voiceEntry && <span className="flex items-center gap-1"><FileAudio size={12} /> Voice reference</span>}
        </div>
        <p className="text-[11px] text-text-muted">{item.entries.map(entry => labels[entry.index]).join(' + ')} · Linked to this saved character</p>
        {visualEntry?.reference.refmod_path && <p className="text-[11px] text-text-muted">Uses the saved H3 RefMod appearance.</p>}
        {visualEntry?.reference.type === 'image' && !visualEntry.reference.refmod_path && <label
          className="flex min-h-8 cursor-pointer items-center gap-2 text-[11px]"
          title="Place the character on neutral white before generation when the portrait background leaks into the scene. Leave off to preserve the original lighting context.">
          <input type="checkbox" disabled={disabled} checked={visualEntry.reference.remove_background === true}
            onChange={event => patchReference(visualEntry.index, { remove_background: event.target.checked })} className="h-3.5 w-3.5 accent-accent-blue" />
          Isolate portrait background
        </label>}
      </div>
    }

    const { reference, index } = item
    return <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        <label className="min-w-[140px] flex-1 space-y-1 text-[11px] text-text-muted">
          <span>Name</span>
          <input type="text" aria-label={`${labels[index]} name`} value={reference.role ?? ''} disabled={disabled}
            onChange={event => patchReference(index, { role: event.target.value })}
            placeholder="Who or what is this?" className={fieldClass} />
        </label>
        {reference.type === 'image' && <label className="min-w-[165px] flex-1 space-y-1 text-[11px] text-text-muted">
          <span>Type</span>
          <select aria-label={`${labels[index]} type`} value={reference.image_intent ?? 'identity'} disabled={disabled}
            onChange={event => patchReference(index, { image_intent: event.target.value as MiniMaxH3Reference['image_intent'] })} className={fieldClass}>
            <option value="identity">Character / identity</option><option value="scene">Scene</option><option value="composition">Composition</option><option value="style">Style reference</option>
          </select>
        </label>}
        {reference.type === 'audio' && <label className="min-w-[165px] flex-1 space-y-1 text-[11px] text-text-muted">
          <span>Type</span>
          <select aria-label={`${labels[index]} type`} value={reference.audio_intent ?? 'voice'} disabled={disabled}
            onChange={event => setAudioIntent(index, event.target.value as MiniMaxH3AudioIntent)}
            title="Voice references preserve identity. Music / performance timeline preserves the exact soundtrack and sets the duration. Style-only borrows musical character without exact audio or timing."
            className={fieldClass}>
            <option value="voice">Voice reference</option><option value="drive">Music / performance timeline</option><option value="style">Music / sound style only</option>
          </select>
        </label>}
      </div>
      {reference.type === 'image' && !reference.refmod_path && (reference.image_intent ?? 'identity') === 'identity' && <label
        className="flex min-h-8 cursor-pointer items-center gap-2 text-[11px] text-text-secondary"
        title="Place the subject on neutral white before generation when the portrait background leaks into the scene. Scene, style and composition references are never altered.">
        <input type="checkbox" disabled={disabled} checked={reference.remove_background === true}
          onChange={event => patchReference(index, { remove_background: event.target.checked })} className="h-3.5 w-3.5 accent-accent-blue" />
        Isolate subject background
      </label>}
      {reference.type === 'video' && (reference.has_audio || reference.audio_path) && <label className="flex min-h-8 cursor-pointer items-center gap-2 text-[11px] text-text-secondary">
        <input type="checkbox" disabled={disabled} checked={reference.include_audio !== false}
          onChange={event => patchReference(index, { include_audio: event.target.checked })} className="h-3.5 w-3.5 accent-accent-blue" />
        Include soundtrack
      </label>}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-text-secondary">
        <label className="inline-flex min-h-8 cursor-pointer items-center hover:text-text-primary focus-within:underline">
          Replace {reference.type}
          <input type="file" disabled={disabled || uploading} aria-label={`Replace ${labels[index]}`} className="sr-only"
            onChange={event => { void replaceReference(reference, event.target.files?.[0]); event.currentTarget.value = '' }} />
        </label>
        {reference.type === 'video' && <>
          <label className="inline-flex min-h-8 cursor-pointer items-center hover:text-text-primary focus-within:underline">
            {reference.audio_path ? 'Replace audio' : 'Attach audio'}
            {/* Browse freely for iOS document providers; attachAudio validates the file. */}
            <input type="file" disabled={disabled || uploading} aria-label={`Attach audio to ${labels[index]}`} className="sr-only"
              onChange={event => { void attachAudio(reference.id, event.target.files?.[0]); event.currentTarget.value = '' }} />
          </label>
          {reference.audio_path && <button type="button" disabled={disabled} title="Remove attached soundtrack"
            onClick={() => patchReference(index, { audio_path: undefined, audio_filename: undefined, audio_duration_seconds: undefined, include_audio: reference.has_audio === true })}
            className="flex min-h-8 min-w-0 items-center gap-1 text-text-muted hover:text-indicator-error">
            <X size={12} className="shrink-0" /><span className="truncate">{reference.audio_filename || 'Attached audio'}</span>
          </button>}
        </>}
      </div>
    </div>
  }

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">
            {scope === 'director' ? 'Omni References' : 'Scene references'}
          </label>
          <span
            title={scope === 'director'
              ? 'Order matters. These references are attached to every H3 Omni shot. Picture, Video, and Audio labels are assigned from top to bottom and can be named in the project description. A Music / performance timeline becomes Director’s exact audio driver.'
              : 'References is the dedicated H3 Omni workspace. Add people, scenes, motion, voices, styles, or an exact music / performance timeline here; use Frames for LTX audio-driven video.'}
            className="text-text-muted cursor-help"
          >
            <Info size={12} />
          </span>
        </div>
        <span className="text-[9px] text-text-muted">{references.length}/{limits.total}</span>
      </div>

      <CharacterToolbarItem>
        <button
          type="button"
          disabled={disabled}
          aria-expanded={libraryOpen}
          aria-label={`Characters${addedCharacterIds.length ? `, ${addedCharacterIds.length} selected` : ''}`}
          title={`${addedCharacterIds.length} saved characters selected`}
          onClick={() => setLibraryOpen(open => !open)}
          className="min-h-12 w-full flex items-center justify-between gap-2 rounded-xl border border-border bg-bg-tertiary px-3 py-2 text-left hover:border-border-light disabled:opacity-50"
        >
          <span className="flex items-center gap-2 text-xs font-semibold text-text-primary">
            <UserRound size={15} className="text-accent-blue" />
            <span className="studio-character-label">Characters<span className="studio-character-detail block text-[10px] font-normal text-text-muted">{addedCharacterIds.length ? `${addedCharacterIds.length} selected` : 'Choose saved'}</span></span>
          </span>
          <span className="flex items-center gap-2">
            <ChevronDown size={14} className={`text-text-muted transition-transform ${libraryOpen ? 'rotate-180' : ''}`} />
          </span>
        </button>
      </CharacterToolbarItem>
      <SidebarDialog open={libraryOpen} title="Characters" variant="library" onClose={() => setLibraryOpen(false)}>
          <div className="border-t border-border p-2.5 space-y-3">
            <p className="text-[11px] leading-relaxed text-text-secondary">
              Choose who appears in your scene. Their saved voice comes with them.
            </p>
            <ReferenceCharacterPicker characters={characters} addedIds={addedCharacterIds} disabled={disabled} scroll={false}
              onAdd={addCharacter} onDelete={character => void removeCharacter(character)} />
            {scope === 'studio' && <button type="button" onClick={() => { setLibraryOpen(false); window.dispatchEvent(new Event('maestro-open-finishing')) }}
              className="min-h-10 w-full rounded-lg border border-border px-3 text-left text-xs text-text-secondary hover:bg-bg-hover">Face refinement & character mapping…</button>}
            <div className="grid grid-cols-2 gap-2 border-t border-border pt-3">
              <ImportCharacterButton disabled={disabled} label="Import file"
                className="flex min-h-10 w-full items-center justify-center gap-1.5 rounded-lg border border-border text-[11px] font-medium text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-40" />
              <button type="button" disabled={disabled} onClick={() => setCharacterFormOpen(open => !open)}
                className="flex min-h-10 items-center justify-center gap-1.5 rounded-lg border border-border text-[11px] font-medium text-text-secondary hover:bg-bg-hover hover:text-text-primary disabled:opacity-40">
                <UserPlus size={13} /> {characterFormOpen ? 'Close form' : 'New character'}
              </button>
            </div>

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
            {error && <p role="alert" className="text-xs text-indicator-error">{error}</p>}
          </div>
      </SidebarDialog>

      <div ref={referenceGrid} className="space-y-2">
        {referenceRows.map((row, rowIndex) => {
          const editedItem = row.find(item => referenceItemKey(item) === editingKey)
          const editedColumn = row.findIndex(item => referenceItemKey(item) === editingKey)
          const itemTitle = (item: ActiveReferenceItem) => item.kind === 'reference' ? labels[item.index] : characterDisplayName(
            characters.find(character => character.id === item.characterId)?.name
            || item.entries.find(entry => entry.reference.type !== 'audio')?.reference.character_name
            || item.entries[0]?.reference.character_name
            || item.entries[0]?.reference.role
            || 'Saved character'
          )
          return <div key={rowIndex} className="grid grid-cols-3 items-start gap-2">
            {row.map((item, column) => {
              const key = referenceItemKey(item)
              const itemIndex = rowIndex * 3 + column
              const character = item.kind === 'character' ? characters.find(candidate => candidate.id === item.characterId) : undefined
              const reference = item.kind === 'character' ? item.entries.find(entry => entry.reference.type !== 'audio')?.reference : item.reference
              const entryIndices = item.kind === 'character' ? item.entries.map(entry => entry.index) : [item.index]
              const subtitle = item.kind === 'character'
                ? `${item.entries.map(entry => labels[entry.index]).join(' + ')}${item.entries.some(entry => entry.reference.type === 'audio') ? ' · Voice' : ''}`
                : reference?.role || reference?.filename
              const preview = item.kind === 'character'
                ? character?.visual.thumbnail_url || (reference?.type === 'image' ? reference.url : `/api/v1/characters/${encodeURIComponent(item.characterId)}/media/thumbnail`)
                : reference?.type === 'image' ? reference.url : undefined
              return <MediaInputCard key={key} title={itemTitle(item)} subtitle={subtitle} disabled={disabled}
                kind={reference?.type || 'image'} mediaUrl={reference?.url} preview={preview}
                expanded={editingKey === key} editorId={editorId} onEdit={() => setEditingKey(current => current === key ? null : key)}
                draggable={!disabled} onDragStart={event => {
                  event.dataTransfer.effectAllowed = 'move'
                  event.dataTransfer.setData('application/x-maestro-reference', key)
                  setDragIndices(entryIndices)
                }} onDragEnd={() => setDragIndices(null)}
                onDragOver={event => { if (!disabled && dragIndices) { event.preventDefault(); event.dataTransfer.dropEffect = 'move' } }}
                onDrop={event => {
                  event.preventDefault()
                  event.stopPropagation()
                  if (!disabled && dragIndices) reorder(dragIndices, entryIndices[0])
                  setDragIndices(null)
                }}
                onRemove={() => {
                  if (editingKey === key) setEditingKey(null)
                  update(references.filter((_, index) => !entryIndices.includes(index)))
                }}
                onEarlier={itemIndex > 0 && !disabled ? () => moveItem(itemIndex, -1) : undefined}
                onLater={itemIndex < activeItems.length - 1 && !disabled ? () => moveItem(itemIndex, 1) : undefined} />
            })}
            {/* No mixed-media accept filter: iOS greys out valid audio files.
                addFiles validates type and both per-type and total budgets. */}
            {canAddMore && rowIndex === referenceRows.length - 1 && <MediaAddTile disabled={disabled} busy={uploading} hint="Image, video or audio" onFiles={files => void addFiles(files)} />}
            {editedItem && <div id={editorId} role="group" aria-label={`${itemTitle(editedItem)} reference settings`}
              onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeEditor() } }}
              className="relative col-span-3 min-w-0 rounded-xl border border-accent-blue/40 bg-bg-tertiary p-2.5">
              <span aria-hidden="true" className="pointer-events-none absolute -top-[5px] h-2 w-2 -translate-x-1/2 rotate-45 border-l border-t border-accent-blue/40 bg-bg-tertiary"
                style={{ left: `${(editedColumn + 0.5) * 100 / 3}%` }} />
              <div className="mb-1.5 flex min-w-0 items-center justify-between gap-2">
                <span className="truncate text-[11px] font-medium text-text-secondary">{itemTitle(editedItem)}</span>
                <button type="button" aria-label="Close reference settings" onClick={closeEditor}
                  className="-mr-1 -mt-1 rounded-lg p-2 text-text-muted hover:bg-bg-hover hover:text-text-primary"><X size={13} /></button>
              </div>
              {referenceFields(editedItem)}
            </div>}
          </div>
        })}
      </div>
      {!canAddMore && <p className="text-[10px] text-text-muted">All reference slots are filled. Remove or replace an input to change this scene.</p>}

      {references.length > 0 && scope !== 'studio' && (
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
