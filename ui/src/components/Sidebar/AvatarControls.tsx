import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from 'react'
import { ArrowLeftRight, ImagePlus, Mic, RotateCcw, Upload, X } from 'lucide-react'
import * as api from '../../api/client'
import { useStore } from '../../stores/useStore'
import { GalleryInput } from '../shared/GalleryInput'
import {
  DEFAULT_AVATAR_SPEAKER_LOCATIONS,
  formatAvatarSpeakerRegions,
  isLongCatAvatarModel,
  isMultiSpeakerAvatarModel,
  parseAvatarSpeakerRegions,
  type AvatarSpeakerRegion,
} from '../../lib/avatarWorkflow'

type AudioSlot = 'voice1' | 'voice2'
type SpeakerField = keyof AvatarSpeakerRegion
type SpeakerDraft = Record<SpeakerField, string>
type StoreSnapshot = ReturnType<typeof useStore.getState>
type AnchorSource = { file: File; name: string } | { path: string; url: string; name: string }

const AUDIO_SLOT_KEYS: Record<AudioSlot, 'audio_guide' | 'audio_guide2'> = {
  voice1: 'audio_guide',
  voice2: 'audio_guide2',
}

const DEFAULT_SPEAKER_DRAFTS: [SpeakerDraft, SpeakerDraft] = [
  { left: '0', top: '0', right: '50', bottom: '100' },
  { left: '50', top: '0', right: '100', bottom: '100' },
]

function basename(value: string): string {
  return value.replace(/\\/g, '/').split('/').filter(Boolean).pop() || value
}

function firstPath(value: unknown): string | null {
  if (typeof value === 'string' && value.trim()) return value
  if (Array.isArray(value)) {
    const first = value.find(item => typeof item === 'string' && item.trim())
    return typeof first === 'string' ? first : null
  }
  return null
}

function imageReferencesAreActive(state: StoreSnapshot): boolean {
  const promptType = String(state.params.video_prompt_type || '')
  const referencesAllowed = !promptType.includes('F')
    && !String(state.params.frames_positions || '').trim()
  if (!referencesAllowed) return false
  const referenceChoices = state.modelOptions?.image_ref_choices?.choices ?? []
  const effectiveRefType = state.imageRefType || (
    referenceChoices.some(([, value]) => value.includes('K'))
      ? 'KI'
      : referenceChoices.some(([, value]) => value === 'I')
        ? 'I'
        : referenceChoices[0]?.[1] || ''
  )
  if (state.imageRefs.length > 0 && effectiveRefType.includes('I')) return true

  return Array.isArray(state.params.image_refs)
    && state.params.image_refs.some(path => typeof path === 'string' && path.trim())
    && promptType.includes('I')
}

function resolveAnchorSource(state: StoreSnapshot): AnchorSource | null {
  if (state.startImage) return { file: state.startImage, name: state.startImage.name }

  const startPath = firstPath(state.params.image_start)
  if (startPath) {
    const name = basename(startPath)
    return { path: startPath, url: api.getFileUrl(name), name }
  }

  if (!imageReferencesAreActive(state)) return null
  const referenceFile = state.imageRefs[0]
  if (referenceFile) return { file: referenceFile, name: referenceFile.name }

  const referencePath = firstPath(state.params.image_refs)
  if (!referencePath) return null
  const name = basename(referencePath)
  return { path: referencePath, url: api.getFileUrl(name), name }
}

function draftsFromStoredValue(value: unknown): [SpeakerDraft, SpeakerDraft] {
  const valid = parseAvatarSpeakerRegions(value)
  if (valid) {
    return valid.map(region => ({
      left: String(region.left),
      top: String(region.top),
      right: String(region.right),
      bottom: String(region.bottom),
    })) as [SpeakerDraft, SpeakerDraft]
  }

  if (typeof value === 'string') {
    const boxes = value.trim().split(/\s+/)
    if (boxes.length === 2) {
      const parsed = boxes.map(box => {
        const parts = box.split(':')
        if (parts.length === 4) return parts
        if (parts.length === 2) return [parts[0], '0', parts[1], '100']
        return null
      })
      if (parsed[0] && parsed[1]) {
        const first = parsed[0]
        const second = parsed[1]
        return [
          { left: first[0], top: first[1], right: first[2], bottom: first[3] },
          { left: second[0], top: second[1], right: second[2], bottom: second[3] },
        ]
      }
    }
  }
  return DEFAULT_SPEAKER_DRAFTS.map(region => ({ ...region })) as [SpeakerDraft, SpeakerDraft]
}

function serializeSpeakerDrafts(drafts: [SpeakerDraft, SpeakerDraft]): string {
  return drafts.map(region => `${region.left}:${region.top}:${region.right}:${region.bottom}`).join(' ')
}

function useFileObjectUrl(file: File | null): string | null {
  const [preview, setPreview] = useState<{ file: File; url: string } | null>(null)
  useEffect(() => {
    if (!file) return
    const nextUrl = URL.createObjectURL(file)
    // Object URLs are a browser resource whose lifetime must follow the file.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPreview({ file, url: nextUrl })
    return () => URL.revokeObjectURL(nextUrl)
  }, [file])
  return preview?.file === file ? preview.url : null
}

function clearActiveReferenceImages() {
  const state = useStore.getState()
  if (!imageReferencesAreActive(state)) return

  for (let index = state.imageRefs.length - 1; index >= 0; index -= 1) {
    useStore.getState().removeImageRef(index)
  }
  const current = useStore.getState()
  current.setParam('image_refs', undefined)
  current.setImageRefType('')
  if (String(current.params.video_prompt_type || '').includes('I')) {
    current.setParam('video_prompt_type', '')
  }
}

export function AvatarControls() {
  const models = useStore(state => state.models)
  const params = useStore(state => state.params)
  const modelType = String(params.model_type || '')
  const selectedModel = models.find(model => model.model_type === modelType) || { model_type: modelType }
  const isAvatarModel = isLongCatAvatarModel(selectedModel)
  const isMulti = isMultiSpeakerAvatarModel(selectedModel)
  const studioVideoWorkflow = useStore(state => state.studioVideoWorkflow)
  const startImage = useStore(state => state.startImage)
  const imageRefs = useStore(state => state.imageRefs)
  const imageRefType = useStore(state => state.imageRefType)
  const setParam = useStore(state => state.setParam)
  const audioGuideFilename = useStore(state => state.audioGuideFilename)
  const audioGuide2Filename = useStore(state => state.audioGuide2Filename)

  const [audioUploading, setAudioUploading] = useState<Record<AudioSlot, boolean>>({ voice1: false, voice2: false })
  const [audioErrors, setAudioErrors] = useState<Record<AudioSlot, string>>({ voice1: '', voice2: '' })
  const [anchorError, setAnchorError] = useState('')
  const [speakerDrafts, setSpeakerDrafts] = useState<[SpeakerDraft, SpeakerDraft]>(() =>
    draftsFromStoredValue(useStore.getState().params.speakers_locations))
  const [imageAspectRatio, setImageAspectRatio] = useState('4 / 3')
  const uploadEpochs = useRef<Record<AudioSlot, number>>({ voice1: 0, voice2: 0 })
  const active = useRef(false)
  const observedSpeakerValue = useRef<unknown>(useStore.getState().params.speakers_locations)
  const locallyWrittenSpeakerValue = useRef<string | null>(null)
  const anchorInput = useRef<HTMLInputElement>(null)
  const voice1Input = useRef<HTMLInputElement>(null)
  const voice2Input = useRef<HTMLInputElement>(null)

  const anchorSource = resolveAnchorSource({
    ...useStore.getState(), startImage, imageRefs, imageRefType, params,
  })
  const anchorFile = anchorSource && 'file' in anchorSource ? anchorSource.file : null
  const anchorFilePreviewUrl = useFileObjectUrl(anchorFile)
  const anchorPreviewUrl = anchorFilePreviewUrl || (anchorSource && 'url' in anchorSource ? anchorSource.url : null)

  const voice1Path = String(params.audio_guide || '')
  const voice2Path = String(params.audio_guide2 || '')
  const voice1Label = audioGuideFilename || (voice1Path ? basename(voice1Path) : '')
  const voice2Label = audioGuide2Filename || (voice2Path ? basename(voice2Path) : '')

  useEffect(() => {
    active.current = true
    const epochs = uploadEpochs.current
    const unsubscribe = useStore.subscribe((next, previous) => {
      const contextChanged = next.params.model_type !== previous.params.model_type
        || next.generationMode !== previous.generationMode
        || next.studioVideoWorkflow !== previous.studioVideoWorkflow
        || next.sidebarMode !== previous.sidebarMode
      if (!contextChanged) return
      epochs.voice1 += 1
      epochs.voice2 += 1
      if (active.current) {
        setAudioUploading({ voice1: false, voice2: false })
        setAudioErrors({ voice1: '', voice2: '' })
      }
    })
    return () => {
      active.current = false
      epochs.voice1 += 1
      epochs.voice2 += 1
      unsubscribe()
    }
  }, [])

  useEffect(() => {
    if (!isMulti || studioVideoWorkflow !== 'avatar') return
    if (useStore.getState().params.speakers_locations == null) {
      locallyWrittenSpeakerValue.current = DEFAULT_AVATAR_SPEAKER_LOCATIONS
      setSpeakerDrafts(draftsFromStoredValue(DEFAULT_AVATAR_SPEAKER_LOCATIONS))
      setParam('speakers_locations', DEFAULT_AVATAR_SPEAKER_LOCATIONS)
    }
  }, [isMulti, studioVideoWorkflow, setParam])

  useEffect(() => {
    if (params.speakers_locations === observedSpeakerValue.current) return
    observedSpeakerValue.current = params.speakers_locations
    if (params.speakers_locations === locallyWrittenSpeakerValue.current) return
    setSpeakerDrafts(draftsFromStoredValue(params.speakers_locations))
  }, [params.speakers_locations])

  const writeSpeakerDrafts = useCallback((nextDrafts: [SpeakerDraft, SpeakerDraft]) => {
    setSpeakerDrafts(nextDrafts)
    const candidate = serializeSpeakerDrafts(nextDrafts)
    const parsed = parseAvatarSpeakerRegions(candidate)
    const value = parsed ? formatAvatarSpeakerRegions(parsed) : candidate
    locallyWrittenSpeakerValue.current = value
    setParam('speakers_locations', value)
  }, [setParam])

  const updateSpeakerField = (speakerIndex: 0 | 1, field: SpeakerField, value: string) => {
    const nextDrafts = speakerDrafts.map(region => ({ ...region })) as [SpeakerDraft, SpeakerDraft]
    nextDrafts[speakerIndex] = { ...nextDrafts[speakerIndex], [field]: value }
    writeSpeakerDrafts(nextDrafts)
  }

  const currentSpeakerValue = params.speakers_locations
  const parsedSpeakerRegions = parseAvatarSpeakerRegions(currentSpeakerValue)
  const displaySpeakerRegions = currentSpeakerValue == null
    ? parseAvatarSpeakerRegions(DEFAULT_AVATAR_SPEAKER_LOCATIONS)
    : parsedSpeakerRegions
  const invalidSpeakerRegions = isMulti && currentSpeakerValue != null && !parsedSpeakerRegions

  const handleAnchorFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setAnchorError('Choose an image file for the anchor.')
      return false
    }
    setAnchorError('')
    clearActiveReferenceImages()
    const state = useStore.getState()
    state.setParam('image_start', undefined)
    state.setStartImage(file)
    return true
  }

  const clearAnchor = () => {
    setAnchorError('')
    useStore.getState().setStartImage(null)
    clearActiveReferenceImages()
  }

  const handleAnchorInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (file) handleAnchorFile(file)
  }

  const handleAnchorDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) handleAnchorFile(file)
  }

  const getAnchorGalleryImages = useCallback(() => {
    const source = resolveAnchorSource(useStore.getState())
    if (!source) return []
    if ('file' in source) return [source.file]
    return [{ url: source.url, name: source.name }]
  }, [])

  const clearAudio = (slot: AudioSlot) => {
    uploadEpochs.current[slot] += 1
    setAudioUploading(previous => ({ ...previous, [slot]: false }))
    setAudioErrors(previous => ({ ...previous, [slot]: '' }))
    const state = useStore.getState()
    state.setParam(AUDIO_SLOT_KEYS[slot], undefined)
    if (slot === 'voice1') state.setAudioGuideFilename(null)
    else state.setAudioGuide2Filename(null)
  }

  const uploadAudio = async (file: File, slot: AudioSlot): Promise<boolean> => {
    const stateAtStart = useStore.getState()
    const modelAtStart = stateAtStart.models.find(model => model.model_type === stateAtStart.params.model_type)
      || { model_type: String(stateAtStart.params.model_type || '') }
    if (!isLongCatAvatarModel(modelAtStart)
      || stateAtStart.studioVideoWorkflow !== 'avatar'
      || stateAtStart.generationMode !== 'video'
      || (slot === 'voice2' && !isMultiSpeakerAvatarModel(modelAtStart))) return false
    if (!file.type.startsWith('audio/')) {
      setAudioErrors(previous => ({ ...previous, [slot]: 'Choose an audio file for this voice.' }))
      return false
    }

    const modelTypeAtStart = String(stateAtStart.params.model_type || '')
    const workflowAtStart = stateAtStart.studioVideoWorkflow
    const modeAtStart = stateAtStart.generationMode
    const sidebarAtStart = stateAtStart.sidebarMode
    const ticket = uploadEpochs.current[slot] + 1
    uploadEpochs.current[slot] = ticket
    const isCurrent = () => {
      const current = useStore.getState()
      const currentModel = current.models.find(model => model.model_type === current.params.model_type)
        || { model_type: String(current.params.model_type || '') }
      return active.current
        && uploadEpochs.current[slot] === ticket
        && String(current.params.model_type || '') === modelTypeAtStart
        && current.studioVideoWorkflow === workflowAtStart
        && current.generationMode === modeAtStart
        && current.sidebarMode === sidebarAtStart
        && isLongCatAvatarModel(currentModel)
        && (slot !== 'voice2' || isMultiSpeakerAvatarModel(currentModel))
    }

    setAudioUploading(previous => ({ ...previous, [slot]: true }))
    setAudioErrors(previous => ({ ...previous, [slot]: '' }))
    const replacing = useStore.getState()
    replacing.setParam(AUDIO_SLOT_KEYS[slot], undefined)
    if (slot === 'voice1') replacing.setAudioGuideFilename(null)
    else replacing.setAudioGuide2Filename(null)
    try {
      const uploaded = await api.uploadAudio(file)
      if (!isCurrent()) return false
      const current = useStore.getState()
      current.setParam(AUDIO_SLOT_KEYS[slot], uploaded.path)
      if (slot === 'voice1') current.setAudioGuideFilename(file.name)
      else current.setAudioGuide2Filename(file.name)
      return true
    } catch (error) {
      if (isCurrent()) {
        setAudioErrors(previous => ({
          ...previous,
          [slot]: error instanceof Error ? error.message : 'Audio upload failed.',
        }))
      }
      return false
    } finally {
      if (active.current && uploadEpochs.current[slot] === ticket) {
        setAudioUploading(previous => ({ ...previous, [slot]: false }))
      }
    }
  }

  const handleAudioInput = (slot: AudioSlot, event: ChangeEvent<HTMLInputElement>) => {
    const file = event.currentTarget.files?.[0]
    event.currentTarget.value = ''
    if (file) void uploadAudio(file, slot)
  }

  const handleAudioDrop = (slot: AudioSlot, event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) void uploadAudio(file, slot)
  }

  const swapSpeakerRegions = () => {
    if (!parseAvatarSpeakerRegions(serializeSpeakerDrafts(speakerDrafts))) return
    writeSpeakerDrafts([speakerDrafts[1], speakerDrafts[0]])
  }

  const resetSpeakerRegions = () => {
    writeSpeakerDrafts(DEFAULT_SPEAKER_DRAFTS.map(region => ({ ...region })) as [SpeakerDraft, SpeakerDraft])
  }

  if (!isAvatarModel || studioVideoWorkflow !== 'avatar') return null

  return (
    <section aria-label="Avatar inputs" className="space-y-3">
      <div className="space-y-2">
        <div>
          <h3 className="text-xs font-semibold text-text-primary">Anchor image</h3>
          <p className="mt-0.5 text-[10px] text-text-muted">Choose the image LongCat should animate.</p>
        </div>
        <div
          onDragOver={event => event.preventDefault()}
          onDrop={handleAnchorDrop}
          className="overflow-hidden rounded-xl border border-dashed border-border bg-bg-tertiary/50 p-2 hover:border-border-light"
        >
          {anchorPreviewUrl ? (
            <>
              <div className="relative mx-auto w-full overflow-hidden rounded-lg bg-black" style={{ aspectRatio: imageAspectRatio }}>
                <img
                  src={anchorPreviewUrl}
                  alt="Anchor image preview"
                  onLoad={event => {
                    const image = event.currentTarget
                    if (image.naturalWidth > 0 && image.naturalHeight > 0) {
                      setImageAspectRatio(image.naturalWidth + ' / ' + image.naturalHeight)
                    }
                  }}
                  className="absolute inset-0 h-full w-full object-contain"
                />
                {isMulti && displaySpeakerRegions?.map((region, index) => (
                  <div
                    key={index}
                    className={index === 0
                      ? 'pointer-events-none absolute border-2 border-sky-300 bg-sky-400/15'
                      : 'pointer-events-none absolute border-2 border-amber-300 bg-amber-400/15'}
                    style={{
                      left: region.left + '%',
                      top: region.top + '%',
                      width: (region.right - region.left) + '%',
                      height: (region.bottom - region.top) + '%',
                    }}
                  >
                    <span className={index === 0
                      ? 'absolute left-0 top-0 rounded-br bg-sky-700/90 px-1 py-0.5 text-[9px] font-semibold text-white'
                      : 'absolute left-0 top-0 rounded-br bg-amber-700/90 px-1 py-0.5 text-[9px] font-semibold text-white'}>
                      Voice {index + 1}
                    </span>
                  </div>
                ))}
              </div>
              <div className="mt-2 flex min-w-0 items-center gap-2">
                <ImagePlus size={14} className="shrink-0 text-text-muted" />
                <span className="min-w-0 flex-1 truncate text-[11px] text-text-primary" title={anchorSource?.name}>
                  {anchorSource?.name || 'Anchor image'}
                </span>
                <label className="shrink-0 cursor-pointer rounded-md px-2 py-1 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary">
                  Change
                  <input ref={anchorInput} type="file" accept="image/*" aria-label="Choose anchor image" className="sr-only" onChange={handleAnchorInput} />
                </label>
                <button type="button" aria-label="Remove anchor image" onClick={clearAnchor}
                  className="shrink-0 rounded-md p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary">
                  <X size={13} />
                </button>
              </div>
            </>
          ) : (
            <label className="flex min-h-24 cursor-pointer flex-col items-center justify-center gap-1 rounded-lg p-3 text-center text-text-secondary hover:bg-bg-hover">
              <input ref={anchorInput} type="file" accept="image/*" aria-label="Choose anchor image" className="sr-only" onChange={handleAnchorInput} />
              <Upload size={16} className="text-text-muted" />
              <span className="text-[11px]">Drop an image here or choose an anchor</span>
              <span className="text-[10px] text-text-muted">PNG, JPEG, or WebP</span>
            </label>
          )}
        </div>
        <GalleryInput kind="image" label="avatar anchor image" onFile={handleAnchorFile} getImages={getAnchorGalleryImages} />
        {anchorError && <p role="alert" className="text-[10px] text-red-400">{anchorError}</p>}
      </div>

      <div className="space-y-2">
        <h3 className="text-xs font-semibold text-text-primary">Voice audio</h3>
        <AudioUploadInput
          slot="voice1"
          filename={voice1Label}
          uploading={audioUploading.voice1}
          error={audioErrors.voice1}
          inputRef={voice1Input}
          onDrop={event => handleAudioDrop('voice1', event)}
          onChange={event => handleAudioInput('voice1', event)}
          onClear={() => clearAudio('voice1')}
        />
        <GalleryInput
          kind="audio"
          label="Voice audio #1"
          disabledReason={audioUploading.voice1 ? 'Wait for the current voice upload to finish.' : undefined}
          onFile={file => uploadAudio(file, 'voice1')}
        />
        {isMulti && (
          <>
            <AudioUploadInput
              slot="voice2"
              filename={voice2Label}
              uploading={audioUploading.voice2}
              error={audioErrors.voice2}
              inputRef={voice2Input}
              onDrop={event => handleAudioDrop('voice2', event)}
              onChange={event => handleAudioInput('voice2', event)}
              onClear={() => clearAudio('voice2')}
            />
            <GalleryInput
              kind="audio"
              label="Voice audio #2"
              disabledReason={audioUploading.voice2 ? 'Wait for the current voice upload to finish.' : undefined}
              onFile={file => uploadAudio(file, 'voice2')}
            />
          </>
        )}
      </div>

      {isMulti && (
        <section aria-label="Speaker regions" className="space-y-2 rounded-xl border border-border bg-bg-tertiary/40 p-3">
          <div className="flex items-start justify-between gap-2">
            <div>
              <h3 className="text-xs font-semibold text-text-primary">Speaker regions</h3>
              <p className="mt-0.5 text-[10px] text-text-muted">Voice 1 uses the first box; Voice 2 uses the second.</p>
            </div>
            <div className="flex shrink-0 gap-1">
              <button type="button" onClick={swapSpeakerRegions} disabled={!parseAvatarSpeakerRegions(serializeSpeakerDrafts(speakerDrafts))}
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] text-text-secondary hover:bg-bg-hover disabled:opacity-40">
                <ArrowLeftRight size={12} /> Swap
              </button>
              <button type="button" onClick={resetSpeakerRegions}
                className="inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] text-text-secondary hover:bg-bg-hover">
                <RotateCcw size={12} /> Reset
              </button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {speakerDrafts.map((region, speakerIndex) => (
              <fieldset key={speakerIndex} className="min-w-0 rounded-lg border border-border p-2">
                <legend className="px-1 text-[10px] font-medium text-text-primary">Voice {speakerIndex + 1}</legend>
                <div className="grid grid-cols-2 gap-1.5">
                  {(['left', 'top', 'right', 'bottom'] as SpeakerField[]).map(field => (
                    <label key={field} className="min-w-0 text-[9px] text-text-muted">
                      {field[0].toUpperCase() + field.slice(1)} (%)
                      <input
                        type="number"
                        inputMode="decimal"
                        min={0}
                        max={100}
                        step="any"
                        value={region[field]}
                        aria-label={`Voice ${speakerIndex + 1} ${field[0].toUpperCase() + field.slice(1)} (%)`}
                        aria-invalid={invalidSpeakerRegions}
                        onChange={event => updateSpeakerField(speakerIndex as 0 | 1, field, event.currentTarget.value)}
                        className="mt-0.5 w-full rounded-md border border-border bg-bg-primary px-1.5 py-1 text-[11px] tabular-nums text-text-primary outline-none focus:border-accent-blue"
                      />
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}
          </div>
          {invalidSpeakerRegions && (
            <p role="alert" className="text-[10px] text-red-400">
              Speaker regions must stay within 0–100 and cover part of the image interior.
            </p>
          )}
        </section>
      )}
    </section>
  )
}

function AudioUploadInput({ slot, filename, uploading, error, inputRef, onDrop, onChange, onClear }: {
  slot: AudioSlot
  filename: string
  uploading: boolean
  error: string
  inputRef: { current: HTMLInputElement | null }
  onDrop: (event: DragEvent<HTMLDivElement>) => void
  onChange: (event: ChangeEvent<HTMLInputElement>) => void
  onClear: () => void
}) {
  const inputLabel = slot === 'voice1' ? 'Choose Voice audio #1' : 'Choose Voice audio #2'
  const cancelLabel = slot === 'voice1' ? 'Cancel Voice audio #1 upload' : 'Cancel Voice audio #2 upload'
  const removeLabel = slot === 'voice1' ? 'Remove Voice audio #1' : 'Remove Voice audio #2'
  return (
    <div
      onDragOver={event => event.preventDefault()}
      onDrop={onDrop}
      className="flex min-w-0 items-center gap-2 rounded-lg border border-border bg-bg-tertiary px-3 py-2"
    >
      <Mic size={14} className="shrink-0 text-text-muted" />
      {filename ? (
        <span className="min-w-0 flex-1 truncate text-xs text-text-primary" title={filename}>{filename}</span>
      ) : (
        <label className="min-w-0 flex-1 cursor-pointer truncate text-[11px] text-text-secondary hover:text-text-primary">
          {uploading ? 'Uploading…' : 'Drop audio or choose a file'}
          <input ref={inputRef} type="file" accept="audio/*" aria-label={inputLabel} className="sr-only" onChange={onChange} />
        </label>
      )}
      {filename && (
        <label className="shrink-0 cursor-pointer rounded-md px-2 py-1 text-[10px] text-text-secondary hover:bg-bg-hover hover:text-text-primary">
          Change
          <input ref={inputRef} type="file" accept="audio/*" aria-label={inputLabel} className="sr-only" onChange={onChange} />
        </label>
      )}
      {uploading && <span role="status" className="shrink-0 text-[10px] text-text-muted">Uploading…</span>}
      {(filename || uploading) && (
        <button type="button" aria-label={uploading && !filename ? cancelLabel : removeLabel} onClick={onClear}
          className="shrink-0 rounded-md p-1 text-text-muted hover:bg-bg-hover hover:text-text-primary">
          <X size={13} />
        </button>
      )}
      {error && <span role="alert" className="basis-full text-[10px] text-red-400">{error}</span>}
    </div>
  )
}
