import { useRef, useCallback, useState } from 'react'
import { Upload, X, UserRoundPen, Loader2, Eye, Plus } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { RecastCharacterMapping } from '../../types'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import { InfoTooltip } from './InfoTooltip'
import { ScailResolutionSelector } from './ScailResolutionSelector'
import * as api from '../../api/client'

const MAPPING_COLORS = ['#0000ff', '#ff0000', '#00c853', '#ff00ff', '#00cfd1']
const MAPPING_LABELS = ['A', 'B', 'C', 'D', 'E']

type ReferencePreview = NonNullable<
  Awaited<ReturnType<typeof api.recastPreview>>['reference_previews']
>[number]
type MappingPreviewResult = NonNullable<
  Awaited<ReturnType<typeof api.recastPreview>>['mapping_results']
>[number]

function formatTimelineTime(seconds?: number | null) {
  if (seconds == null || !Number.isFinite(seconds)) return ''
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${minutes}:${remainder.toFixed(1).padStart(4, '0')}`
}

function emptyMapping(index: number): RecastCharacterMapping {
  return {
    id: `recast-${Date.now()}-${index}`,
    target: '',
    refFile: null,
    refPath: '',
    refUrl: '',
    additionalRefs: [],
    referenceAlignedToSource: false,
  }
}

/**
 * Recast sub-mode — deterministic SCAIL-2 source-person → reference mapping.
 * Every card owns one stable semantic color; additional images on that card
 * reuse the same color and therefore describe another view of the same person.
 */
export function RecastControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoPath = useStore(s => s.editVideoPath)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)
  const mappings = useStore(s => s.editRecastMappings)
  const setMappings = useStore(s => s.setEditRecastMappings)
  const useRelighting = useStore(s => s.editRecastUseRelighting)
  const resolutionProfile = useStore(s => s.editRecastResolutionProfile)

  const [previewImg, setPreviewImg] = useState<string | null>(null)
  const [previewState, setPreviewState] = useState<'idle' | 'loading' | 'found' | 'notfound' | 'error'>('idle')
  const [previewMatch, setPreviewMatch] = useState<{ matched: number; requested: number } | null>(null)
  const [mappingResults, setMappingResults] = useState<MappingPreviewResult[]>([])
  const [referencePreviews, setReferencePreviews] = useState<ReferencePreview[]>([])
  const [previewError, setPreviewError] = useState('')
  const videoFileRef = useRef<HTMLInputElement>(null)

  const resetPreview = useCallback(() => {
    setPreviewImg(null)
    setPreviewState('idle')
    setPreviewMatch(null)
    setMappingResults([])
    setReferencePreviews([])
    setPreviewError('')
  }, [])

  const updateMapping = useCallback((index: number, patch: Partial<RecastCharacterMapping>) => {
    const next = useStore.getState().editRecastMappings.map((mapping, mappingIndex) => (
      mappingIndex === index ? { ...mapping, ...patch } : mapping
    ))
    setMappings(next)
    resetPreview()
  }, [resetPreview, setMappings])

  const handleVideoUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      const url = URL.createObjectURL(file)
      const video = document.createElement('video')
      video.src = url
      video.onloadedmetadata = () => {
        const duration = video.duration && isFinite(video.duration) ? video.duration : 0
        const resolution = `${video.videoWidth}x${video.videoHeight}`
        setEditVideo(file, result.path, url, duration, resolution)
      }
      resetPreview()
    } catch {
      console.error('Failed to upload video')
    }
  }, [resetPreview, setEditVideo])

  const handlePrimaryUpload = useCallback(async (mappingIndex: number, file: File) => {
    try {
      const result = await api.uploadImage(file)
      updateMapping(mappingIndex, {
        refFile: file,
        refPath: result.path,
        refUrl: URL.createObjectURL(file),
        referenceAlignedToSource: false,
      })
    } catch {
      console.error('Failed to upload reference image')
    }
  }, [updateMapping])

  const handleAdditionalUploads = useCallback(async (mappingIndex: number, files: File[]) => {
    try {
      const available = Math.max(
        0,
        4 - (useStore.getState().editRecastMappings[mappingIndex]?.additionalRefs.length || 0),
      )
      const selected = files.slice(0, available)
      const uploaded = await Promise.all(selected.map(async file => {
        const result = await api.uploadImage(file)
        return { file, path: result.path, url: URL.createObjectURL(file) }
      }))
      const current = useStore.getState().editRecastMappings[mappingIndex]
      if (!current) return
      updateMapping(mappingIndex, {
        additionalRefs: [...current.additionalRefs, ...uploaded],
      })
    } catch {
      console.error('Failed to upload additional reference view')
    }
  }, [updateMapping])

  const handlePreview = useCallback(async () => {
    if (!editVideoPath || mappings.length === 0) return
    setPreviewState('loading')
    setPreviewError('')
    try {
      const res = await api.recastPreview({
        video_path: editVideoPath,
        character_mappings: mappings.map(mapping => ({
          id: mapping.id,
          target: mapping.target.trim() || 'person',
          ...(mapping.refPath ? { ref_image_path: mapping.refPath } : {}),
          additional_ref_image_paths: mapping.additionalRefs
            .map(reference => reference.path)
            .filter(Boolean),
          reference_aligned_to_source: mapping.referenceAlignedToSource,
        })),
        // Recast's identity preparation is intentionally automatic now.
        isolate_reference: true,
        auto_face_detail: true,
        resolution_profile: resolutionProfile,
        time: editStartTime,
        end_time: editEndTime,
      })
      setPreviewImg(res.preview)
      setPreviewMatch({ matched: res.matched_people, requested: res.requested_people })
      setMappingResults(res.mapping_results || [])
      setReferencePreviews(res.reference_previews || [])
      setPreviewState(res.found ? 'found' : 'notfound')
    } catch (error) {
      console.error('Recast preview failed:', error)
      setPreviewError(error instanceof Error ? error.message : 'Preview failed')
      setPreviewState('error')
    }
  }, [editVideoPath, mappings, editStartTime, editEndTime, resolutionProfile])

  const addMapping = useCallback(() => {
    if (mappings.length >= 5) return
    setMappings([...mappings, emptyMapping(mappings.length)])
    resetPreview()
  }, [mappings, resetPreview, setMappings])

  const removeMapping = useCallback((index: number) => {
    if (mappings.length <= 1) return
    setMappings(mappings.filter((_, mappingIndex) => mappingIndex !== index))
    resetPreview()
  }, [mappings, resetPreview, setMappings])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        <UserRoundPen size={13} className="shrink-0 text-accent-blue" />
        <span className="text-[10px] font-medium text-text-primary">
          Replace characters
        </span>
        <InfoTooltip
          placement="bottom"
          label="About Recast"
          text={"Add one card for each person you want to replace. Maestro isolates references, adds a face-detail view when useful, and preserves detected bystanders automatically.\n\nFast is recommended (8 steps). HQ uses the full 40-step schedule."}
        />
      </div>

      <ScailResolutionSelector
        value={resolutionProfile}
        workflow="Recast"
        onChange={profile => {
          useStore.setState({ editRecastResolutionProfile: profile })
          resetPreview()
        }}
      />

      {!editVideoFile ? (
        <div
          onDragOver={event => event.preventDefault()}
          onDrop={event => {
            event.preventDefault()
            const file = event.dataTransfer.files[0]
            if (file && file.type.startsWith('video/')) void handleVideoUpload(file)
          }}
          onClick={() => videoFileRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop a video or click to upload</p>
          <input
            ref={videoFileRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={event => {
              if (event.target.files?.[0]) void handleVideoUpload(event.target.files[0])
            }}
          />
        </div>
      ) : (
        <div className="relative">
          <button
            onClick={() => {
              clearEditVideo()
              resetPreview()
            }}
            className="absolute top-1.5 right-1.5 z-20 p-1 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
          >
            <X size={14} />
          </button>
          <VideoTimelineSelector
            videoUrl={editVideoUrl}
            duration={editVideoDuration}
            startTime={editStartTime}
            endTime={editEndTime}
            onStartChange={time => {
              useStore.setState({ editStartTime: time })
              resetPreview()
            }}
            onEndChange={time => {
              useStore.setState({ editEndTime: time })
              resetPreview()
            }}
          />
          <p className="text-[9px] text-text-muted mt-1 truncate">{editVideoFile.name}</p>
        </div>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1">
            <label className="text-[10px] text-text-muted uppercase tracking-wider">
              Characters ({mappings.length}/5)
            </label>
            <InfoTooltip
              label="About character mappings"
              text="Each card links one source person to one replacement. Its color stays fixed across all reference views so targets cannot swap."
            />
          </div>
          <button
            onClick={handlePreview}
            disabled={!editVideoPath || previewState === 'loading' || mappings.some(mapping => !mapping.target.trim())}
            className="flex items-center gap-1 px-2 py-1 rounded bg-bg-tertiary border border-border text-[9px] text-text-secondary hover:text-text-primary hover:border-accent-blue/50 transition-colors disabled:opacity-40"
            title="Scan the selected timeline and preview each character anchor"
          >
            {previewState === 'loading'
              ? <Loader2 size={11} className="animate-spin" />
              : <Eye size={11} />}
            Preview
          </button>
        </div>

        {mappings.map((mapping, mappingIndex) => {
          const mappingResult = mappingResults.find(result => result.mapping_index === mappingIndex)
          const previews = referencePreviews.filter(preview => preview.mapping_index === mappingIndex)
          return (
            <div key={mapping.id} className="rounded-lg border border-border bg-bg-secondary/40 p-2 space-y-2">
              <div className="flex items-center gap-1.5">
                <span
                  className="w-3 h-3 rounded-full border border-white/30 shrink-0"
                  style={{ backgroundColor: MAPPING_COLORS[mappingIndex] }}
                />
                <span className="text-[10px] font-medium text-text-primary">
                  Character {MAPPING_LABELS[mappingIndex]}
                </span>
                {mappings.length > 1 && (
                  <button
                    onClick={() => removeMapping(mappingIndex)}
                    className="ml-auto p-0.5 text-text-muted hover:text-status-error"
                    title="Remove this character mapping"
                  >
                    <X size={12} />
                  </button>
                )}
              </div>

              <div>
                <label className="text-[9px] text-text-muted block mb-0.5">Who to replace</label>
                <input
                  type="text"
                  value={mapping.target}
                  onChange={event => updateMapping(mappingIndex, { target: event.target.value })}
                  placeholder={mappingIndex === 0 ? 'woman in red' : 'man behind her'}
                  className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                />
                {mappingResult && (
                  <p className={`text-[9px] mt-0.5 ${mappingResult.found ? 'text-accent-green' : 'text-status-warning'}`}>
                    {mappingResult.found
                      ? `Found at ${formatTimelineTime(mappingResult.anchor_time_seconds)}.`
                      : 'Not found in the selected timeline.'}
                    {mappingResult.overlap_fraction > 0.02 && ` ${(mappingResult.overlap_fraction * 100).toFixed(0)}% overlaps an earlier card.`}
                  </p>
                )}
              </div>

              <div>
                <label className="text-[9px] text-text-muted block mb-0.5">Replacement image</label>
                {!mapping.refPath ? (
                  <label className="block border border-dashed border-border rounded p-3 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30">
                    <Upload size={15} className="mx-auto mb-1 text-text-muted" />
                    <span className="text-[9px] text-text-secondary">Upload character</span>
                    <input
                      type="file"
                      accept="image/*"
                      className="hidden"
                      onChange={event => {
                        if (event.target.files?.[0]) void handlePrimaryUpload(mappingIndex, event.target.files[0])
                        event.currentTarget.value = ''
                      }}
                    />
                  </label>
                ) : (
                  <div className="relative inline-block">
                    <img src={mapping.refUrl} alt={`Character ${MAPPING_LABELS[mappingIndex]} reference`} className="h-24 rounded border border-border" />
                    <button
                      onClick={() => updateMapping(mappingIndex, {
                        refFile: null,
                        refPath: '',
                        refUrl: '',
                        additionalRefs: [],
                        referenceAlignedToSource: false,
                      })}
                      className="absolute top-1 right-1 p-0.5 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80"
                    >
                      <X size={12} />
                    </button>
                  </div>
                )}
              </div>

              {mapping.refPath && !mapping.referenceAlignedToSource && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1">
                      <span className="text-[9px] text-text-muted">More views</span>
                      <InfoTooltip
                        label="About additional character views"
                        text="Optional close-up, side, or back views of this same character. Multi-reference is experimental; add only views that reveal details missing from the main image."
                      />
                    </div>
                    <span className="text-[8px] text-text-muted">
                      {mapping.additionalRefs.length}/4
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {mapping.additionalRefs.map((reference, referenceIndex) => (
                      <div key={`${reference.path}-${referenceIndex}`} className="relative">
                        <img src={reference.url} alt="Additional character view" className="h-14 w-14 object-cover rounded border border-border" />
                        <button
                          onClick={() => updateMapping(mappingIndex, {
                            additionalRefs: mapping.additionalRefs.filter((_, index) => index !== referenceIndex),
                          })}
                          className="absolute -top-1 -right-1 p-0.5 rounded-full bg-black/70 text-white"
                        >
                          <X size={9} />
                        </button>
                      </div>
                    ))}
                    {mapping.additionalRefs.length < 4 && (
                      <label className="h-14 w-14 flex flex-col items-center justify-center rounded border border-dashed border-border cursor-pointer hover:border-accent-blue/50 text-text-muted">
                        <Plus size={13} />
                        <span className="text-[7px]">view</span>
                        <input
                          type="file"
                          accept="image/*"
                          multiple
                          className="hidden"
                          onChange={event => {
                            if (event.target.files?.length) {
                              void handleAdditionalUploads(mappingIndex, Array.from(event.target.files))
                            }
                            event.currentTarget.value = ''
                          }}
                        />
                      </label>
                    )}
                  </div>
                </div>
              )}

              {previews.length > 0 && (
                <div className="space-y-1.5 border-t border-border pt-1.5">
                  <div className="flex items-center gap-1">
                    <p className="text-[9px] text-accent-green">Prepared references</p>
                    <InfoTooltip
                      label="About prepared references"
                      text="These are the identity, spatial-reference, and semantic-mask inputs that will be sent to SCAIL-2."
                    />
                  </div>
                  {previews.map(preview => (
                    <div
                      key={`${preview.view_index}-${preview.kind}`}
                      title={`Mask: ${preview.mask_source}${preview.detail_source ? ` · ${preview.detail_source}` : ''}`}
                    >
                      <p className="text-[8px] text-text-muted mb-0.5">
                        {preview.kind === 'primary'
                          ? 'Primary'
                          : preview.kind === 'auto_face_detail'
                            ? 'Face detail'
                            : `View ${preview.view_index}`} · {preview.prepared_size.join('×')}
                      </p>
                      <div className={`grid gap-1 ${preview.clip_identity_image ? 'grid-cols-3' : 'grid-cols-2'}`}>
                        {preview.clip_identity_image && (
                          <div>
                            <img src={preview.clip_identity_image} alt="Identity reference" className="w-full rounded border border-border" />
                            <span className="text-[7px] text-text-muted">identity</span>
                          </div>
                        )}
                        <div>
                          <img src={preview.prepared_image} alt="Prepared reference" className="w-full rounded border border-border" />
                          <span className="text-[7px] text-text-muted">reference</span>
                        </div>
                        <div>
                          <img src={preview.semantic_mask} alt="Semantic reference mask" className="w-full rounded border border-border" />
                          <span className="text-[7px] text-text-muted">mask</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )
        })}

        {mappings.length < 5 && (
          <button
            onClick={addMapping}
            className="w-full flex items-center justify-center gap-1 py-1.5 rounded border border-dashed border-border text-[9px] text-text-secondary hover:text-accent-blue hover:border-accent-blue/50"
          >
            <Plus size={11} />
            Add character
          </button>
        )}
      </div>

      {previewImg && (
        <div>
          <div className="mb-1 flex items-center gap-1">
            <p className="text-[9px] text-text-muted">Source selection</p>
            <InfoTooltip
              label="About source selection"
              text="Maestro scans the selected timeline. Each tile shows one character card at its strongest detected anchor; the same color remains assigned across shots."
            />
          </div>
          <img src={previewImg} alt="Target mapping preview" className="w-full rounded border border-border" />
          {previewState === 'found' && previewMatch && (
            <p className="text-[9px] text-accent-green mt-0.5">
              Found all {previewMatch.matched} mapped {previewMatch.matched === 1 ? 'person' : 'people'}.
            </p>
          )}
          {previewState === 'notfound' && (
            <p className="text-[9px] text-status-warning mt-0.5">
              Found {previewMatch?.matched ?? 0} of {previewMatch?.requested ?? mappings.length}. Refine the highlighted card description or adjust the selected range.
            </p>
          )}
        </div>
      )}
      {previewState === 'error' && (
        <p className="text-[9px] text-status-error">
          Preview failed{previewError ? ` — ${previewError}` : ' — check the server log.'}
        </p>
      )}

      <div className="flex items-center gap-1">
        <label className="flex cursor-pointer items-center gap-1.5 text-[9px] text-text-secondary">
          <input
            type="checkbox"
            checked={useRelighting}
            onChange={event => useStore.setState({ editRecastUseRelighting: event.target.checked })}
            className="accent-accent-blue"
          />
          <span>Match lighting</span>
        </label>
        <InfoTooltip
          placement="top"
          label="About matching replacement lighting"
          text="Uses Z.ai's official SCAIL-2 Relighting LoRA to blend lighting and shadows. It may affect nearby people in crowded shots. Select the same LoRA in Advanced to expose its single strength control."
        />
      </div>
    </div>
  )
}
