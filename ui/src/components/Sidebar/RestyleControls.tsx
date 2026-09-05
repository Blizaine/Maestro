import { useCallback, useRef, useState } from 'react'
import {
  ChevronDown,
  ChevronRight,
  Eye,
  Loader2,
  Paintbrush,
  Pencil,
  Plus,
  Upload,
  X,
} from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { RepaintRegionMapping } from '../../types'
import { VideoTimelineSelector } from '../shared/VideoTimelineSelector'
import { InfoTooltip } from './InfoTooltip'
import { ScailResolutionSelector } from './ScailResolutionSelector'
import * as api from '../../api/client'

const REGION_COLORS = ['#0000ff', '#ff0000', '#00c853', '#ff00ff', '#00cfd1']

function newRegion(index: number): RepaintRegionMapping {
  return {
    id: `repaint-${Date.now()}-${index}`,
    source: '',
    target: '',
  }
}

type PreviewResult = Awaited<ReturnType<typeof api.repaintPreview>>

/**
 * The internal sub-mode id remains `restyle` for saved-output compatibility,
 * but the product-facing workflow is Repaint: edit the selected first frame,
 * then animate it with the source video's exact SCAIL-2 motion path.
 */
export function RestyleControls() {
  const editVideoFile = useStore(s => s.editVideoFile)
  const editVideoPath = useStore(s => s.editVideoPath)
  const editVideoUrl = useStore(s => s.editVideoUrl)
  const editVideoDuration = useStore(s => s.editVideoDuration)
  const editStartTime = useStore(s => s.editStartTime)
  const editEndTime = useStore(s => s.editEndTime)
  const setEditVideo = useStore(s => s.setEditVideo)
  const clearEditVideo = useStore(s => s.clearEditVideo)
  const targetFramePath = useStore(s => s.editRepaintFramePath)
  const targetFrameUrl = useStore(s => s.editRepaintFrameUrl)
  const setTargetFrame = useStore(s => s.setEditRepaintFrame)
  const mappings = useStore(s => s.editRepaintMappings)
  const setMappings = useStore(s => s.setEditRepaintMappings)
  const resolutionProfile = useStore(s => s.editRepaintResolutionProfile)
  const sendFrameToImageMode = useStore(s => s.sendFrameToImageMode)

  const videoInputRef = useRef<HTMLInputElement>(null)
  const [regionDisclosure, setRegionDisclosure] = useState({
    mappingCount: mappings.length,
    open: mappings.length > 0,
  })
  const showRegions = regionDisclosure.mappingCount === mappings.length
    ? regionDisclosure.open
    : mappings.length > 0
  const [previewState, setPreviewState] = useState<'idle' | 'loading' | 'found' | 'error'>('idle')
  const [preview, setPreview] = useState<PreviewResult | null>(null)
  const [previewError, setPreviewError] = useState('')

  const resetPreview = useCallback(() => {
    setPreviewState('idle')
    setPreview(null)
    setPreviewError('')
  }, [])

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
        // An edited frame is composition-specific. A new source must not
        // silently reuse a target made for the previous video.
        setTargetFrame(null, '', '')
      }
      resetPreview()
    } catch (error) {
      console.error('Repaint source upload failed:', error)
    }
  }, [resetPreview, setEditVideo, setTargetFrame])

  const handleTargetUpload = useCallback(async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      setTargetFrame(file, result.path, URL.createObjectURL(file))
      resetPreview()
    } catch (error) {
      console.error('Repaint target-frame upload failed:', error)
    }
  }, [resetPreview, setTargetFrame])

  const updateMapping = useCallback((index: number, patch: Partial<RepaintRegionMapping>) => {
    const next = useStore.getState().editRepaintMappings.map((mapping, mappingIndex) => (
      mappingIndex === index ? { ...mapping, ...patch } : mapping
    ))
    setMappings(next)
    resetPreview()
  }, [resetPreview, setMappings])

  const addMapping = useCallback(() => {
    if (mappings.length >= 5) return
    setMappings([...mappings, newRegion(mappings.length)])
    resetPreview()
  }, [mappings, resetPreview, setMappings])

  const removeMapping = useCallback((index: number) => {
    setMappings(mappings.filter((_, mappingIndex) => mappingIndex !== index))
    resetPreview()
  }, [mappings, resetPreview, setMappings])

  const handlePreview = useCallback(async () => {
    if (
      !editVideoPath
      || !targetFramePath
      || mappings.length === 0
      || mappings.some(mapping => !mapping.source.trim() || !mapping.target.trim())
    ) return
    setPreviewState('loading')
    setPreviewError('')
    try {
      const result = await api.repaintPreview({
        video_path: editVideoPath,
        target_frame_path: targetFramePath,
        region_mappings: mappings.map(mapping => ({
          id: mapping.id,
          source: mapping.source.trim(),
          target: mapping.target.trim(),
        })),
        time: editStartTime,
      })
      setPreview(result)
      setPreviewState(result.found ? 'found' : 'error')
      if (!result.found) {
        setPreviewError('One or more descriptions did not match. Refine the highlighted fields.')
      }
    } catch (error) {
      setPreviewState('error')
      setPreviewError(error instanceof Error ? error.message : 'Region preview failed')
    }
  }, [editStartTime, editVideoPath, mappings, targetFramePath])

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-1.5">
        <Paintbrush size={13} className="shrink-0 text-accent-blue" />
        <span className="text-[10px] font-medium text-text-primary">
          Repaint from first frame
        </span>
        <InfoTooltip
          placement="bottom"
          label="About Repaint"
          text={"Edit the selected first frame, then SCAIL-2 carries that look through the source motion, interaction, and camera movement.\n\nFast is recommended (6 steps). HQ uses the full 40-step schedule."}
        />
      </div>

      <ScailResolutionSelector
        value={resolutionProfile}
        workflow="Repaint"
        onChange={profile => {
          useStore.setState({ editRepaintResolutionProfile: profile })
          resetPreview()
        }}
      />

      {!editVideoFile ? (
        <div
          onDragOver={event => event.preventDefault()}
          onDrop={event => {
            event.preventDefault()
            const file = event.dataTransfer.files[0]
            if (file?.type.startsWith('video/')) void handleVideoUpload(file)
          }}
          onClick={() => videoInputRef.current?.click()}
          className="border-2 border-dashed border-border rounded-lg p-6 text-center cursor-pointer hover:border-accent-blue/50 hover:bg-bg-hover/30 transition-all"
        >
          <Upload size={24} className="mx-auto mb-2 text-text-muted" />
          <p className="text-xs text-text-secondary">Drop the source video or click to upload</p>
          <input
            ref={videoInputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={event => {
              if (event.target.files?.[0]) void handleVideoUpload(event.target.files[0])
              event.currentTarget.value = ''
            }}
          />
        </div>
      ) : (
        <div className="relative">
          <button
            onClick={() => {
              clearEditVideo()
              setTargetFrame(null, '', '')
              resetPreview()
            }}
            className="absolute top-1.5 right-1.5 z-20 p-1 rounded-full bg-black/60 text-white/80 hover:text-white hover:bg-black/80 transition-colors"
            title="Remove source video"
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
              // The target frame must correspond to the selected first frame.
              setTargetFrame(null, '', '')
              resetPreview()
            }}
            onEndChange={time => useStore.setState({ editEndTime: time })}
          />
          <p className="text-[9px] text-text-muted mt-1 truncate">{editVideoFile.name}</p>
        </div>
      )}

      <div className="space-y-2 rounded-lg border border-border bg-bg-secondary/40 p-2.5">
        <div className="flex items-center gap-1">
          <p className="text-[10px] font-medium text-text-primary">Edited first frame</p>
          <InfoTooltip
            label="About the edited first frame"
            text="Change anything you want while keeping the same canvas, viewpoint, and pose. This frame defines the new visual."
          />
        </div>

        {targetFramePath && (
          <div className="relative">
            <img
              src={targetFrameUrl}
              alt="Edited first frame"
              className="w-full max-h-48 object-contain rounded border border-border bg-black/20"
            />
            <button
              onClick={() => {
                setTargetFrame(null, '', '')
                resetPreview()
              }}
              className="absolute top-1.5 right-1.5 p-1 rounded-full bg-black/65 text-white/80 hover:text-white"
              title="Remove edited frame"
            >
              <X size={12} />
            </button>
          </div>
        )}

        <button
          onClick={() => void sendFrameToImageMode('repaint')}
          disabled={!editVideoPath}
          className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded border border-accent-blue/40 bg-accent-blue/10 text-[10px] text-accent-blue hover:bg-accent-blue/20 disabled:opacity-40"
        >
          <Pencil size={11} />
          {targetFramePath ? 'Edit first frame again' : 'Edit first frame'}
        </button>

        <label className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 rounded border border-dashed border-border text-[10px] text-text-secondary hover:text-text-primary hover:border-accent-blue/50 cursor-pointer">
          <Upload size={11} />
          {targetFramePath ? 'Replace edited frame' : 'Upload edited frame'}
          <input
            type="file"
            accept="image/*"
            className="hidden"
            onChange={event => {
              if (event.target.files?.[0]) void handleTargetUpload(event.target.files[0])
              event.currentTarget.value = ''
            }}
          />
        </label>
      </div>

      <div className="rounded-lg border border-border overflow-hidden">
        <div className="flex items-center">
          <button
            onClick={() => setRegionDisclosure(current => ({
              mappingCount: mappings.length,
              open: current.mappingCount === mappings.length ? !current.open : false,
            }))}
            className="flex flex-1 items-center gap-1.5 px-2.5 py-2 text-left hover:bg-bg-hover/40"
          >
            {showRegions ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
            <span className="text-[10px] text-text-primary">Track changed regions</span>
            <span className="ml-auto text-[9px] text-text-muted">Optional</span>
          </button>
          <div className="pr-2">
            <InfoTooltip
              label="About changed-region tracking"
              text="Leave this empty for whole-frame Repaint. Add a source-to-edited-frame mapping only when an object or person needs an explicit color correspondence."
            />
          </div>
        </div>

        {showRegions && (
          <div className="space-y-2 border-t border-border p-2.5">
            {mappings.map((mapping, index) => {
              const result = preview?.mapping_results.find(item => item.mapping_index === index)
              return (
                <div key={mapping.id} className="rounded border border-border bg-bg-secondary/50 p-2 space-y-1.5">
                  <div className="flex items-center gap-1.5">
                    <span
                      className="h-3 w-3 rounded-full border border-white/30"
                      style={{ backgroundColor: REGION_COLORS[index] }}
                    />
                    <span className="text-[9px] text-text-secondary">Region {index + 1}</span>
                    <button
                      onClick={() => removeMapping(index)}
                      className="ml-auto p-0.5 text-text-muted hover:text-status-error"
                      title="Remove region"
                    >
                      <X size={11} />
                    </button>
                  </div>
                  <input
                    type="text"
                    value={mapping.source}
                    onChange={event => updateMapping(index, { source: event.target.value })}
                    placeholder="In source video: electric wrench"
                    className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-[10px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                  />
                  <input
                    type="text"
                    value={mapping.target}
                    onChange={event => updateMapping(index, { target: event.target.value })}
                    placeholder="In edited frame: sci-fi gun"
                    className="w-full bg-bg-tertiary border border-border rounded px-2 py-1.5 text-[10px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                  />
                  {result && (
                    <p className={`text-[8px] ${result.source_found && result.target_found ? 'text-accent-green' : 'text-status-warning'}`}>
                      Source {result.source_found ? 'found' : 'not found'} · edited frame {result.target_found ? 'found' : 'not found'}
                    </p>
                  )}
                </div>
              )
            })}

            <div className="flex gap-1.5">
              {mappings.length < 5 && (
                <button
                  onClick={addMapping}
                  className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded border border-dashed border-border text-[9px] text-text-secondary hover:text-accent-blue hover:border-accent-blue/50"
                >
                  <Plus size={11} />
                  Add region
                </button>
              )}
              {mappings.length > 0 && (
                <button
                  onClick={() => void handlePreview()}
                  disabled={
                    previewState === 'loading'
                    || !editVideoPath
                    || !targetFramePath
                    || mappings.some(mapping => !mapping.source.trim() || !mapping.target.trim())
                  }
                  className="flex-1 flex items-center justify-center gap-1 py-1.5 rounded border border-border bg-bg-tertiary text-[9px] text-text-secondary hover:text-text-primary disabled:opacity-40"
                >
                  {previewState === 'loading'
                    ? <Loader2 size={11} className="animate-spin" />
                    : <Eye size={11} />}
                  Preview regions
                </button>
              )}
            </div>

            {preview && (
              <div className="grid grid-cols-2 gap-1.5">
                <div>
                  <img src={preview.source_preview} alt="Source region preview" className="w-full rounded border border-border" />
                  <p className="text-[8px] text-text-muted mt-0.5">source video</p>
                </div>
                <div>
                  <img src={preview.target_preview} alt="Edited-frame region preview" className="w-full rounded border border-border" />
                  <p className="text-[8px] text-text-muted mt-0.5">edited frame</p>
                </div>
              </div>
            )}
            {previewState === 'error' && previewError && (
              <p className="text-[9px] text-status-error">{previewError}</p>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
