import { useState, useMemo, useEffect } from 'react'
import { X, Upload, Plus, Music, Film, Mic } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'

// Unified, media-driven "Inputs" panel for Studio Frames mode (image_mode 0).
//
// Goal: a single image-forward tile surface where the media you add auto-selects
// the pipeline (image -> frame, audio -> soundtrack/voice, video -> control),
// replacing the old scattered sections + the audio mode dropdown.
//
// Shipped so far:
//   slice 1  — start / end frame tiles (replacing ImageUpload in Frames mode)
//   slice 1b — inject reference frames as tiles, out of the Advanced menu.
//   slice 2  — audio (soundtrack) + control-video tiles. Dropping audio sets the
//              model's soundtrack audio_prompt_type + audio_guide; adding a
//              control video sets "K" + video_guide. AudioModeSection's dropdown
//              is hidden in Frames mode (Sidebar) — the tiles route it instead.
// Next: voice-ref, references, then one unified drop zone + role auto-detect.
//
// Existing params stay the source of truth, so Load Settings restore keeps
// working (tiles are derived from params).

interface InjectedFrame {
  path: string
  filename: string
  position: string
  previewUrl: string
  window: number
  offset: string
  // The original File, kept for freshly-added frames so re-roling an inject
  // back to a native start/end frame can reuse it (restored frames have null
  // and fall back to their uploaded path).
  file: File | null
}

const OFFSET_PRESETS = [
  { value: 'start', label: 'Start', pct: 0 },
  { value: '25%', label: '25%', pct: 0.25 },
  { value: 'middle', label: 'Mid', pct: 0.5 },
  { value: '75%', label: '75%', pct: 0.75 },
  { value: 'end', label: 'End', pct: 1.0 },
] as const

const WINDOW_TOKEN_RE = /^[Ww](\d+):(\d{1,3})$/

const calcPositionToken = (windowIdx: number, offset: string): string => {
  const preset = OFFSET_PRESETS.find(p => p.value === offset)
  const pct = Math.round((preset?.pct ?? 1.0) * 100)
  return `W${windowIdx + 1}:${pct}`
}

const snapToOffsetPreset = (pct: number): string => {
  let closest: (typeof OFFSET_PRESETS)[number] = OFFSET_PRESETS[0]
  let minDist = Math.abs(pct - closest.pct)
  for (const p of OFFSET_PRESETS) {
    const dist = Math.abs(pct - p.pct)
    if (dist < minDist) { closest = p; minDist = dist }
  }
  return closest.value
}

const basename = (p: string) => p.replace(/\\/g, '/').split('/').pop() || p

const getMediaDuration = (file: File): Promise<number | null> => {
  const isVid = file.type.startsWith('video/') || /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)
  return new Promise(resolve => {
    const url = URL.createObjectURL(file)
    const el: HTMLMediaElement = isVid ? document.createElement('video') : new Audio()
    el.addEventListener('loadedmetadata', () => {
      const d = el.duration; URL.revokeObjectURL(url); resolve(Number.isFinite(d) ? d : null)
    })
    el.addEventListener('error', () => { URL.revokeObjectURL(url); resolve(null) })
    el.src = url
  })
}

export function InputsPanel() {
  const modelOptions = useStore(s => s.modelOptions)
  const startImage = useStore(s => s.startImage)
  const endImage = useStore(s => s.endImage)
  const setStartImage = useStore(s => s.setStartImage)
  const setEndImage = useStore(s => s.setEndImage)
  const supportsEndFrame = useStore(s => s.modelOptions?.supports_end_frame ?? false)
  const strengthLabel = useStore(s => s.modelOptions?.input_video_strength_label ?? '')
  const inputVideoStrength = useStore(s => s.params.input_video_strength ?? 1.0)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const durationSeconds = useStore(s => s.durationSeconds)
  const slidingWindowSeconds = useStore(s => s.slidingWindowSeconds)
  const slidingWindowOverlap = useStore(s => s.slidingWindowOverlap)
  const audioGuideFilename = useStore(s => s.audioGuideFilename)
  const setAudioGuideFilename = useStore(s => s.setAudioGuideFilename)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)
  const setGuideVideoFps = useStore(s => s.setGuideVideoFps)
  const voiceRefEnabled = useStore(s => !!s.servicesConfig?.voice_reference_enabled)
  const directorVoiceRef = useStore(s => s.directorVoiceRef)
  const setDirectorVoiceRef = useStore(s => s.setDirectorVoiceRef)
  const identityScale = useStore(s => s.directorIdentityGuidanceScale)
  const setIdentityScale = useStore(s => s.setDirectorIdentityGuidanceScale)
  const imageRefs = useStore(s => s.imageRefs)
  const addImageRef = useStore(s => s.addImageRef)
  const removeImageRef = useStore(s => s.removeImageRef)
  const reorderImageRefs = useStore(s => s.reorderImageRefs)
  const imageRefType = useStore(s => s.imageRefType)
  const setImageRefType = useStore(s => s.setImageRefType)
  const removeBackgroundRefs = useStore(s => s.removeBackgroundRefs)
  const setRemoveBackgroundRefs = useStore(s => s.setRemoveBackgroundRefs)
  const continueVideo = useStore(s => s.continueVideo)
  const continueVideoUrl = useStore(s => s.continueVideoUrl)
  const continueVideoDuration = useStore(s => s.continueVideoDuration)
  const setContinueVideo = useStore(s => s.setContinueVideo)
  const clearContinueVideo = useStore(s => s.clearContinueVideo)
  const isExtend = (params.image_mode as number) === 3

  const [selected, setSelected] = useState<string | null>(null)
  const [injectedFrames, setInjectedFrames] = useState<InjectedFrame[]>([])
  const [frameUploading, setFrameUploading] = useState(false)
  const [videoGuideFilename, setVideoGuideFilename] = useState<string | null>(null)
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null)
  const [frameDragKey, setFrameDragKey] = useState<string | null>(null)
  const [frameDragOverKey, setFrameDragOverKey] = useState<string | null>(null)

  // ── Inject capability + window layout ──────────────────────────────
  const supportsInject = useMemo(() => {
    const cfg = (modelOptions?.guide_preprocessing || modelOptions?.guide_custom_choices) as
      { choices?: [string, string][]; selection?: string[] } | undefined
    if (!cfg) return false
    const values = cfg.choices ? cfg.choices.map(([, v]) => v) : (cfg.selection || [])
    return values.some(v => typeof v === 'string' && v.includes('KFI'))
  }, [modelOptions])

  const windowInfo = useMemo(() => {
    const fps = modelOptions?.fps ?? 25
    const swDefaults = modelOptions?.sliding_window_defaults
    const discardFrames = swDefaults?.discard_last_frames ?? 0
    const overlapFrames = slidingWindowOverlap ?? swDefaults?.overlap_default ?? 0
    const discardSec = discardFrames / fps
    const overlapSec = overlapFrames / fps
    const stride = slidingWindowSeconds - discardSec - overlapSec
    const windowCount = stride > 0 && durationSeconds > slidingWindowSeconds
      ? Math.max(1, 1 + Math.ceil((durationSeconds - slidingWindowSeconds + discardSec) / stride))
      : 1
    return { fps, windowCount }
  }, [modelOptions, durationSeconds, slidingWindowSeconds, slidingWindowOverlap])

  // ── Audio / control-video capability + current state ───────────────
  const audioCfg = modelOptions?.audio_prompt_type_sources as
    { choices?: [string, string][]; selection?: string[]; default?: string } | undefined
  const audioVals = audioCfg ? (audioCfg.choices ? audioCfg.choices.map(([, v]) => v) : (audioCfg.selection || [])) : []
  const audioOnly = !!modelOptions?.audio_only
  const soundtrackVal = audioVals.find(v => typeof v === 'string' && v.includes('A'))
  const supportsSoundtrack = !!soundtrackVal && !audioOnly
  const supportsControlVid = audioVals.includes('K')
  const audioPT = (params.audio_prompt_type as string) || ''
  const audioBase = audioPT.replace(/[NV]/g, '')
  const audioFlags = audioPT.replace(/[^NV]/g, '')
  const hasSoundtrack = !audioOnly && audioBase.includes('A')
  const hasControlVid = audioBase === 'K'
  const soundtrackName = audioGuideFilename || (params.audio_guide ? basename(params.audio_guide as string) : null)
  const controlVidName = videoGuideFilename || (params.video_guide ? basename(params.video_guide as string) : null)

  // ── Guide video (motion source) for guide_custom_choices models ────
  // Models like SCAIL-2 take a Control Video as the motion/scene guide
  // (video_prompt_type contains 'V') with no audio coupling. Models with
  // guide_preprocessing keep their upload in Advanced Settings, and
  // K-audio models keep the soundtrack-coupled tile above — this tile
  // only fills the gap between them.
  const guideCfg = modelOptions?.guide_custom_choices as { choices?: [string, string][]; default?: string } | undefined
  const guideDefault = guideCfg?.default || ''
  const guideProcess = ((params.video_prompt_type as string) || guideDefault).replace(/T$/, '')
  const supportsGuideVid = !!guideCfg && !modelOptions?.guide_preprocessing && !supportsControlVid && guideProcess.includes('V')
  const hasGuideVid = supportsGuideVid && !!params.video_guide

  // ── Reference images (image_ref_choices) ───────────────────────────
  const refCfg = modelOptions?.image_ref_choices as { choices?: [string, string][] } | undefined
  const supportsRefs = !!refCfg
  const hasLandscapeMode = refCfg?.choices?.some(([, v]) => v.includes('K')) ?? false
  const hasPeopleMode = refCfg?.choices?.some(([, v]) => v === 'I') ?? false
  const refBgLabel = modelOptions?.background_removal_label
  const defaultRefType = hasLandscapeMode ? 'KI' : hasPeopleMode ? 'I' : ''

  // Auto-set the ref type when references are added/removed (mirrors ImageRefSection).
  useEffect(() => {
    if (imageRefs.length > 0 && imageRefType === '') setImageRefType(defaultRefType)
    else if (imageRefs.length === 0 && imageRefType !== '') setImageRefType('')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageRefs.length])

  const pickReferences = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.png,.jpg,.jpeg,.webp,.bmp'
    input.multiple = true
    input.onchange = () => Array.from(input.files || []).forEach(addImageRef)
    input.click()
  }

  // Extend mode: the source video to continue from.
  const handleAddExtendSource = async (file: File) => {
    if (!file.type.startsWith('video/')) return
    try {
      const result = await api.uploadImage(file)
      const url = URL.createObjectURL(file)
      const video = document.createElement('video')
      video.src = url
      video.onloadedmetadata = () => {
        setContinueVideo(file, result.path, url, (video.duration && isFinite(video.duration)) ? video.duration : 0)
      }
    } catch (e) {
      console.error('Extend source upload failed:', e)
    }
  }

  // Restore inject tiles from params (Load Settings / KFI toggled in Advanced).
  useEffect(() => {
    const vpt = params.video_prompt_type || ''
    const refs = params.image_refs as string[] | undefined
    const positions = ((params.frames_positions as string) || '').split(' ').filter(Boolean)
    if (vpt.includes('KFI') && refs && refs.length > 0) {
      const restored: InjectedFrame[] = refs.map((refPath, i) => {
        const filename = basename(refPath)
        const pos = positions[i] || 'L'
        let win = 0, offset = 'end'
        const m = WINDOW_TOKEN_RE.exec(pos)
        if (m) {
          win = Math.max(0, parseInt(m[1], 10) - 1)
          offset = snapToOffsetPreset(Math.min(100, parseInt(m[2], 10)) / 100)
        }
        return { path: refPath, filename, position: pos, previewUrl: `/api/v1/uploads/${filename}`, window: win, offset, file: null }
      })
      const same = restored.length === injectedFrames.length &&
        restored.every((r, i) => r.path === injectedFrames[i]?.path && r.position === injectedFrames[i]?.position)
      if (!same) setInjectedFrames(restored)
    } else if (!vpt.includes('KFI') && injectedFrames.length > 0) {
      setInjectedFrames([])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.video_prompt_type, params.image_refs, params.frames_positions])

  const pickFile = (accept: string, onFile: (f: File) => void) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = accept
    input.onchange = e => {
      const f = (e.target as HTMLInputElement).files?.[0]
      if (f) onFile(f)
    }
    input.click()
  }
  const pickImage = (onFile: (f: File) => void) => pickFile('image/*', onFile)

  // ── Inject handlers ────────────────────────────────────────────────
  const syncFrameParams = (frames: InjectedFrame[]) => {
    if (frames.length === 0) {
      setParam('image_refs', undefined)
      setParam('frames_positions', undefined)
      const vpt = (params.video_prompt_type as string) || ''
      if (vpt.includes('KFI')) setParam('video_prompt_type', vpt.replace('KFI', ''))
      return
    }
    setParam('image_refs', frames.map(f => f.path) as unknown as never)
    setParam('frames_positions', frames.map(f => f.position).join(' ') as unknown as never)
    const vpt = (params.video_prompt_type as string) || ''
    if (!vpt.includes('KFI')) setParam('video_prompt_type', 'KFI')
  }

  const addInjectFrame = async (file: File | null, path: string | null, previewUrl: string | null, offset: string, windowIdx = 0) => {
    setFrameUploading(true)
    try {
      let p = path
      if (!p && file) p = (await api.uploadImage(file)).path
      if (!p) return
      const newFrame: InjectedFrame = {
        path: p, filename: file?.name || basename(p), file: file ?? null,
        position: calcPositionToken(windowIdx, offset),
        previewUrl: previewUrl || `/api/v1/uploads/${basename(p)}`,
        window: windowIdx, offset,
      }
      const updated = [...injectedFrames, newFrame]
      setInjectedFrames(updated)
      syncFrameParams(updated)
    } catch (e) {
      console.error('Frame upload failed:', e)
    } finally {
      setFrameUploading(false)
    }
  }
  const handleRemoveFrame = (index: number) => {
    const updated = injectedFrames.filter((_, i) => i !== index)
    setInjectedFrames(updated)
    syncFrameParams(updated)
  }

  // ── Unified, window-aware "Frame" model (one concept, pipeline invisible) ──
  // Every frame is just a (window, offset). Its designation picks the pipe:
  //   W1 Start  -> native i2v start frame (image_start)
  //   W1 End    -> native end frame (image_end)
  //   everything else (W1 25/Mid/75, and ALL of W2+) -> injected keyframe
  // All tiles share the same controls (window + offset) — start/end aren't
  // special-cased. Extend mode: the source video is the anchor, so all inject.
  const offsetLabel = (offset: string) => OFFSET_PRESETS.find(p => p.value === offset)?.label ?? offset
  const offsetPct = (offset: string) => OFFSET_PRESETS.find(p => p.value === offset)?.pct ?? 1
  const lastWindow = Math.max(0, windowInfo.windowCount - 1)
  const hasStart = !!startImage || (!isExtend && !!params.image_start)
  const hasEnd = !!endImage || (!isExtend && !!params.image_end)

  const frameRoleFor = (window: number, offset: string): 'start' | 'end' | 'inject' => {
    if (isExtend) return 'inject'
    if (window <= 0 && offset === 'start') return 'start'
    if (supportsEndFrame && window <= 0 && offset === 'end') return 'end'
    return 'inject'
  }
  // Position along the whole timeline (window + within-window fraction) — used
  // for sorting the row and for drag-to-reposition interpolation.
  const frameKey = (window: number, offset: string) => window + offsetPct(offset)

  type FrameTile = { key: string; kind: 'start' | 'end' | 'inject'; injectIndex?: number; preview: string; offset: string; window: number; sortKey: number }

  // Render-only list, SORTED by timeline position so the row always reads left
  // (start) to right (end) and a frame repositions itself when you change it.
  const frameTiles = useMemo<FrameTile[]>(() => {
    const out: FrameTile[] = []
    if (!isExtend) {
      const startPreview = startImage ? URL.createObjectURL(startImage)
        : (params.image_start ? `/api/v1/uploads/${basename(params.image_start as string)}` : null)
      if (startPreview) out.push({ key: 'frame-start', kind: 'start', preview: startPreview, offset: 'start', window: 0, sortKey: 0 })
    }
    injectedFrames.forEach((f, i) => out.push({ key: `frame-inj-${i}`, kind: 'inject', injectIndex: i, preview: f.previewUrl, offset: f.offset, window: f.window, sortKey: frameKey(f.window, f.offset) }))
    if (!isExtend) {
      const endPreview = endImage ? URL.createObjectURL(endImage)
        : (params.image_end ? `/api/v1/uploads/${basename(params.image_end as string)}` : null)
      if (endPreview) out.push({ key: 'frame-end', kind: 'end', preview: endPreview, offset: 'end', window: 0, sortKey: frameKey(0, 'end') })
    }
    out.sort((a, b) => a.sortKey - b.sortKey)
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [startImage, endImage, injectedFrames, params.image_start, params.image_end, isExtend, supportsEndFrame, lastWindow])

  const canAddFrame = isExtend ? supportsInject : (!hasStart || (supportsEndFrame && !hasEnd) || supportsInject)

  // "+ Frame": smart default — 1st image = start, 2nd = end (where supported),
  // the rest injected keyframes that walk forward through the windows: in a
  // multi-window clip the next inject lands at the END of the next window
  // (3rd frame -> window 2 End), never on the native last-window end. Single
  // window -> Mid.
  const handleAddFrameSmart = async (file: File) => {
    if (!isExtend && !hasStart) { setStartImage(file); return }
    if (!isExtend && supportsEndFrame && !hasEnd) { setEndImage(file); return }
    let w = 0, off = 'middle'
    if (lastWindow >= 1) {
      w = Math.min(injectedFrames.length + 1, lastWindow)
      off = 'end'
    }
    await addInjectFrame(file, null, URL.createObjectURL(file), off, w)
  }

  // Set a frame's (window, offset), re-routing it across pipelines as needed.
  // A move onto an already-taken native start/end slot falls back to an inject
  // just inside that boundary instead of stealing it.
  const setFramePosition = async (tile: FrameTile, newWindow: number, newOffset: string) => {
    let role = frameRoleFor(newWindow, newOffset)
    if (role === 'start' && hasStart && tile.kind !== 'start') { newOffset = '25%'; role = 'inject' }
    if (role === 'end' && hasEnd && tile.kind !== 'end') { newOffset = '75%'; role = 'inject' }

    if (tile.kind === role) {
      if (role === 'inject' && tile.injectIndex !== undefined) {
        const updated = injectedFrames.map((f, i) =>
          i === tile.injectIndex ? { ...f, window: newWindow, offset: newOffset, position: calcPositionToken(newWindow, newOffset) } : f)
        setInjectedFrames(updated); syncFrameParams(updated)
      }
      return
    }
    // Capture the image, then move it from its old home to its new one.
    let file: File | null = null, path: string | null = null, previewUrl: string | null = tile.preview
    if (tile.kind === 'start') { file = startImage; path = (params.image_start as string) || null }
    else if (tile.kind === 'end') { file = endImage; path = (params.image_end as string) || null }
    else if (tile.injectIndex !== undefined) {
      const f = injectedFrames[tile.injectIndex]; file = f?.file ?? null; path = f?.path ?? null; previewUrl = f?.previewUrl ?? previewUrl
    }
    if (tile.kind === 'start') setStartImage(null)
    else if (tile.kind === 'end') setEndImage(null)
    else if (tile.injectIndex !== undefined) {
      const updated = injectedFrames.filter((_, i) => i !== tile.injectIndex)
      setInjectedFrames(updated); syncFrameParams(updated)
    }
    if (role === 'start') {
      if (file) setStartImage(file); else if (path) { setStartImage(null); setParam('image_start', path) }
    } else if (role === 'end') {
      if (file) setEndImage(file); else if (path) { setEndImage(null); setParam('image_end', path) }
    } else {
      await addInjectFrame(file, path, previewUrl, newOffset, newWindow)
    }
    setSelected(null)
  }

  // Drag-to-reposition: dropping frame A onto tile B gives A a timeline slot
  // just before B (midway between B and its left neighbour), then snaps to the
  // nearest window/offset. Native start/end are guarded by setFramePosition.
  const repositionFrameBefore = async (draggedKey: string, target: FrameTile) => {
    if (draggedKey === target.key) return
    const dragged = frameTiles.find(t => t.key === draggedKey)
    if (!dragged) return
    const rest = frameTiles.filter(t => t.key !== draggedKey)
    const ti = rest.findIndex(t => t.key === target.key)
    const leftKey = ti > 0 ? rest[ti - 1].sortKey : target.sortKey - 1
    const targetKey = (leftKey + target.sortKey) / 2
    const w = Math.min(lastWindow, Math.max(0, Math.floor(targetKey)))
    const pct = Math.min(1, Math.max(0, targetKey - w))
    await setFramePosition(dragged, w, snapToOffsetPreset(pct))
  }

  const removeFrameTile = (tile: FrameTile) => {
    if (tile.kind === 'start') setStartImage(null)
    else if (tile.kind === 'end') setEndImage(null)
    else if (tile.injectIndex !== undefined) handleRemoveFrame(tile.injectIndex)
    if (selected === tile.key) setSelected(null)
  }

  // Strip offset buttons: which to show, and which are a TAKEN native slot.
  const framePresetVisible = (offset: string): boolean => {
    if (isExtend) return supportsInject
    if (offset === 'start') return true
    if (offset === 'end') return supportsEndFrame || supportsInject
    return supportsInject
  }
  const framePresetDisabled = (tile: FrameTile, offset: string): boolean => {
    if (isExtend) return false
    const role = frameRoleFor(tile.window, offset)
    if (role === 'start') return hasStart && tile.kind !== 'start'
    if (role === 'end') return hasEnd && tile.kind !== 'end'
    return false
  }
  const frameRoutingHint = (tile: FrameTile): string => {
    if (isExtend) return 'Injected as a keyframe into the new content.'
    const role = frameRoleFor(tile.window, tile.offset)
    if (role === 'start') return 'Used as the first frame (image-to-video start).'
    if (role === 'end') return 'Used as the final frame.'
    return 'Injected as a keyframe at this point in the timeline.'
  }

  // ── Audio / control-video handlers ─────────────────────────────────
  const handleAddSoundtrack = async (file: File) => {
    try {
      const result = await api.uploadAudio(file)
      setParam('audio_guide', result.path)
      setAudioGuideFilename(file.name)
      setParam('audio_prompt_type', (soundtrackVal || 'A') + audioFlags)
      const dur = await getMediaDuration(file)
      if (dur && dur > 0) setDurationSeconds(Math.round(dur * 10) / 10)
    } catch (e) {
      console.error('Soundtrack upload failed:', e)
    }
  }
  const removeSoundtrack = () => {
    setParam('audio_guide', undefined)
    setAudioGuideFilename(null)
    setParam('audio_prompt_type', audioFlags)  // back to text mode, keep N/V flags
    if (selected === 'audio') setSelected(null)
  }
  const handleAddControlVid = async (file: File) => {
    try {
      const result = await api.uploadImage(file)  // full video kept (generic upload)
      setParam('video_guide', result.path)
      setVideoGuideFilename(file.name)
      setParam('audio_prompt_type', 'K')
    } catch (e) {
      console.error('Control video upload failed:', e)
    }
  }
  const removeControlVid = () => {
    setParam('video_guide', undefined)
    setVideoGuideFilename(null)
    setParam('audio_prompt_type', '')
    if (selected === 'ctrlvid') setSelected(null)
  }
  const handleAddGuideVid = async (file: File) => {
    try {
      const result = await api.uploadImage(file)
      setParam('video_guide', result.path)
      setVideoGuideFilename(file.name)
      // Lock in the guide process letters: defaults are not hydrated into
      // params client-side, so without this a user who never opens
      // Advanced Settings would submit video_prompt_type '' and the model
      // would not receive the control video at all.
      if (!params.video_prompt_type && guideDefault) setParam('video_prompt_type', guideDefault)
      // Real fps of the guide, probed server-side — startGeneration uses
      // it for the seconds→frames conversion on force_fps="control" models.
      setGuideVideoFps(result.fps && result.fps > 0 ? result.fps : null)
      const dur = await getMediaDuration(file)
      if (dur && dur > 0) setDurationSeconds(Math.round(dur * 10) / 10)
    } catch (e) {
      console.error('Guide video upload failed:', e)
    }
  }
  const removeGuideVid = () => {
    setParam('video_guide', undefined)
    setVideoGuideFilename(null)
    setGuideVideoFps(null)
    if (selected === 'guidevid') setSelected(null)
  }
  const toggleAudioFlag = (flag: 'N' | 'V') => {
    const cur = (params.audio_prompt_type as string) || ''
    setParam('audio_prompt_type', cur.includes(flag) ? cur.replace(flag, '') : cur + flag)
  }

  const selectedFrameTile = frameTiles.find(t => t.key === selected) || null

  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Inputs</label>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {/* Extend-from source video (Extend mode only) — the timeline anchor. */}
        {isExtend && (continueVideo ? (
          <div onClick={() => setSelected(selected === 'extend' ? null : 'extend')}
            className={`relative w-[90px] h-[90px] shrink-0 rounded-xl overflow-hidden border cursor-pointer transition-colors ${selected === 'extend' ? 'border-accent-blue' : 'border-border hover:border-border-light'}`}>
            {continueVideoUrl && <video src={continueVideoUrl} muted className="absolute inset-0 w-full h-full object-cover" />}
            <button onClick={e => { e.stopPropagation(); clearContinueVideo(); if (selected === 'extend') setSelected(null) }}
              className="absolute top-1 right-1 z-10 rounded-full bg-black/45 text-white p-0.5 hover:bg-black/70" aria-label="Remove"><X size={12} /></button>
            <div className="absolute inset-x-0 bottom-0 bg-black/55 px-1.5 py-1">
              <span className="text-[10px] text-white/95">Extend from{continueVideoDuration > 0 ? ` · ${continueVideoDuration.toFixed(1)}s` : ''}</span>
            </div>
          </div>
        ) : (
          <AddTile label="Extend from" icon={<Film size={18} />} onClick={() => pickFile('video/*', handleAddExtendSource)} onDropFile={handleAddExtendSource} dropAccept="video" />
        ))}

        {/* Unified "Frame" tiles — start / end / injected keyframes, one concept,
            sorted by timeline position and draggable to reposition. The per-tile
            position strip below routes each to its pipeline. */}
        {frameTiles.map(tile => (
          <div key={tile.key} draggable
            onDragStart={e => { setFrameDragKey(tile.key); e.dataTransfer.setData('frame-key', tile.key); e.dataTransfer.effectAllowed = 'move' }}
            onDragEnd={() => { setFrameDragKey(null); setFrameDragOverKey(null) }}
            onDragOver={e => { if (frameDragKey && frameDragKey !== tile.key) { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setFrameDragOverKey(tile.key) } }}
            onDragLeave={() => setFrameDragOverKey(prev => (prev === tile.key ? null : prev))}
            onDrop={e => {
              const dk = e.dataTransfer.getData('frame-key')
              if (!dk) return
              e.preventDefault(); e.stopPropagation(); setFrameDragOverKey(null); setFrameDragKey(null)
              if (dk !== tile.key) void repositionFrameBefore(dk, tile)
            }}
            onClick={() => setSelected(selected === tile.key ? null : tile.key)}
            className={`relative w-[90px] h-[90px] shrink-0 rounded-xl overflow-hidden border cursor-grab active:cursor-grabbing transition-colors ${
              frameDragOverKey === tile.key ? 'border-accent-blue border-2'
                : selected === tile.key ? 'border-accent-blue' : 'border-border hover:border-border-light'
            }`}>
            <img src={tile.preview} alt={`Frame ${offsetLabel(tile.offset)}`} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
            <button onClick={e => { e.stopPropagation(); removeFrameTile(tile) }}
              className="absolute top-1 right-1 z-10 rounded-full bg-black/45 text-white p-0.5 hover:bg-black/70 transition-colors" aria-label="Remove"><X size={12} /></button>
            <div className="absolute inset-x-0 bottom-0 bg-black/55 px-1.5 py-1">
              <span className="text-[10px] text-white/95">{windowInfo.windowCount > 1 ? `W${tile.window + 1} · ` : ''}{offsetLabel(tile.offset)}</span>
            </div>
          </div>
        ))}
        {canAddFrame && (
          <AddTile label={frameUploading ? 'Uploading…' : 'Frame'} icon={<Plus size={18} />}
            onClick={() => pickImage(handleAddFrameSmart)} onDropFile={handleAddFrameSmart} dropAccept="image" />
        )}

        {/* Soundtrack (audio) */}
        {hasSoundtrack ? (
          <Tile role="Soundtrack" filledIcon={<Music size={20} />} filledLabel={soundtrackName ?? undefined}
            imgSrc={null} selected={selected === 'audio'} onClear={removeSoundtrack}
            onSelect={() => setSelected(selected === 'audio' ? null : 'audio')} />
        ) : supportsSoundtrack && !hasControlVid && (
          <AddTile label="Soundtrack" icon={<Music size={18} />} onClick={() => pickFile('.wav,.mp3,.flac,.ogg,.m4a,.mp4,.mov,.mkv,.webm', handleAddSoundtrack)} onDropFile={handleAddSoundtrack} />
        )}

        {/* Control video */}
        {hasControlVid ? (
          <Tile role="Control video" filledIcon={<Film size={20} />} filledLabel={controlVidName ?? undefined}
            imgSrc={null} selected={selected === 'ctrlvid'} onClear={removeControlVid}
            onSelect={() => setSelected(selected === 'ctrlvid' ? null : 'ctrlvid')} />
        ) : supportsControlVid && !hasSoundtrack && (
          <AddTile label="Control video" icon={<Film size={18} />} onClick={() => pickFile('.mp4,.webm,.mkv,.mov', handleAddControlVid)} onDropFile={handleAddControlVid} dropAccept="video" />
        )}

        {/* Guide video (motion source) — guide_custom_choices models (SCAIL-2 etc.) */}
        {hasGuideVid ? (
          <Tile role="Control video" filledIcon={<Film size={20} />} filledLabel={controlVidName ?? undefined}
            imgSrc={null} selected={selected === 'guidevid'} onClear={removeGuideVid}
            onSelect={() => setSelected(selected === 'guidevid' ? null : 'guidevid')} />
        ) : supportsGuideVid && (
          <AddTile label="Control video" icon={<Film size={18} />} onClick={() => pickFile('.mp4,.webm,.mkv,.mov', handleAddGuideVid)} onDropFile={handleAddGuideVid} dropAccept="video" />
        )}

        {/* Voice reference (ID-LoRA) — keeps the speaker's voice consistent. */}
        {voiceRefEnabled && (directorVoiceRef ? (
          <Tile role="Voice ref" filledIcon={<Mic size={20} />} filledLabel={directorVoiceRef.name}
            imgSrc={null} selected={selected === 'voiceref'}
            onClear={() => { setDirectorVoiceRef(null); if (selected === 'voiceref') setSelected(null) }}
            onSelect={() => setSelected(selected === 'voiceref' ? null : 'voiceref')} />
        ) : (
          <AddTile label="Voice ref" icon={<Mic size={18} />} onClick={() => pickFile('.wav,.mp3,.flac,.ogg,.m4a', setDirectorVoiceRef)} onDropFile={setDirectorVoiceRef} dropAccept="audio" />
        ))}

        {/* Reference images (ordered; first = main subject/landscape). Drag to reorder. */}
        {supportsRefs && imageRefs.map((file, i) => (
          <div key={`ref-${i}-${file.name}`} draggable
            onDragStart={e => { e.dataTransfer.setData('ref-index', String(i)); e.dataTransfer.effectAllowed = 'move' }}
            onDragOver={e => { e.preventDefault(); e.dataTransfer.dropEffect = 'move'; setDragOverIndex(i) }}
            onDragLeave={() => setDragOverIndex(null)}
            onDrop={e => {
              e.preventDefault(); e.stopPropagation(); setDragOverIndex(null)
              const from = parseInt(e.dataTransfer.getData('ref-index'), 10)
              if (!isNaN(from) && from !== i) reorderImageRefs(from, i)
            }}
            onClick={() => setSelected(selected === `ref-${i}` ? null : `ref-${i}`)}
            className={`relative w-[90px] h-[90px] shrink-0 rounded-xl overflow-hidden border cursor-grab active:cursor-grabbing transition-colors ${
              dragOverIndex === i ? 'border-accent-blue border-2' : selected === `ref-${i}` ? 'border-accent-blue' : 'border-border hover:border-border-light'
            }`}>
            <img src={URL.createObjectURL(file)} alt={`Ref ${i + 1}`} className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
            <span className="absolute top-1 left-1 z-10 rounded bg-black/55 text-white text-[9px] px-1">{i + 1}</span>
            <button onClick={e => { e.stopPropagation(); removeImageRef(i); if (selected === `ref-${i}`) setSelected(null) }}
              className="absolute top-1 right-1 z-10 rounded-full bg-black/45 text-white p-0.5 hover:bg-black/70" aria-label="Remove"><X size={12} /></button>
            <div className="absolute inset-x-0 bottom-0 bg-black/55 px-1.5 py-1">
              <span className="text-[10px] text-white/95">{i === 0 && hasLandscapeMode && imageRefType === 'KI' ? 'Main ref' : 'Reference'}</span>
            </div>
          </div>
        ))}
        {supportsRefs && <AddTile label="Reference" icon={<Plus size={18} />} onClick={pickReferences} onDropFile={addImageRef} dropAccept="image" />}
      </div>

      {/* Option strip — Frame: position picker (routes start / end / inject
          invisibly) + role-specific strength. */}
      {selectedFrameTile && (
        <Strip>
          <div className="flex items-center gap-1.5">
            {windowInfo.windowCount > 1 && (
              <>
                <span className="text-[10px] text-text-muted shrink-0">Window</span>
                <select value={selectedFrameTile.window}
                  onChange={e => setFramePosition(selectedFrameTile, parseInt(e.target.value), selectedFrameTile.offset)}
                  className="shrink-0 bg-bg-secondary border border-border rounded px-1 py-0.5 text-[11px] text-text-primary focus:outline-none focus:border-accent-blue">
                  {Array.from({ length: windowInfo.windowCount }, (_, wi) => <option key={wi} value={wi}>{wi + 1}</option>)}
                </select>
                <span className="text-[10px] text-text-muted shrink-0">at</span>
              </>
            )}
            <div className="flex gap-0.5 flex-1">
              {OFFSET_PRESETS.filter(p => framePresetVisible(p.value)).map(preset => {
                const disabled = framePresetDisabled(selectedFrameTile, preset.value)
                const active = selectedFrameTile.offset === preset.value
                return (
                  <button key={preset.value} disabled={disabled}
                    onClick={() => setFramePosition(selectedFrameTile, selectedFrameTile.window, preset.value)}
                    className={`flex-1 text-[10px] py-0.5 rounded transition-colors ${
                      active ? 'bg-accent-blue text-white'
                        : disabled ? 'bg-bg-secondary text-text-muted cursor-not-allowed'
                        : 'bg-bg-secondary text-text-muted hover:text-text-primary hover:bg-bg-hover'
                    }`}>{preset.label}</button>
                )
              })}
            </div>
          </div>
          {selectedFrameTile.kind === 'inject' ? (
            <>
              <Row label="Injection strength" value={(params.injection_strength ?? 1.0).toFixed(2)} />
              <input type="range" min={0} max={1} step={0.05} value={params.injection_strength ?? 1.0}
                onChange={e => setParam('injection_strength', parseFloat(e.target.value))} className="w-full h-1 accent-accent-blue" />
            </>
          ) : strengthLabel ? (
            <>
              <Row label={strengthLabel} value={inputVideoStrength.toFixed(2)} />
              <input type="range" min={0} max={1} step={0.01} value={inputVideoStrength}
                onChange={e => setParam('input_video_strength', parseFloat(e.target.value))} className="w-full h-1 accent-accent-blue" />
            </>
          ) : null}
          <p className="text-[9px] text-text-muted">{frameRoutingHint(selectedFrameTile)}</p>
        </Strip>
      )}

      {/* Option strip — extend source: source video strength */}
      {selected === 'extend' && continueVideo && (
        <Strip>
          <Row label="Source video strength" value={inputVideoStrength.toFixed(2)} />
          <input type="range" min={0} max={1} step={0.05} value={inputVideoStrength}
            onChange={e => setParam('input_video_strength', parseFloat(e.target.value))} className="w-full h-1 accent-accent-blue" />
          <p className="text-[9px] text-text-muted">1.0 = seamless continuation; lower gives more creative freedom. New content is appended after the source.</p>
        </Strip>
      )}

      {/* Option strip — soundtrack: audio strength + processing flags */}
      {selected === 'audio' && hasSoundtrack && (
        <Strip>
          <Row label={modelOptions?.audio_scale_name || 'Audio strength'} value={(((params as unknown as Record<string, unknown>).modality_scale as number) ?? 1.0).toFixed(1)} />
          <input type="range" min={0.1} max={3.0} step={0.1} value={((params as unknown as Record<string, unknown>).modality_scale as number) ?? 1.0}
            onChange={e => setParam('modality_scale' as keyof typeof params, parseFloat(e.target.value) as never)} className="w-full h-1 accent-accent-blue" />
          <label className="flex items-center gap-2 cursor-pointer pt-1">
            <input type="checkbox" checked={audioPT.includes('N')} onChange={() => toggleAudioFlag('N')} className="accent-accent-blue" />
            <span className="text-[10px] text-text-secondary">Normalize audio volume</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={audioPT.includes('V')} onChange={() => toggleAudioFlag('V')} className="accent-accent-blue" />
            <span className="text-[10px] text-text-secondary">Remove background music</span>
          </label>
        </Strip>
      )}

      {/* Option strip — voice reference: identity guidance scale */}
      {selected === 'voiceref' && directorVoiceRef && (
        <Strip>
          <Row label="Identity scale" value={String(identityScale)} />
          <input type="range" min={0} max={10} step={0.5} value={identityScale}
            onChange={e => setIdentityScale(parseFloat(e.target.value))} className="w-full h-1 accent-accent-blue" />
          <p className="text-[9px] text-text-muted">~5s voice sample. With an active ID-LoRA, keeps the speaker's voice consistent across clips.</p>
        </Strip>
      )}

      {/* Option strip — references: focus mode + background removal */}
      {selected?.startsWith('ref-') && imageRefs.length > 0 && (
        <Strip>
          {hasLandscapeMode && hasPeopleMode && (
            <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
              <button onClick={() => setImageRefType('KI')}
                className={`flex-1 text-[10px] py-1 rounded-md transition-all ${imageRefType === 'KI' ? 'bg-bg-active text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}>Subject / Landscape</button>
              <button onClick={() => setImageRefType('I')}
                className={`flex-1 text-[10px] py-1 rounded-md transition-all ${imageRefType === 'I' ? 'bg-bg-active text-text-primary' : 'text-text-secondary hover:text-text-primary'}`}>People / Objects</button>
            </div>
          )}
          {hasLandscapeMode && imageRefType === 'KI' && (
            <p className="text-[9px] text-text-muted">First image is the main subject/landscape; the rest are people/objects. Drag tiles to reorder.</p>
          )}
          {refBgLabel && (
            <label className="flex items-start gap-2 cursor-pointer">
              <input type="checkbox" checked={removeBackgroundRefs} onChange={e => setRemoveBackgroundRefs(e.target.checked)} className="mt-0.5 accent-accent-blue shrink-0" />
              <span className="text-[10px] text-text-secondary leading-tight">{refBgLabel}</span>
            </label>
          )}
        </Strip>
      )}
    </div>
  )
}

function Strip({ children }: { children: React.ReactNode }) {
  return <div className="mt-2 px-1 space-y-1.5">{children}</div>
}
function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <label className="text-[10px] text-text-muted">{label}</label>
      <span className="text-[10px] text-text-muted tabular-nums">{value}</span>
    </div>
  )
}

function AddTile({ label, icon, onClick, onDropFile, dropAccept }: {
  label: string; icon?: React.ReactNode; onClick: () => void
  onDropFile?: (f: File) => void; dropAccept?: 'image' | 'audio' | 'video'
}) {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (!f || !onDropFile) return
    if (dropAccept && !f.type.startsWith(`${dropAccept}/`)) return
    onDropFile(f)
  }
  return (
    <button onClick={onClick}
      onDrop={onDropFile ? handleDrop : undefined}
      onDragOver={onDropFile ? (e => e.preventDefault()) : undefined}
      className="w-[90px] h-[90px] shrink-0 rounded-xl border border-dashed border-border hover:border-accent-blue flex flex-col items-center justify-center gap-1 text-text-muted hover:text-text-primary transition-colors">
      {icon ?? <Plus size={18} />}
      <span className="text-[10px] text-center px-1">{label}</span>
    </button>
  )
}

function Tile({ role, imgSrc, icon, badge, selected, filledIcon, filledLabel, onPick, onClear, onSelect, onDropFile }: {
  role: string
  imgSrc: string | null
  icon?: React.ReactNode
  badge?: number
  selected: boolean
  filledIcon?: React.ReactNode   // non-image filled tiles (audio/video)
  filledLabel?: string
  onPick?: () => void
  onClear: () => void
  onSelect: () => void
  onDropFile?: (f: File) => void
}) {
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith('image/') && onDropFile) onDropFile(f)
  }
  const filled = !!imgSrc || !!filledIcon
  return (
    <div onDrop={handleDrop} onDragOver={e => e.preventDefault()}
      onClick={() => (filled ? onSelect() : onPick?.())}
      className={`relative w-[90px] h-[90px] shrink-0 rounded-xl overflow-hidden cursor-pointer border transition-colors ${
        selected ? 'border-accent-blue' : filled ? 'border-border hover:border-border-light' : 'border-dashed border-border hover:border-border-light'
      }`}>
      {filled ? (
        <>
          {imgSrc ? (
            <img src={imgSrc} alt={role} className="absolute inset-0 w-full h-full object-cover" />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-bg-tertiary/50 text-text-secondary">
              {filledIcon}
              {filledLabel && <span className="text-[8px] text-text-muted px-1 truncate max-w-full">{filledLabel}</span>}
            </div>
          )}
          {badge !== undefined && <span className="absolute top-1 left-1 z-10 rounded bg-black/55 text-white text-[9px] px-1">{badge}</span>}
          <button onClick={e => { e.stopPropagation(); onClear() }}
            className="absolute top-1 right-1 z-10 rounded-full bg-black/45 text-white p-0.5 hover:bg-black/70 transition-colors" aria-label="Remove">
            <X size={12} />
          </button>
          <div className="absolute inset-x-0 bottom-0 bg-black/55 px-1.5 py-1">
            <span className="text-[10px] text-white/95">{role}</span>
          </div>
        </>
      ) : (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-bg-tertiary/40 text-text-muted">
          {icon ?? <Upload size={15} />}
          <span className="text-[10px] text-center px-1">{role}</span>
        </div>
      )}
    </div>
  )
}
