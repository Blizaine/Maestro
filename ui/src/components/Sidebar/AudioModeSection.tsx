import { useState } from 'react'
import { Plus, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import { ChoiceControl } from '../shared/ChoiceControl'
import { FileUploadZone } from '../shared/FileUploadZone'
import * as api from '../../api/client'

export function AudioModeSection() {
  const modelOptions = useStore(s => s.modelOptions)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const audioGuideFilename = useStore(s => s.audioGuideFilename)
  const setAudioGuideFilename = useStore(s => s.setAudioGuideFilename)
  const [videoGuideFilename, setVideoGuideFilename] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  // Dynamic multi-voice state
  const ttsVoiceCount = useStore(s => s.ttsVoiceCount)
  const ttsVoices = useStore(s => s.ttsVoices)
  const addTtsVoice = useStore(s => s.addTtsVoice)
  const removeTtsVoice = useStore(s => s.removeTtsVoice)
  const setTtsVoiceName = useStore(s => s.setTtsVoiceName)
  const setTtsVoiceFile = useStore(s => s.setTtsVoiceFile)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)

  if (!modelOptions?.audio_prompt_type_sources) return null

  const isAudioOnly = modelOptions.audio_only
  const config = modelOptions.audio_prompt_type_sources
  const audioValue = (params.audio_prompt_type || config.default || '') as string
  const audioBaseMode = audioValue.replace(/[NV]/g, '')
  const needsAudioUpload = audioBaseMode.includes('A') && !isAudioOnly
  const needsVideoGuideUpload = audioValue === 'K' && !modelOptions.guide_preprocessing
  // Models that derive audio_prompt_type purely from the voice-clone slot count
  // (e.g. KugelAudio: 0→"", 1→"A", 2+→"AB") opt out of the manual ChoiceControl
  // by setting `audio_mode_from_voice_count: true` in their model_def. The Add
  // Voice / Remove Voice buttons are the only mode-selection UI for those
  // models, eliminating the dual-source-of-truth confusion that Phase 6's
  // ChoiceControl unhide (commit 19eda0b) introduced.
  const hideAudioModeChoice = Boolean((modelOptions as { audio_mode_from_voice_count?: boolean }).audio_mode_from_voice_count)
  // Max voice slots the model accepts. Defaults to 6 (Kugel); Scenema sets 2
  // since it only consumes slots 1-2 (A2 / AB2 modes). The UI caps the
  // "Add Voice" button at this limit so users aren't offered slots that
  // would be silently discarded by the backend.
  const maxVoiceCount = ((modelOptions as { max_voice_count?: number }).max_voice_count) ?? 6

  const getAudioDuration = (file: File): Promise<number | null> => {
    // Use HTML5 <video> element for video files and <audio> for audio.
    // <video> can decode pure-audio formats too (since it's a superset
    // of <audio> capability in practice), so we could use <video>
    // unconditionally — but Audio is lighter and works for the common
    // audio-file path, so we branch on MIME / extension.
    const isVideo = file.type.startsWith('video/') ||
      /\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)
    return new Promise(resolve => {
      const url = URL.createObjectURL(file)
      const el: HTMLMediaElement = isVideo
        ? document.createElement('video')
        : new Audio()
      el.addEventListener('loadedmetadata', () => {
        const dur = el.duration
        URL.revokeObjectURL(url)
        resolve(Number.isFinite(dur) ? dur : null)
      })
      el.addEventListener('error', () => { URL.revokeObjectURL(url); resolve(null) })
      el.src = url
    })
  }

  const handleLegacyUpload = async (file: File, paramKey: 'audio_guide' | 'audio_guide2' | 'video_guide', setFilename: (n: string | null) => void) => {
    setUploading(true)
    try {
      // Route audio_guide uploads through /api/v1/upload-audio, which
      // accepts both audio AND video files (extracting the audio track
      // for the latter). This lets the user drop a music video onto the
      // soundtrack slot without converting to mp3 first.
      // video_guide uploads (control video, used as a motion reference)
      // keep using the generic /api/v1/upload because the FULL video is
      // the input — extracting audio would defeat the purpose.
      const isAudioGuide = paramKey === 'audio_guide' || paramKey === 'audio_guide2'
      const result = isAudioGuide
        ? await api.uploadAudio(file)
        : await api.uploadImage(file)
      setParam(paramKey as keyof import('../../types').GenerateParams, result.path)
      // For video-uploaded-as-audio_guide, the file the backend stored is
      // an extracted WAV with a generated name. Show the original filename
      // in the UI so the user recognizes their upload, but the backing
      // path points to the extracted WAV.
      setFilename(file.name)
      if (paramKey === 'audio_guide' && !isAudioOnly) {
        const dur = await getAudioDuration(file)
        if (dur && dur > 0) setDurationSeconds(Math.round(dur * 10) / 10)
      }
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setUploading(false)
    }
  }

  const handleVoiceUpload = async (file: File, index: number) => {
    setUploading(true)
    try {
      const result = await api.uploadImage(file)
      setTtsVoiceFile(index, file.name, result.path)
      // Also set legacy params for backward compat
      const key = index === 0 ? 'audio_guide' : `audio_guide${index + 1}`
      setParam(key as keyof import('../../types').GenerateParams, result.path)
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setUploading(false)
    }
  }

  const clearVoice = (index: number) => {
    setTtsVoiceFile(index, null, null)
    const key = index === 0 ? 'audio_guide' : `audio_guide${index + 1}`
    setParam(key as keyof import('../../types').GenerateParams, undefined)
  }

  return (
    <div className="space-y-3">
      {/* Audio mode selector — shown for any model that exposes audio_prompt_type_sources
          UNLESS the model opted into voice-count-driven mode (KugelAudio). For those,
          the voice-slot buttons below are the only mode UI to eliminate dual-source-
          of-truth drift. Previously gated on !isAudioOnly (Phase 6 commit 19eda0b
          removed that to let Scenema/Index TTS2 show their explicit text/single-ref/
          two-ref choices). */}
      {!hideAudioModeChoice && (
        <ChoiceControl
          config={config}
          value={audioBaseMode}
          onChange={val => {
            const flags = audioValue.replace(/[^NV]/g, '')
            setParam('audio_prompt_type', val + flags)
            if (!val.includes('A')) {
              setParam('audio_guide', undefined)
              setAudioGuideFilename(null)
            }
            if (val !== 'K') {
              setParam('video_guide', undefined)
              setVideoGuideFilename(null)
            }
            const isAudioDriven = val.includes('A') && !isAudioOnly
            setParam('modality_scale' as keyof import('../../types').GenerateParams, isAudioDriven ? 1.0 : 1.0)
          }}
          label="Audio Mode"
        />
      )}

      {/* TTS Voice Cloning — dynamic 1-6 voices */}
      {isAudioOnly && (
        <div className="space-y-2">
          {/* Add Voice button — always at top */}
          {ttsVoiceCount < maxVoiceCount && (
            <button
              onClick={addTtsVoice}
              className="w-full py-1.5 rounded-lg text-[10px] font-medium border border-dashed border-border text-text-muted hover:text-text-primary hover:border-border-light transition-colors flex items-center justify-center gap-1.5"
            >
              <Plus size={12} />
              {ttsVoiceCount === 0 ? 'Add Voice Clone' : `Add Voice (${ttsVoiceCount}/${maxVoiceCount})`}
            </button>
          )}

          {ttsVoiceCount === 0 && (
            <p className="text-[9px] text-text-muted text-center">
              Text-only mode. Add voices to clone specific speakers.
            </p>
          )}

          {/* Voice zones grid — 2 columns */}
          {ttsVoiceCount > 0 && (
            <div className="grid grid-cols-2 gap-2">
              {ttsVoices.slice(0, ttsVoiceCount).map((voice, i) => (
                <div key={i} className="bg-bg-tertiary/50 border border-border rounded-lg p-2 relative">
                  <button
                    onClick={() => removeTtsVoice(i)}
                    className="absolute -top-1.5 -right-1.5 p-0.5 rounded-full bg-bg-secondary border border-border text-text-muted hover:text-red-400 hover:border-red-400/50 transition-colors z-10"
                  >
                    <X size={10} />
                  </button>
                  <label className="text-[9px] text-text-muted uppercase tracking-wider block mb-1">
                    Voice {i + 1}
                  </label>
                  <FileUploadZone
                    label={uploading ? '...' : 'Drop audio'}
                    accept=".wav,.mp3,.flac,.ogg,.m4a"
                    filename={voice.filename}
                    onFile={f => handleVoiceUpload(f, i)}
                    onClear={() => clearVoice(i)}
                  />
                  <input
                    type="text"
                    placeholder="Speaker name"
                    value={voice.name}
                    onChange={e => setTtsVoiceName(i, e.target.value)}
                    className="w-full mt-1 bg-bg-tertiary border border-border rounded px-1.5 py-0.5 text-[9px] text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
                  />
                </div>
              ))}
            </div>
          )}

          {/* Audio processing toggles — show when any voices are active */}
          {ttsVoiceCount > 0 && (
            <div className="space-y-1.5 pt-1">
              {ttsVoiceCount >= 2 && (
                <p className="text-[9px] text-text-muted">
                  Names auto-fill from prompt. Each name maps to the voice above it.
                </p>
              )}
              <label className="flex items-center gap-2 cursor-pointer group">
                <input type="checkbox"
                  checked={audioValue.includes('N')}
                  onChange={e => {
                    const current = (params.audio_prompt_type || '') as string
                    setParam('audio_prompt_type', e.target.checked ? current + 'N' : current.replace('N', ''))
                  }}
                  className="accent-accent-blue" />
                <span className="text-[10px] text-text-secondary group-hover:text-text-primary transition-colors">
                  Normalize audio volumes
                </span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer group">
                <input type="checkbox"
                  checked={audioValue.includes('V')}
                  onChange={e => {
                    const current = (params.audio_prompt_type || '') as string
                    setParam('audio_prompt_type', e.target.checked ? current + 'V' : current.replace('V', ''))
                  }}
                  className="accent-accent-blue" />
                <span className="text-[10px] text-text-secondary group-hover:text-text-primary transition-colors">
                  Remove background music
                </span>
              </label>
              {ttsVoiceCount >= 2 && (
                <label className="flex items-center gap-2 cursor-pointer group">
                  <input type="checkbox"
                    checked={!!(params as unknown as Record<string, unknown>).tts_dynaudnorm}
                    onChange={e => setParam('tts_dynaudnorm' as keyof import('../../types').GenerateParams, e.target.checked ? 1 : undefined)}
                    className="accent-accent-blue" />
                  <span className="text-[10px] text-text-secondary group-hover:text-text-primary transition-colors">
                    Smooth speaker volumes
                  </span>
                </label>
              )}
            </div>
          )}
        </div>
      )}

      {/* Non-TTS: Audio file upload (LTX soundtrack mode).
          Accepts both audio files AND video files — backend extracts
          the audio track from video via ffmpeg before storing as WAV. */}
      {!isAudioOnly && needsAudioUpload && (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
            Audio File
          </label>
          <FileUploadZone
            label={uploading ? 'Uploading...' : 'Drop audio or video (audio will be extracted)'}
            accept=".wav,.mp3,.flac,.ogg,.m4a,.mp4,.mov,.mkv,.webm,.avi,.m4v"
            filename={audioGuideFilename}
            onFile={file => handleLegacyUpload(file, 'audio_guide', setAudioGuideFilename)}
            onClear={() => {
              setParam('audio_guide', undefined)
              setAudioGuideFilename(null)
            }}
          />
        </div>
      )}

      {/* Non-TTS: Audio strength slider (when audio file is uploaded) */}
      {!isAudioOnly && needsAudioUpload && audioGuideFilename && (
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-[10px] text-text-muted uppercase tracking-wider">
              {modelOptions?.audio_scale_name || 'Prompt Audio Strength'}
            </label>
            <span className="text-[10px] text-text-secondary">
              {((params as unknown as Record<string, unknown>).modality_scale as number ?? 1.0).toFixed(1)}
            </span>
          </div>
          <input
            type="range" min={0.1} max={3.0} step={0.1}
            value={(params as unknown as Record<string, unknown>).modality_scale as number ?? 1.0}
            onChange={e => setParam('modality_scale' as keyof import('../../types').GenerateParams, parseFloat(e.target.value))}
            className="w-full accent-accent-blue"
          />
          <div className="flex justify-between text-[9px] text-text-muted mt-0.5">
            <span>0.1</span><span>1.0 (Default)</span><span>3.0 (Experimental TTS Boost)</span>
          </div>
        </div>
      )}

      {/* Non-TTS: Video guide upload (control video for soundtrack) */}
      {!isAudioOnly && needsVideoGuideUpload && (
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
            Control Video
          </label>
          <FileUploadZone
            label={uploading ? 'Uploading...' : 'Drop video file (.mp4)'}
            accept=".mp4,.webm,.mkv"
            filename={videoGuideFilename}
            onFile={file => handleLegacyUpload(file, 'video_guide', setVideoGuideFilename)}
            onClear={() => {
              setParam('video_guide', undefined)
              setVideoGuideFilename(null)
            }}
          />
        </div>
      )}
    </div>
  )
}
