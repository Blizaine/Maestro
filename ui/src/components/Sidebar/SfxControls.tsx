import { useState } from 'react'
import { useStore } from '../../stores/useStore'
import { FileUploadZone } from '../shared/FileUploadZone'
import * as api from '../../api/client'

/**
 * SFX mode controls for MMAudio — sound effects generation.
 * User can optionally upload a video clip, plus prompt/neg prompt and duration.
 */
export function SfxControls() {
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)
  const durationSeconds = useStore(s => s.durationSeconds)
  const setDurationSeconds = useStore(s => s.setDurationSeconds)
  const [videoFilename, setVideoFilename] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)

  const sfxPrompt = ((params as unknown as Record<string, unknown>).MMAudio_prompt as string) || ''
  const sfxNegPrompt = ((params as unknown as Record<string, unknown>).MMAudio_neg_prompt as string) || ''
  const textWeight = ((params as unknown as Record<string, unknown>).sfx_text_weight as number) ?? 1.0

  const handleVideoUpload = async (file: File) => {
    setUploading(true)
    try {
      const result = await api.uploadImage(file)
      setParam('video_guide' as keyof typeof params, result.path)
      setVideoFilename(file.name)

      // Try to get video duration
      const url = URL.createObjectURL(file)
      const video = document.createElement('video')
      video.preload = 'metadata'
      video.onloadedmetadata = () => {
        if (Number.isFinite(video.duration) && video.duration > 0) {
          setDurationSeconds(Math.round(video.duration * 10) / 10)
        }
        URL.revokeObjectURL(url)
      }
      video.onerror = () => URL.revokeObjectURL(url)
      video.src = url
    } catch (e) {
      console.error('Upload failed:', e)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Video clip upload (optional) */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Video Clip <span className="normal-case text-text-muted">(optional)</span>
        </label>
        <FileUploadZone
          label={uploading ? 'Uploading...' : 'Drop video to generate matching audio'}
          accept=".mp4,.webm,.avi,.mov,.mkv"
          filename={videoFilename}
          onFile={handleVideoUpload}
          onClear={() => {
            setParam('video_guide' as keyof typeof params, undefined)
            setVideoFilename(null)
          }}
        />
        <p className="text-[9px] text-text-muted mt-1">
          With video: generates matching sound effects. Without: generates from text prompt.
        </p>
      </div>

      {/* Duration (shown when no video — max 20s, MMAudio single-pass limit) */}
      {!videoFilename && (
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-[11px] text-text-muted uppercase tracking-wider">Duration</label>
            <span className="text-xs text-text-secondary">{Math.min(durationSeconds, 20)}s</span>
          </div>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={Math.min(durationSeconds, 20)}
            onChange={e => setDurationSeconds(Number(e.target.value))}
          />
        </div>
      )}

      {/* SFX Prompt */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Sound Description
        </label>
        <textarea
          value={sfxPrompt}
          onChange={e => setParam('MMAudio_prompt' as keyof typeof params, e.target.value)}
          placeholder="e.g. rain falling on roof, distant thunder, birds chirping"
          rows={2}
          className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
          style={{ resize: 'vertical', minHeight: 48 }}
        />
      </div>

      {/* Negative prompt */}
      <div>
        <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
          Negative Prompt
        </label>
        <input
          type="text"
          value={sfxNegPrompt}
          onChange={e => setParam('MMAudio_neg_prompt' as keyof typeof params, e.target.value)}
          placeholder="e.g. music, speech, talking"
          className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
        />
      </div>

      {/* Text prompt weight */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label className="text-[11px] text-text-muted uppercase tracking-wider">
            Prompt Strength
          </label>
          <span className="text-xs text-text-secondary">{textWeight.toFixed(1)}x</span>
        </div>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={textWeight}
          onChange={e => setParam('sfx_text_weight' as keyof typeof params, parseFloat(e.target.value))}
        />
        <p className="text-[9px] text-text-muted mt-0.5">
          How strongly the text prompt influences the output vs the video content.
        </p>
      </div>
    </div>
  )
}
