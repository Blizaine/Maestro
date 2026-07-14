import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { useStore } from '../stores/useStore'
import { VideoTimelineSelector } from './shared/VideoTimelineSelector'
import * as api from '../api/client'
import { modelDisplayName } from '../lib/modelDisplay'

export function RetakeDialog() {
  const retakeOpen = useStore(s => s.retakeDialogOpen)
  const retakeFile = useStore(s => s.retakeSourceFile)
  const closeRetake = useStore(s => s.closeRetakeDialog)
  const activeWorkspace = useStore(s => s.activeWorkspace)
  const loadOutputs = useStore(s => s.loadOutputs)
  const [startTime, setStartTime] = useState(0)
  const [endTime, setEndTime] = useState(5)
  const [duration, setDuration] = useState(0)
  const [prompt, setPrompt] = useState('')
  const [negPrompt, setNegPrompt] = useState('')
  const [regenerateAudio, setRegenerateAudio] = useState(true)
  const [seed, setSeed] = useState(-1)
  const [steps, setSteps] = useState(8)
  const [guidance, setGuidance] = useState(1.0)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const savedVideoModel = useStore(s => s.selectedModelPerMode?.video)
  const currentModel = useStore(s => s.params.model_type)
  const modelType = savedVideoModel || currentModel
  const models = useStore(s => s.models)
  const modelLabel = modelDisplayName(modelType, models)
  const activatedLoras = useStore(s => s.params.activated_loras) as string[] || []
  const lorasMultipliers = useStore(s => s.params.loras_multipliers) as string || ''

  // Get video duration on open
  useEffect(() => {
    if (!retakeFile) return
    const video = document.createElement('video')
    video.src = api.getFileUrl(retakeFile)
    video.onloadedmetadata = () => {
      const dur = video.duration && isFinite(video.duration) ? video.duration : 10
      setDuration(dur)
      setEndTime(dur)
      setStartTime(0)
    }
  }, [retakeFile])

  if (!retakeOpen || !retakeFile) return null

  const videoUrl = api.getFileUrl(retakeFile)

  const handleSubmit = async () => {
    setSubmitting(true)
    setError(null)
    setSuccess(null)
    try {
      const result = await api.submitRetake({
        video_path: retakeFile,
        start_time: startTime,
        end_time: endTime,
        prompt: prompt || 'retake',
        model_type: modelType as string,
        negative_prompt: negPrompt,
        seed,
        guidance_scale: guidance,
        num_inference_steps: steps,
        retake_engine: 'native',
        regenerate_audio: regenerateAudio,
        activated_loras: activatedLoras,
        loras_multipliers: lorasMultipliers,
        workspace: activeWorkspace,
      })
      setSuccess(`Retake queued: ${result.retake_frames}`)
      loadOutputs()
      setTimeout(() => closeRetake(), 1500)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={closeRetake}>
      <div
        className="bg-bg-secondary border border-border rounded-xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <div>
            <h2 className="text-sm font-semibold text-text-primary">Retake</h2>
            <p className="text-[10px] text-text-muted">Select the part you want to fix, then describe the change</p>
          </div>
          <button onClick={closeRetake} className="p-1 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors">
            <X size={16} />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* Timeline selector */}
          <VideoTimelineSelector
            videoUrl={videoUrl}
            duration={duration}
            startTime={startTime}
            endTime={endTime}
            onStartChange={setStartTime}
            onEndChange={setEndTime}
          />

          {/* Prompt */}
          <div>
            <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">
              What should happen in this section?
            </label>
            <textarea value={prompt}
              onChange={e => setPrompt(e.target.value)}
              placeholder="Describe the new content for the selected time range..."
              rows={2}
              className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue"
              style={{ resize: 'vertical', minHeight: 48 }} />
          </div>

          {/* Regenerate Audio */}
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={regenerateAudio}
              onChange={e => setRegenerateAudio(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-border accent-accent-blue" />
            <span className="text-xs text-text-secondary">Regenerate Audio</span>
            <span className="text-[9px] text-text-muted ml-auto">
              {regenerateAudio ? 'New audio from prompt' : 'Keep source audio'}
            </span>
          </label>

          {/* Advanced toggle */}
          <button onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-[10px] text-text-muted hover:text-text-primary transition-colors">
            {showAdvanced ? '▾' : '▸'} Advanced
          </button>
          {showAdvanced && (
            <div className="space-y-2 pl-2 border-l border-border/50">
              <div>
                <label className="text-[10px] text-text-muted uppercase tracking-wider mb-1 block">Negative Prompt</label>
                <input type="text" value={negPrompt}
                  onChange={e => setNegPrompt(e.target.value)}
                  placeholder="What to avoid..."
                  className="w-full bg-bg-tertiary border border-border rounded px-2.5 py-1.5 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-accent-blue" />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-[9px] text-text-muted block mb-0.5">Seed</label>
                  <input type="number" value={seed} onChange={e => setSeed(parseInt(e.target.value) || -1)}
                    className="w-full bg-bg-tertiary border border-border rounded px-1.5 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue" />
                </div>
                <div>
                  <label className="text-[9px] text-text-muted block mb-0.5">Steps</label>
                  <input type="number" min={1} max={50} value={steps} onChange={e => setSteps(parseInt(e.target.value) || 8)}
                    className="w-full bg-bg-tertiary border border-border rounded px-1.5 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue" />
                </div>
                <div>
                  <label className="text-[9px] text-text-muted block mb-0.5">Guidance</label>
                  <input type="number" min={0} max={20} step={0.1} value={guidance}
                    onChange={e => setGuidance(parseFloat(e.target.value) || 1.0)}
                    className="w-full bg-bg-tertiary border border-border rounded px-1.5 py-1 text-[10px] text-text-primary focus:outline-none focus:border-accent-blue" />
                </div>
              </div>
            </div>
          )}

          {/* Model + LoRA info */}
          <p className="text-[9px] text-text-muted">
            <span title={modelType}>Model: {modelLabel}</span> | Engine: Native
            {activatedLoras.length > 0 && ` | LoRAs: ${activatedLoras.length}`}
          </p>

          {/* Error/Success */}
          {error && <div className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 rounded px-2 py-1.5">{error}</div>}
          {success && <div className="text-[10px] text-indicator-success bg-green-500/10 border border-green-500/20 rounded px-2 py-1.5">{success}</div>}

          {/* Submit */}
          <button onClick={handleSubmit} disabled={submitting || !prompt}
            className="w-full py-2.5 rounded-lg text-sm font-medium bg-accent-blue text-white hover:bg-accent-blue/80 transition-colors disabled:opacity-40 disabled:cursor-not-allowed">
            {submitting ? 'Submitting...' : 'Retake'}
          </button>
        </div>
      </div>
    </div>
  )
}
