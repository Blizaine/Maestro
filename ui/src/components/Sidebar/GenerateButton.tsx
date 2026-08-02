import { useState, useEffect } from 'react'
import { Play, AlertTriangle } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function GenerateButton() {
  const jobs = useStore(s => s.jobs)
  const startGeneration = useStore(s => s.startGeneration)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const [cooldown, setCooldown] = useState(false)

  // Check if i2v-only model needs a start image. Video mode only: edit
  // sub-modes supply their own source media (Recast runs the i2v-only
  // SCAIL-2 against a source video + reference image, no start image).
  const generationMode = useStore(s => s.generationMode)
  const isI2vOnly = useStore(s => s.modelOptions?.i2v_class && !s.modelOptions?.t2v_class)
  const hasStartImage = useStore(s => !!(s.startImage || s.params.image_start))
  const needsImage = generationMode === 'video' && isI2vOnly && !hasStartImage
  const editSubMode = useStore(s => s.editSubMode)
  const editVideoPath = useStore(s => s.editVideoPath)
  const outpaintVideoBox = useStore(s => s.outpaintVideoBox)
  const isOutpaint = generationMode === 'avatar' && editSubMode === 'outpaint'
  const needsOutpaintSource = isOutpaint && !editVideoPath
  const hasOutpaintArea = (
    outpaintVideoBox.x > 0.0005
    || outpaintVideoBox.y > 0.0005
    || outpaintVideoBox.x + outpaintVideoBox.w < 0.9995
    || outpaintVideoBox.y + outpaintVideoBox.h < 0.9995
  )
  const needsOutpaintArea = isOutpaint && !!editVideoPath && !hasOutpaintArea
  const blocked = needsImage || needsOutpaintSource || needsOutpaintArea

  // Brief gray flash after clicking
  useEffect(() => {
    if (!cooldown) return
    const timer = setTimeout(() => setCooldown(false), 1000)
    return () => clearTimeout(timer)
  }, [cooldown])

  const handleClick = () => {
    if (blocked) return
    setCooldown(true)
    startGeneration()
    setSidebarOpen(false)
  }

  const queueCount = jobs.length

  if (blocked) {
    const label = needsImage
      ? 'Need image'
      : needsOutpaintSource
        ? 'Need source'
        : 'Choose canvas'
    const title = needsOutpaintArea
      ? 'Choose a larger output aspect or resize the source to create an area for Outpaint to generate.'
      : undefined
    return (
      <button
        disabled
        title={title}
        className="px-4 py-2 rounded-lg flex items-center gap-1.5 bg-amber-500/20 text-indicator-warning cursor-not-allowed text-xs font-medium whitespace-nowrap"
      >
        <AlertTriangle size={13} />
        {label}
      </button>
    )
  }

  return (
    <button
      onClick={handleClick}
      disabled={cooldown}
      className={`px-4 py-2 rounded-lg flex items-center gap-1.5 font-medium text-xs transition-all whitespace-nowrap ${
        cooldown
          ? 'bg-bg-active text-text-muted cursor-not-allowed'
          // Default theme: bg-cta resolves to a flat accent-blue (both
          // gradient stops point at --color-accent-blue). Golden Hour:
          // resolves to a red→orange sunset gradient. shadow-accent-glow
          // is empty in default and a warm bloom in Golden Hour.
          : 'bg-cta hover:brightness-110 shadow-accent-glow text-white'
      }`}
    >
      <Play size={13} fill={cooldown ? 'currentColor' : 'white'} />
      {cooldown ? 'Queued' : queueCount > 0 ? `Go (${queueCount})` : 'Generate'}
    </button>
  )
}
