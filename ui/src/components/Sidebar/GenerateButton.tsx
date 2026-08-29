import { useState } from 'react'
import { AlertTriangle, ListPlus, Loader2, Play } from 'lucide-react'
import { modelSupportsImageWorkflow, useStore } from '../../stores/useStore'

export function GenerateButton() {
  const startGeneration = useStore(s => s.startGeneration)
  const setSidebarOpen = useStore(s => s.setSidebarOpen)
  const [pendingAction, setPendingAction] = useState<'generate' | 'queue' | null>(null)

  // Check if i2v-only model needs a start image. Video mode only: edit
  // sub-modes supply their own source media (Recast runs the i2v-only
  // SCAIL-2 against a source video + reference image, no start image).
  const generationMode = useStore(s => s.generationMode)
  const isI2vOnly = useStore(s => s.modelOptions?.i2v_class && !s.modelOptions?.t2v_class)
  const isOmniReference = useStore(s => s.modelOptions?.omni_reference === true)
  const hasOmniVisualReference = useStore(s =>
    s.params.minimax_h3_references?.some(
      reference => reference.type === 'image' || reference.type === 'video',
    ) === true,
  )
  const hasStartImage = useStore(s => !!(s.startImage || s.params.image_start))
  const needsImage = generationMode === 'video' && isI2vOnly && !isOmniReference && !hasStartImage
  const needsReference = generationMode === 'video' && isOmniReference
    && !hasOmniVisualReference
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
  const imageWorkflow = useStore(s => s.studioImageWorkflow)
  const imageSourcePath = useStore(s => s.imageWorkflowSourcePath)
  const imageMaskPath = useStore(s => s.imageWorkflowMaskPath)
  const imageRefs = useStore(s => s.imageRefs)
  const imagePadding = useStore(s => s.imageOutpaintPadding)
  const currentModel = useStore(s => s.models.find(model => model.model_type === s.params.model_type))
  const needsImageEditSource = generationMode === 'image'
    && imageWorkflow === 'edit'
    && imageRefs.length === 0
  const needsImageWorkflowSource = generationMode === 'image'
    && (imageWorkflow === 'inpaint' || imageWorkflow === 'outpaint')
    && !imageSourcePath
  const needsImageMask = generationMode === 'image'
    && imageWorkflow === 'inpaint'
    && !imageMaskPath
  const needsImageOutpaintArea = generationMode === 'image'
    && imageWorkflow === 'outpaint'
    && Object.values(imagePadding).every(value => value === 0)
  const incompatibleImageModel = generationMode === 'image'
    && !modelSupportsImageWorkflow(currentModel, imageWorkflow)
  const blocked = needsImage || needsReference || needsOutpaintSource || needsOutpaintArea
    || needsImageEditSource || needsImageWorkflowSource || needsImageMask
    || needsImageOutpaintArea || incompatibleImageModel
  const imageMode = useStore(s => Number(s.params.image_mode || 0))
  const queueSupported = generationMode !== 'avatar'
    && !(generationMode === 'video' && imageMode === 4)

  const submit = async (action: 'generate' | 'queue') => {
    if (blocked || pendingAction || (action === 'queue' && !queueSupported)) return
    setPendingAction(action)
    if (action === 'generate') setSidebarOpen(false)
    try {
      await startGeneration(action === 'queue' ? 'queue' : 'now')
    } finally {
      setPendingAction(null)
    }
  }

  if (blocked) {
    const label = needsImage
      ? 'Need image'
      : needsReference
        ? 'Need reference'
      : incompatibleImageModel
        ? 'Need model'
      : needsImageEditSource || needsImageWorkflowSource
        ? 'Need source'
      : needsImageMask
        ? 'Need mask'
      : needsImageOutpaintArea
        ? 'Choose canvas'
      : needsOutpaintSource
        ? 'Need source'
        : 'Choose canvas'
    const title = needsOutpaintArea
      ? 'Choose a larger output aspect or resize the source to create an area for Outpaint to generate.'
      : needsReference
        ? 'Add at least one image or video reference. Audio cannot be the only reference.'
        : incompatibleImageModel
          ? 'Enable or select a model compatible with this Image workflow.'
        : needsImageMask
          ? 'Upload a black-and-white mask. White areas will be regenerated.'
        : needsImageOutpaintArea
          ? 'Expand at least one side of the source canvas.'
        : undefined
    return (
      <div className="grid w-[132px] shrink-0 grid-cols-[2fr_1fr] overflow-hidden rounded-lg bg-amber-500/20 text-indicator-warning">
        <button
          type="button"
          disabled
          title={title}
          className="flex cursor-not-allowed items-center justify-center gap-1.5 whitespace-nowrap px-2 py-2 text-xs font-medium"
        >
          <AlertTriangle size={13} />
          {label}
        </button>
        <button
          type="button"
          disabled
          title={title || `${label} before adding this generation to the queue.`}
          aria-label="Add to queue unavailable"
          className="flex cursor-not-allowed items-center justify-center border-l border-current/15"
        >
          <ListPlus size={14} />
        </button>
      </div>
    )
  }

  const pending = pendingAction !== null

  return (
    <div className={`grid w-[132px] shrink-0 grid-cols-[2fr_1fr] overflow-hidden rounded-lg font-medium text-white shadow-accent-glow transition-all ${
      pending ? 'bg-bg-active text-text-muted' : 'bg-cta'
    }`}>
      <button
        type="button"
        onClick={() => void submit('generate')}
        disabled={pending}
        title="Generate now"
        className="flex items-center justify-center gap-1.5 whitespace-nowrap px-2 py-2 text-xs transition-colors hover:bg-white/10 disabled:cursor-wait disabled:hover:bg-transparent"
      >
        {pendingAction === 'generate'
          ? <Loader2 size={13} className="animate-spin" />
          : <Play size={13} fill="currentColor" />}
        Generate
      </button>
      <button
        type="button"
        onClick={() => void submit('queue')}
        disabled={pending || !queueSupported}
        title={queueSupported
          ? 'Hold current Studio settings in the queue without starting generation'
          : 'Add to Queue is not available for specialized Transform and Blend workflows yet'}
        aria-label="Add current Studio settings to the queue"
        className="flex items-center justify-center border-l border-white/20 transition-colors hover:bg-white/10 disabled:cursor-not-allowed disabled:border-text-muted/20 disabled:opacity-40 disabled:hover:bg-transparent"
      >
        {pendingAction === 'queue'
          ? <Loader2 size={14} className="animate-spin" />
          : <ListPlus size={14} />}
      </button>
    </div>
  )
}
