import { useEffect } from 'react'
import { Crop, ImagePlus, Scan, Sparkles } from 'lucide-react'
import { modelSupportsImageWorkflow, useStore } from '../../stores/useStore'
import type { StudioImageWorkflow } from '../../types'
import {
  StudioWorkflowSelect,
  type StudioWorkflowGroup,
  type StudioWorkflowOption,
} from './StudioWorkflowSelect'

const IMAGE_WORKFLOW_GROUPS: StudioWorkflowGroup<StudioImageWorkflow>[] = [
  {
    label: 'Create',
    options: [
      { value: 'generate', label: 'Generate', description: 'Create from text or edit supplied images', icon: ImagePlus },
    ],
  },
  {
    label: 'Transform',
    options: [
      { value: 'inpaint', label: 'Inpaint', description: 'Regenerate only a masked area', icon: Crop },
      { value: 'outpaint', label: 'Outpaint', description: 'Expand beyond the original canvas', icon: Scan },
    ],
  },
  {
    label: 'Finish',
    options: [
      { value: 'upscale', label: 'Upscale', description: 'Enhance resolution and detail', icon: Sparkles },
    ],
  },
]

const OPTIONS = Object.fromEntries(
  IMAGE_WORKFLOW_GROUPS.flatMap(group => group.options).map(option => [option.value, option]),
) as Record<StudioImageWorkflow, StudioWorkflowOption<StudioImageWorkflow>>

export function ImageWorkflowSelector() {
  const workflow = useStore(state => state.studioImageWorkflow)
  const setWorkflow = useStore(state => state.setStudioImageWorkflow)
  const models = useStore(state => state.models)
  const enabledModels = useStore(state => state.enabledModels)
  const currentModelType = useStore(state => state.params.model_type)
  const hasReferenceImages = useStore(state => state.imageRefs.length > 0)
  const selectModel = useStore(state => state.selectModel)
  const nsfwMode = useStore(state => state.servicesConfig?.nsfw_mode ?? false)

  // Changing workflow should never leave a hidden/incompatible image model
  // active behind the selector. Pick the first enabled compatible model while
  // preserving the user's current choice whenever it already fits.
  useEffect(() => {
    if (workflow === 'upscale' || models.length === 0) return
    const current = models.find(model => model.model_type === currentModelType)
    if (modelSupportsImageWorkflow(current, workflow, hasReferenceImages)) return
    const fallback = models.find(model => (
      enabledModels.has(model.model_type)
      && (!model.nsfw_only || nsfwMode)
      && modelSupportsImageWorkflow(model, workflow, hasReferenceImages)
    ))
    if (fallback) selectModel(fallback.model_type)
  }, [currentModelType, enabledModels, hasReferenceImages, models, nsfwMode, selectModel, workflow])

  return (
    <StudioWorkflowSelect<StudioImageWorkflow>
      value={workflow}
      activeOption={OPTIONS[workflow]}
      groups={IMAGE_WORKFLOW_GROUPS}
      hint="Create · Transform · Finish"
      onChange={setWorkflow}
    />
  )
}
