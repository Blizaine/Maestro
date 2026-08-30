import { useEffect } from 'react'
import {
  Blend,
  FlaskConical,
  Paintbrush,
  PanelsTopLeft,
  RefreshCw,
  Scan,
  Sparkles,
  StepForward,
  UsersRound,
  WandSparkles,
} from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { StudioVideoWorkflow } from '../../types'
import {
  StudioWorkflowSelect,
  type StudioWorkflowGroup,
  type StudioWorkflowOption,
} from './StudioWorkflowSelect'

type DisplayVideoWorkflow = StudioVideoWorkflow | 'legacy_multishot' | 'legacy_inpaint'

const VIDEO_WORKFLOW_GROUPS: StudioWorkflowGroup<DisplayVideoWorkflow>[] = [
  {
    label: 'Create',
    options: [
      { value: 'frames', label: 'Generate', description: 'Text, guided-frame, or Omni generation', icon: PanelsTopLeft },
      { value: 'extend', label: 'Extend', description: 'Continue an existing clip', icon: StepForward },
      { value: 'blend', label: 'Blend', description: 'Bridge two visual anchors', icon: Blend },
    ],
  },
  {
    label: 'Transform',
    options: [
      { value: 'retake', label: 'Retake', description: 'Regenerate a selected section', icon: RefreshCw },
      { value: 'prompt_edit', label: 'Prompt Edit', description: 'Modify a clip with natural language', icon: WandSparkles },
      { value: 'outpaint', label: 'Outpaint', description: 'Expand or reframe the canvas', icon: Scan },
      { value: 'repaint', label: 'Repaint', description: 'Replace objects or regions', icon: Paintbrush },
      { value: 'recast', label: 'Recast', description: 'Replace people or characters', icon: UsersRound },
    ],
  },
  {
    label: 'Finish',
    options: [
      { value: 'upscale', label: 'Upscale', description: 'Enhance resolution and detail', icon: Sparkles },
      { value: 'film_grain', label: 'Film Grain', description: 'Add a cinematic grain finish', icon: FlaskConical },
    ],
  },
]

const OPTIONS = Object.fromEntries(
  VIDEO_WORKFLOW_GROUPS.flatMap(group => group.options).map(option => [option.value, option]),
) as Record<StudioVideoWorkflow, StudioWorkflowOption<DisplayVideoWorkflow>>

const LEGACY_MULTISHOT: StudioWorkflowOption<DisplayVideoWorkflow> = {
  value: 'legacy_multishot',
  label: 'Multi-Shot (Legacy)',
  description: 'Loaded from an older Maestro project',
  icon: PanelsTopLeft,
}

const LEGACY_INPAINT: StudioWorkflowOption<DisplayVideoWorkflow> = {
  value: 'legacy_inpaint',
  label: 'Inpaint (Legacy)',
  description: 'Loaded from an experimental Maestro project',
  icon: FlaskConical,
}

function deriveActiveWorkflow(
  generationMode: string,
  imageMode: number,
  editSubMode: string,
  toolsTool: string,
): DisplayVideoWorkflow {
  if (generationMode === 'tools' && toolsTool === 'upscale') return 'upscale'
  if (generationMode === 'tools' && toolsTool === 'film_grain') return 'film_grain'
  if (generationMode === 'avatar') {
    if (editSubMode === 'inpaint') return 'legacy_inpaint'
    if (editSubMode === 'edit_anything') return 'prompt_edit'
    if (editSubMode === 'restyle') return 'repaint'
    if (editSubMode === 'outpaint') return 'outpaint'
    if (editSubMode === 'recast') return 'recast'
    return 'retake'
  }
  if (imageMode === 2) return 'legacy_multishot'
  if (imageMode === 3) return 'extend'
  if (imageMode === 4) return 'blend'
  return 'frames'
}

export function VideoWorkflowSelector() {
  const generationMode = useStore(state => state.generationMode)
  const imageMode = Number(useStore(state => state.params.image_mode) || 0)
  const editSubMode = useStore(state => state.editSubMode)
  const toolsTool = useStore(state => state.toolsTool)
  const rememberedWorkflow = useStore(state => state.studioVideoWorkflow)
  const setWorkflow = useStore(state => state.setStudioVideoWorkflow)

  const activeWorkflow = deriveActiveWorkflow(
    generationMode,
    imageMode,
    editSubMode,
    toolsTool,
  )
  const activeOption = activeWorkflow === 'legacy_multishot'
    ? LEGACY_MULTISHOT
    : activeWorkflow === 'legacy_inpaint'
      ? LEGACY_INPAINT
      : OPTIONS[activeWorkflow]

  // Gallery loads and Edit → Image round-trips can set the legacy engine
  // modes directly. Remember any regular workflow they land on so returning
  // from Image or Audio opens the same Video workflow rather than Frames.
  useEffect(() => {
    if (
      activeWorkflow !== 'legacy_multishot'
      && activeWorkflow !== 'legacy_inpaint'
      && activeWorkflow !== rememberedWorkflow
    ) {
      setWorkflow(activeWorkflow)
    }
  }, [activeWorkflow, rememberedWorkflow, setWorkflow])

  return (
    <StudioWorkflowSelect<DisplayVideoWorkflow>
      value={activeWorkflow}
      activeOption={activeOption}
      groups={VIDEO_WORKFLOW_GROUPS}
      hint="Create · Transform · Finish"
      onChange={workflow => {
        if (workflow === 'legacy_multishot' || workflow === 'legacy_inpaint') return
        setWorkflow(workflow)
      }}
    />
  )
}
