import { useEffect } from 'react'
import {
  Blend,
  FlaskConical,
  Globe,
  Images,
  Film,
  PanelsTopLeft,
  Scan,
  Sparkles,
  StepForward,
  Type,
  UsersRound,
} from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { StudioVideoWorkflow } from '../../types'
import {
  StudioWorkflowSelect,
  type StudioWorkflowGroup,
  type StudioWorkflowOption,
} from './StudioWorkflowSelect'

type MenuVideoWorkflow = Exclude<StudioVideoWorkflow, 'retake' | 'prompt_edit' | 'repaint' | 'recast'>
type ActiveVideoWorkflow = MenuVideoWorkflow | 'edit_clip' | 'legacy_multishot' | 'legacy_inpaint'
type DisplayVideoWorkflow = ActiveVideoWorkflow | 'worlds'

const VIDEO_WORKFLOW_GROUPS: StudioWorkflowGroup<DisplayVideoWorkflow>[] = [
  {
    label: 'Create',
    options: [
      { value: 'frames', label: 'Frames', description: 'Text, first / last, injected frames, audio, or control video', icon: PanelsTopLeft },
      { value: 'references', label: 'References', description: 'Generate with H3 Omni people, scenes, motion, or voices', icon: Images },
      { value: 'avatar', label: 'Avatar', description: 'Animate people from an anchor image and one or two voice tracks', icon: UsersRound },
      { value: 'extend', label: 'Extend', description: 'Continue an existing clip', icon: StepForward },
      { value: 'blend', label: 'Blend', description: 'Bridge two visual anchors', icon: Blend },
      { value: 'titles', label: 'Titles', description: 'Create video titles and send them to the gallery', icon: Type },
    ],
  },
  {
    label: 'Animate',
    options: [
      { value: 'animate', label: 'Animate', description: 'Animate an edited frame with a control video', icon: UsersRound },
    ],
  },
  {
    label: 'Edit',
    options: [
      { value: 'edit_clip', label: 'Edit Clip', description: 'Retake, replace people, or edit a scene in an existing clip', icon: Film },
      { value: 'outpaint', label: 'Outpaint', description: 'Expand or reframe the canvas around a clip', icon: Scan },
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
  VIDEO_WORKFLOW_GROUPS.flatMap(group => group.options)
    .filter(option => option.value !== 'edit_clip')
    .map(option => [option.value, option]),
) as Record<MenuVideoWorkflow, StudioWorkflowOption<DisplayVideoWorkflow>>

const EDIT_CLIP: StudioWorkflowOption<DisplayVideoWorkflow> = {
  value: 'edit_clip',
  label: 'Edit Clip',
  description: 'Retake, replace people, or edit a scene in an existing clip',
  icon: Film,
}

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
  rememberedWorkflow: StudioVideoWorkflow,
): ActiveVideoWorkflow {
  if (generationMode === 'tools' && toolsTool === 'upscale') return 'upscale'
  if (generationMode === 'tools' && toolsTool === 'film_grain') return 'film_grain'
  if (generationMode === 'avatar') {
    if (editSubMode === 'inpaint') return 'legacy_inpaint'
    if (editSubMode === 'outpaint') return 'outpaint'
    return 'edit_clip'
  }
  // Titles has its own UI and always uses the regular video output path. Its
  // durable workflow marker wins over a stale specialized image_mode value.
  if (generationMode === 'video' && rememberedWorkflow === 'titles') return 'titles'
  if (imageMode === 2) return 'legacy_multishot'
  if (imageMode === 3) return 'extend'
  if (imageMode === 4) return 'blend'
  // Load Settings restores the user-facing workflow explicitly. Several
  // specialized endpoints normalize their engine payload back to
  // image_mode=0 before writing the sidecar (Extend is the important
  // example), so image_mode alone cannot distinguish Frames from the mode
  // that actually produced the clip. Trust the durable workflow marker for
  // every ordinary Video workflow; the generationMode branches above remain
  // authoritative for Transform and Finish outputs.
  if (
    rememberedWorkflow === 'animate'
    || rememberedWorkflow === 'avatar'
    || rememberedWorkflow === 'frames'
    || rememberedWorkflow === 'references'
    || rememberedWorkflow === 'extend'
    || rememberedWorkflow === 'blend'
  ) return rememberedWorkflow
  return 'frames'
}

export function VideoWorkflowSelector({ onOpenWorlds }: { onOpenWorlds?: () => void } = {}) {
  const worldsEnabled = useStore(state => state.servicesConfig?.worlds_enabled === true)
  const generationMode = useStore(state => state.generationMode)
  const imageMode = Number(useStore(state => state.params.image_mode) || 0)
  const editSubMode = useStore(state => state.editSubMode)
  const toolsTool = useStore(state => state.toolsTool)
  const rememberedWorkflow = useStore(state => state.studioVideoWorkflow)
  const setWorkflow = useStore(state => state.setStudioVideoWorkflow)
  const modelType = useStore(state => state.params.model_type)
  const viggleAvailable = useStore(state => state.models.some(model => model.model_type === 'viggle_animate'))
  const groups = worldsEnabled && onOpenWorlds
    ? VIDEO_WORKFLOW_GROUPS.map(group => group.label === 'Create'
      ? { ...group, options: [...group.options, {
        value: 'worlds' as const,
        label: 'Worlds',
        description: 'Experimental · Create and explore saved worlds',
        icon: Globe,
      }] }
      : group)
    : VIDEO_WORKFLOW_GROUPS

  const activeWorkflow = deriveActiveWorkflow(
    generationMode,
    imageMode,
    editSubMode,
    toolsTool,
    rememberedWorkflow,
  )
  const activeOption = activeWorkflow === 'legacy_multishot'
    ? LEGACY_MULTISHOT
    : activeWorkflow === 'legacy_inpaint'
      ? LEGACY_INPAINT
      : activeWorkflow === 'edit_clip'
        ? EDIT_CLIP
        : OPTIONS[activeWorkflow]

  // Gallery loads and Edit → Image round-trips can set the legacy engine
  // modes directly. Remember any regular workflow they land on so returning
  // from Image or Audio opens the same Video workflow rather than Frames.
  useEffect(() => {
    if (
      activeWorkflow !== 'legacy_multishot'
      && activeWorkflow !== 'legacy_inpaint'
      && activeWorkflow !== 'edit_clip'
      && (activeWorkflow !== rememberedWorkflow
        || (activeWorkflow === 'animate' && viggleAvailable && modelType !== 'viggle_animate')
        || (activeWorkflow === 'titles' && imageMode !== 0))
    ) {
      setWorkflow(activeWorkflow)
    }
  }, [activeWorkflow, imageMode, rememberedWorkflow, setWorkflow, viggleAvailable, modelType])

  return (
    <StudioWorkflowSelect<DisplayVideoWorkflow>
      value={activeWorkflow}
      activeOption={activeOption}
      groups={groups}
      hint="Create · Animate · Edit · Finish"
      onChange={workflow => {
        if (workflow === 'worlds') {
          onOpenWorlds?.()
          return
        }
        if (workflow === 'legacy_multishot' || workflow === 'legacy_inpaint') return
        if (workflow === 'edit_clip') {
          const state = useStore.getState()
          if (state.generationMode !== 'avatar') state.setGenerationMode('avatar')
          if (!['retake', 'recast', 'edit_anything', 'restyle'].includes(state.editSubMode)) {
            useStore.getState().setEditSubMode('retake')
            } else {
              useStore.getState().setEditSubMode(state.editSubMode)
          }
          return
        }
        setWorkflow(workflow)
      }}
    />
  )
}
