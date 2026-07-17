import { useEffect } from 'react'
import { useStore } from '../../stores/useStore'
import type { EditSubMode } from '../../types'

const ALL_SUB_MODES: { value: EditSubMode; label: string; experimental?: boolean }[] = [
  { value: 'retake', label: 'Retake' },
  { value: 'inpaint', label: 'Inpaint', experimental: true },
  { value: 'outpaint', label: 'Outpaint' },
  { value: 'restyle', label: 'Restyle', experimental: true },
  { value: 'edit_anything', label: 'Edit Anything' },
  { value: 'recast', label: 'Recast' },
]

export function EditSubModeToggle() {
  const editSubMode = useStore(s => s.editSubMode)
  const setEditSubMode = useStore(s => s.setEditSubMode)
  const showExperimental = useStore(s => !!s.servicesConfig?.show_experimental)

  // When experimental features are gated off, hide the experimental
  // sub-modes from the toggle. If the user happened to be in one of
  // them when they turned the gate off (or before it was first set),
  // snap them back to 'retake' so the toggle has something highlighted
  // and the sidebar doesn't render orphaned sub-mode controls.
  const subModes = showExperimental
    ? ALL_SUB_MODES
    : ALL_SUB_MODES.filter(m => !m.experimental)

  useEffect(() => {
    if (!showExperimental && (editSubMode === 'inpaint' || editSubMode === 'restyle')) {
      setEditSubMode('retake')
    }
  }, [showExperimental, editSubMode, setEditSubMode])

  return (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      {subModes.map(m => (
        <button
          key={m.value}
          onClick={() => setEditSubMode(m.value)}
          className={`flex-1 text-[10px] py-1.5 rounded-md transition-all whitespace-nowrap ${
            editSubMode === m.value
              ? 'bg-bg-active text-text-primary'
              : 'text-text-secondary hover:text-text-primary'
          }`}
        >
          {m.label}
        </button>
      ))}
    </div>
  )
}
