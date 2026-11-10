import { Paintbrush, RefreshCw, UsersRound, WandSparkles } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import type { EditSubMode } from '../../types'

type EditClipTask = 'retake' | 'replace_people' | 'edit_scene'

const TASKS: { value: EditClipTask; label: string; icon: typeof RefreshCw }[] = [
  { value: 'retake', label: 'Retake', icon: RefreshCw },
  { value: 'replace_people', label: 'Replace People', icon: UsersRound },
  { value: 'edit_scene', label: 'Edit Scene', icon: WandSparkles },
]

function taskForEditSubMode(mode: EditSubMode): EditClipTask {
  if (mode === 'recast') return 'replace_people'
  if (mode === 'edit_anything' || mode === 'restyle') return 'edit_scene'
  return 'retake'
}

export function EditClipTaskSelector() {
  const editSubMode = useStore(state => state.editSubMode)
  const setEditSubMode = useStore(state => state.setEditSubMode)
  const activeTask = taskForEditSubMode(editSubMode)

  return (
    <section aria-label="Edit Clip tasks" className="space-y-2 rounded-xl border border-border bg-bg-tertiary/50 p-2.5">
      <div className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Task</div>
      <div role="group" aria-label="Edit Clip task" className="grid grid-cols-3 gap-1.5">
        {TASKS.map(task => {
          const Icon = task.icon
          const selected = activeTask === task.value
          return (
            <button
              key={task.value}
              type="button"
              aria-pressed={selected}
              onClick={() => setEditSubMode(task.value === 'retake'
                ? 'retake'
                : task.value === 'replace_people'
                  ? 'recast'
                  : 'edit_anything')}
              className={`flex min-h-12 flex-col items-center justify-center gap-1 rounded-lg border px-1.5 py-1.5 text-center transition-colors ${
                selected
                  ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                  : 'border-transparent text-text-secondary hover:border-border hover:bg-bg-hover'
              }`}
            >
              <Icon size={14} className={selected ? 'text-accent-blue' : 'text-text-muted'} />
              <span className="text-[10px] leading-tight">{task.label}</span>
            </button>
          )
        })}
      </div>

      {activeTask === 'edit_scene' && (
        <div className="space-y-1.5 border-t border-border pt-2">
          <div className="text-[10px] font-medium uppercase tracking-wider text-text-muted">Edit Scene method</div>
          <div role="group" aria-label="Edit Scene method" className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              aria-pressed={editSubMode === 'edit_anything'}
              onClick={() => setEditSubMode('edit_anything')}
              className={`flex min-h-9 items-center justify-center gap-1.5 rounded-lg border px-2 text-[10px] transition-colors ${editSubMode === 'edit_anything'
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-secondary hover:bg-bg-hover'
              }`}
            >
              <WandSparkles size={12} /> Prompt Edit
            </button>
            <button
              type="button"
              aria-pressed={editSubMode === 'restyle'}
              onClick={() => setEditSubMode('restyle')}
              className={`flex min-h-9 items-center justify-center gap-1.5 rounded-lg border px-2 text-[10px] transition-colors ${editSubMode === 'restyle'
                ? 'border-accent-blue bg-accent-blue/10 text-text-primary'
                : 'border-border text-text-secondary hover:bg-bg-hover'
              }`}
            >
              <Paintbrush size={12} /> Repaint
            </button>
          </div>
        </div>
      )}
    </section>
  )
}
