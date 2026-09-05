import { Check, ChevronDown, type LucideIcon } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

export interface StudioWorkflowOption<T extends string> {
  value: T
  label: string
  description: string
  icon: LucideIcon
}

export interface StudioWorkflowGroup<T extends string> {
  label: string
  options: StudioWorkflowOption<T>[]
}

interface StudioWorkflowSelectProps<T extends string> {
  value: T
  activeOption: StudioWorkflowOption<T>
  groups: StudioWorkflowGroup<T>[]
  hint: string
  onChange: (value: T) => void
}

/**
 * Compact grouped workflow picker shared by Studio's media workspaces.
 * It stays in document flow while open, which avoids clipping inside the
 * scrollable sidecar and remains comfortable to use on narrow mobile drawers.
 */
export function StudioWorkflowSelect<T extends string>({
  value,
  activeOption,
  groups,
  hint,
  onChange,
}: StudioWorkflowSelectProps<T>) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const ActiveIcon = activeOption.icon

  useEffect(() => {
    if (!open) return
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false)
      }
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', closeOnOutsideClick)
    document.addEventListener('keydown', closeOnEscape)
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick)
      document.removeEventListener('keydown', closeOnEscape)
    }
  }, [open])

  return (
    <div ref={rootRef}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] text-text-muted uppercase tracking-wider">Workflow</span>
        <span className="text-[10px] text-text-muted">{hint}</span>
      </div>
      <button
        type="button"
        onClick={() => setOpen(current => !current)}
        aria-expanded={open}
        className={`w-full flex items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-colors ${
          open
            ? 'border-accent-blue bg-accent-blue/10'
            : 'border-border bg-bg-tertiary hover:border-border-light'
        }`}
      >
        <ActiveIcon size={15} className="shrink-0 text-accent-blue" />
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-medium text-text-primary truncate">
            {activeOption.label}
          </span>
          <span className="block mt-0.5 text-[10px] text-text-muted truncate">
            {activeOption.description}
          </span>
        </span>
        <ChevronDown
          size={14}
          className={`shrink-0 text-text-muted transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </button>

      {open && (
        <div className="mt-1.5 rounded-xl border border-border bg-bg-secondary shadow-xl p-1.5">
          {groups.map((group, groupIndex) => (
            <div
              key={group.label}
              className={groupIndex > 0 ? 'mt-1 pt-1.5 border-t border-border' : ''}
            >
              <div className="px-2 py-1 text-[10px] text-text-muted uppercase tracking-wider font-medium">
                {group.label}
              </div>
              {group.options.map(option => {
                const Icon = option.icon
                const selected = option.value === value
                return (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => {
                      onChange(option.value)
                      setOpen(false)
                    }}
                    className={`w-full grid grid-cols-[26px_minmax(0,1fr)_16px] items-center gap-2 rounded-lg px-2 py-1.5 text-left transition-colors ${
                      selected
                        ? 'bg-accent-blue/10 text-text-primary'
                        : 'text-text-secondary hover:bg-bg-hover hover:text-text-primary'
                    }`}
                  >
                    <span className="w-6 h-6 rounded-md bg-accent-blue/10 text-accent-blue flex items-center justify-center">
                      <Icon size={13} />
                    </span>
                    <span className="min-w-0">
                      <span className="block text-xs font-medium truncate">{option.label}</span>
                      <span className="block text-[10px] text-text-muted truncate">{option.description}</span>
                    </span>
                    {selected ? <Check size={13} className="text-accent-blue" /> : <span />}
                  </button>
                )
              })}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
