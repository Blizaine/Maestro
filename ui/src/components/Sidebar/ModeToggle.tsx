import { useStore } from '../../stores/useStore'

export function ModeToggle() {
  const imageMode = useStore(s => s.params.image_mode)
  const setParam = useStore(s => s.setParam)

  const modes = [
    { value: 0, label: 'Frames' },
    { value: 2, label: 'Multi-Shot' },
    { value: 3, label: 'Extend' },
    { value: 4, label: 'Blend' },
  ]

  return (
    <div className="flex bg-bg-tertiary rounded-lg p-0.5 border border-border">
      {modes.map(m => (
        <button
          key={m.value}
          onClick={() => setParam('image_mode', m.value)}
          className={`flex-1 text-xs py-1.5 rounded-md transition-all ${
            imageMode === m.value
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
