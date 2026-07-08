import { Play, Film } from 'lucide-react'
import { useStore } from '../../stores/useStore'

export function VideoPlayer() {
  const outputs = useStore(s => s.filteredOutputs())
  const selectedOutput = useStore(s => s.selectedOutput)
  const job = useStore(s => s.jobs[0])
  const isGenerating = useStore(s => s.isGenerating)

  const selected = outputs[selectedOutput]

  return (
    <div className="flex-1 flex items-center justify-center rounded-xl bg-bg-tertiary border border-border overflow-hidden relative">
      {isGenerating && job ? (
        <div className="flex flex-col items-center gap-4 text-text-muted">
          <div className="relative">
            <Film size={48} className="animate-pulse" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-text-secondary">Generating...</p>
            <p className="text-xs mt-1">{job.message}</p>
          </div>
          <div className="w-48 bg-bg-active rounded-full h-1.5 overflow-hidden">
            <div
              className="h-full bg-accent-green rounded-full transition-all duration-300"
              style={{ width: `${job.progress * 100}%` }}
            />
          </div>
        </div>
      ) : selected ? (
        selected.type === 'video' ? (
          <video
            key={selected.url}
            src={selected.url}
            controls
            className="w-full h-full object-contain"
            loop
          />
        ) : selected.type === 'audio' ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-bg-active flex items-center justify-center">
              <Play size={24} className="text-text-muted" />
            </div>
            <p className="text-xs text-text-muted mb-2">{selected.name}</p>
            <audio key={selected.url} src={selected.url} controls className="w-64" />
          </div>
        ) : (
          <img
            key={selected.url}
            src={selected.url}
            alt={selected.name}
            className="w-full h-full object-contain"
          />
        )
      ) : (
        <div className="flex flex-col items-center gap-3 text-text-muted">
          <div className="w-16 h-16 rounded-2xl bg-bg-active flex items-center justify-center">
            <Play size={24} />
          </div>
          <p className="text-sm">Your generated videos will appear here</p>
        </div>
      )}
    </div>
  )
}
