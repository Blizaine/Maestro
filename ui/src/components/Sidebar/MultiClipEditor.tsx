import { Upload, X } from 'lucide-react'
import { useStore } from '../../stores/useStore'

function ClipDropZone({ file, onFile, onClear }: {
  file: File | null
  onFile: (f: File) => void
  onClear: () => void
}) {
  return (
    <div
      className="relative border border-dashed border-border rounded-lg p-2 flex items-center justify-center gap-1 cursor-pointer hover:border-border-light transition-colors min-h-[60px]"
      onDrop={e => {
        e.preventDefault()
        const f = e.dataTransfer.files[0]
        if (f && f.type.startsWith('image/')) onFile(f)
      }}
      onDragOver={e => e.preventDefault()}
      onClick={() => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = 'image/*'
        input.onchange = (ev) => {
          const f = (ev.target as HTMLInputElement).files?.[0]
          if (f) onFile(f)
        }
        input.click()
      }}
    >
      {file ? (
        <>
          <img
            src={URL.createObjectURL(file)}
            alt="clip"
            className="w-full h-full object-cover rounded absolute inset-0"
          />
          <button
            onClick={e => { e.stopPropagation(); onClear() }}
            className="absolute top-1 right-1 bg-bg-primary/80 rounded-full p-0.5 hover:bg-bg-hover z-10"
          >
            <X size={10} />
          </button>
        </>
      ) : (
        <>
          <Upload size={14} className="text-text-muted" />
          <span className="text-[10px] text-text-muted">Start Image</span>
        </>
      )}
    </div>
  )
}

export function MultiClipEditor() {
  const clips = useStore(s => s.clips)
  const singlePromptMode = useStore(s => s.singlePromptMode)
  const setSinglePromptMode = useStore(s => s.setSinglePromptMode)
  const setClipPrompt = useStore(s => s.setClipPrompt)
  const setClipStartImage = useStore(s => s.setClipStartImage)
  const slidingWindowSeconds = useStore(s => s.slidingWindowSeconds)

  if (clips.length === 0) return null

  return (
    <div className="space-y-3">
      <label className="flex items-center gap-2 text-xs text-text-secondary cursor-pointer">
        <input
          type="checkbox"
          checked={singlePromptMode}
          onChange={e => setSinglePromptMode(e.target.checked)}
          className="rounded border-border accent-accent-blue"
        />
        Same prompt for all clips
      </label>

      <div className="space-y-2">
        {clips.map((clip, i) => (
          <div key={i} className="border border-border rounded-lg p-2 space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-text-muted uppercase tracking-wider font-medium">
                Clip {i + 1}
              </span>
              <span className="text-[10px] text-text-muted ml-auto">
                {slidingWindowSeconds}s
              </span>
            </div>

            <ClipDropZone
              file={clip.startImage}
              onFile={f => setClipStartImage(i, f)}
              onClear={() => setClipStartImage(i, null)}
            />

            <textarea
              value={singlePromptMode && i > 0 ? clips[0].prompt : clip.prompt}
              onChange={e => {
                if (singlePromptMode) {
                  setClipPrompt(0, e.target.value)
                } else {
                  setClipPrompt(i, e.target.value)
                }
              }}
              disabled={singlePromptMode && i > 0}
              placeholder={`Describe clip ${i + 1}...`}
              rows={2}
              className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-muted resize-none focus:outline-none focus:border-accent-blue transition-colors disabled:opacity-40"
            />
          </div>
        ))}
      </div>
    </div>
  )
}
