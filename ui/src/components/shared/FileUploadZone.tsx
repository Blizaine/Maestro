import { useCallback } from 'react'
import { Upload, X } from 'lucide-react'

export function FileUploadZone({ label, accept, filename, onFile, onClear }: {
  label: string
  accept: string
  filename: string | null
  onFile: (file: File) => void
  onClear: () => void
}) {
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  if (filename) {
    return (
      <div className="flex items-center gap-2 bg-bg-tertiary border border-border rounded-lg px-3 py-2">
        <span className="text-xs text-text-primary truncate flex-1">{filename}</span>
        <button
          onClick={onClear}
          className="p-0.5 rounded hover:bg-bg-hover text-text-muted hover:text-text-primary transition-colors shrink-0"
        >
          <X size={12} />
        </button>
      </div>
    )
  }

  return (
    <div
      className="border border-dashed border-border rounded-lg p-3 flex flex-col items-center justify-center gap-1 cursor-pointer hover:border-border-light transition-colors"
      onDrop={handleDrop}
      onDragOver={e => e.preventDefault()}
      onClick={() => {
        const input = document.createElement('input')
        input.type = 'file'
        input.accept = accept
        input.onchange = (e) => {
          const file = (e.target as HTMLInputElement).files?.[0]
          if (file) onFile(file)
        }
        input.click()
      }}
    >
      <Upload size={14} className="text-text-muted" />
      <span className="text-[10px] text-text-muted text-center">{label}</span>
    </div>
  )
}
