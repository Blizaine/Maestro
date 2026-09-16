import { useRef, useState } from 'react'
import { Download, Loader2, Upload, Mic } from 'lucide-react'
import * as api from '../../api/client'
import type { SavedOmniCharacter } from '../../types'
import { characterDisplayName } from '../../lib/characters'

const buttonClass = 'inline-flex items-center justify-center gap-1 rounded px-1.5 py-1 text-[10px] text-accent-blue hover:bg-bg-hover disabled:opacity-40 disabled:cursor-not-allowed'

export function ExportCharacterButton({ character, disabled = false, className }: {
  character: SavedOmniCharacter; disabled?: boolean; className?: string
}) {
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [download, setDownload] = useState<{ url: string; filename: string } | null>(null)
  const run = async () => {
    setBusy(true)
    setError('')
    setDownload(null)
    setMessage('Preparing character…')
    try {
      const result = await api.exportCharacter(character.id, setMessage)
      if (!result.url || !result.filename) throw new Error('Character export did not return a download.')
      setDownload({ url: result.url, filename: result.filename })
      const link = document.createElement('a')
      link.href = result.url
      link.download = result.filename
      document.body.appendChild(link)
      link.click()
      link.remove()
      setMessage('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Character export failed.')
    } finally {
      setBusy(false)
    }
  }
  return <div className="min-w-0">
    <button type="button" onClick={() => void run()} disabled={disabled || busy} className={className || buttonClass}
      aria-label={`Export ${characterDisplayName(character.name)}`} title="Download a .maestro.safetensors character with its saved voice">
      {busy ? <Loader2 size={11} className="animate-spin" /> : <Download size={11} />} Export
    </button>
    {busy && <p role="status" className="text-[9px] text-text-muted break-words">{message}</p>}
    {error && <p role="alert" className="text-[9px] text-indicator-error break-words">{error}</p>}
    {download && <a href={download.url} download={download.filename} className="block text-[9px] text-accent-blue underline break-all">Download file</a>}
  </div>
}

export function ImportCharacterButton({ disabled = false, onImported, className, label = 'Import character / RefMod' }: {
  disabled?: boolean; onImported?: (character: SavedOmniCharacter) => void; className?: string; label?: string
}) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const run = async (files: File[]) => {
    const file = files.find(item => item.name.toLowerCase().endsWith('.safetensors'))
    if (!file || files.filter(item => item.name.toLowerCase().endsWith('.safetensors')).length !== 1) {
      setError('Choose one character or RefMod .safetensors file. An older RefMod can include its matching .json file.')
      return
    }
    setBusy(true)
    setError('')
    setMessage('Uploading character…')
    try {
      const character = await api.importCharacterFile(file, setMessage, files.find(item => item.name.toLowerCase().endsWith('.json')))
      onImported?.(character)
      setMessage(`Imported ${characterDisplayName(character.name)}${character.voice ? ' with voice' : ''}.`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Character import failed.')
      setMessage('')
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }
  return <div className="min-w-0">
    <button type="button" onClick={() => input.current?.click()} disabled={disabled || busy} className={className || buttonClass}>
      {busy ? <Loader2 size={12} className="animate-spin" /> : <Upload size={12} />} {label}
    </button>
    <input ref={input} type="file" accept=".safetensors,.json" multiple className="hidden" aria-label="Import character file"
      onChange={event => void run(Array.from(event.target.files ?? []))} />
    {message && <p role="status" className="text-[10px] text-text-muted break-words">{message}</p>}
    {error && <p role="alert" className="text-[10px] text-indicator-error break-words">{error}</p>}
  </div>
}

export function CharacterVoiceButton({ character }: { character: SavedOmniCharacter }) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const run = async (file?: File) => {
    if (!file) return
    setBusy(true)
    setError('')
    try {
      const uploaded = await api.uploadAudio(file)
      await api.attachCharacterVoice(character.id, uploaded.path)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not save the voice.')
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }
  return <div className="min-w-0">
    <button type="button" onClick={() => input.current?.click()} disabled={busy} className={buttonClass}
      aria-label={`${character.voice ? 'Replace' : 'Add'} voice for ${characterDisplayName(character.name)}`}>
      {busy ? <Loader2 size={11} className="animate-spin" /> : <Mic size={11} />}{character.voice ? 'Replace voice' : 'Add voice'}
    </button>
    <input ref={input} type="file" className="hidden" accept="audio/*,video/*" onChange={event => void run(event.target.files?.[0])} />
    {error && <p role="alert" className="text-[10px] text-indicator-error break-words">{error}</p>}
  </div>
}
