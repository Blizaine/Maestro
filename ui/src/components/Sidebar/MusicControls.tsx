import { useLayoutEffect, useRef, useState } from 'react'
import { Music, Sparkles, Loader2 } from 'lucide-react'
import { useStore } from '../../stores/useStore'
import * as api from '../../api/client'
import type { GenerateParams } from '../../types'

const TEXTAREA_BASE =
  'w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary ' +
  'placeholder:text-text-muted focus:outline-none focus:border-accent-blue resize-none overflow-hidden'

// Textarea that grows to fit its content (no inner scrollbar — the sidebar
// already scrolls). Re-measures on every value change, so it also expands
// when the LLM fills it in. A min-height keeps it from collapsing when empty.
export function AutoGrowTextarea({
  value,
  onChange,
  placeholder,
  extraClass = '',
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  extraClass?: string
}) {
  const ref = useRef<HTMLTextAreaElement>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [value])
  return (
    <textarea
      ref={ref}
      value={value}
      onChange={e => onChange(e.target.value)}
      placeholder={placeholder}
      className={`${TEXTAREA_BASE} ${extraClass}`}
    />
  )
}

function StyleField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Style / Music Caption</label>
      <AutoGrowTextarea
        value={value}
        onChange={onChange}
        placeholder="Describe it like you're briefing musicians — genre, instruments, mood, production, vocals. e.g. dreamy bedroom-pop with shimmering reverb guitars and warm analog synths, soft breathy female vocals, nostalgic and intimate, gently mid-tempo"
        extraClass="min-h-[3.5rem]"
      />
    </div>
  )
}

function LyricsField({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div>
      <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Lyrics</label>
      <AutoGrowTextarea
        value={value}
        onChange={onChange}
        placeholder={'[Verse]\nYour lyrics here…\n[Chorus]\n…'}
        extraClass="min-h-[8rem] font-mono"
      />
    </div>
  )
}

export function MusicControls() {
  const description = useStore(s => s.musicDescription)
  const setDescription = useStore(s => s.setMusicDescription)
  const instrumental = useStore(s => s.musicInstrumental)
  const setInstrumental = useStore(s => s.setMusicInstrumental)
  const params = useStore(s => s.params)
  const setParam = useStore(s => s.setParam)

  const style = (params.alt_prompt as string) || ''
  const lyrics = (params.prompt as string) || ''
  const [writing, setWriting] = useState(false)
  const [writeError, setWriteError] = useState<string | null>(null)

  // alt_prompt = Music Caption (style); prompt = Lyrics. Both flow straight
  // to ACE-Step's generate (input_prompt=lyrics, alt_prompt=caption).
  const setStyle = (v: string) => setParam('alt_prompt' as keyof GenerateParams, v)
  const setLyrics = (v: string) => setParam('prompt', v)

  const toggleInstrumental = (on: boolean) => {
    setInstrumental(on)
    if (on) {
      setLyrics('[Instrumental]')
    } else if (lyrics.trim().toLowerCase() === '[instrumental]') {
      setLyrics('')
    }
  }

  const handleWriteSong = async () => {
    if (!description.trim() || writing) return
    setWriting(true)
    setWriteError(null)
    try {
      const r = await api.writeSong({ description: description.trim(), instrumental })
      if (r.style) setStyle(r.style)
      setLyrics(instrumental ? '[Instrumental]' : (r.lyrics || ''))
    } catch (e) {
      setWriteError(e instanceof Error ? e.message : 'Song writing failed')
    } finally {
      setWriting(false)
    }
  }

  return (
    <div className="space-y-3">
      {/* Header + instrumental toggle */}
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-text-muted uppercase tracking-wider flex items-center gap-1.5">
          <Music size={12} /> Song
        </label>
        <label className="flex items-center gap-1.5 cursor-pointer group text-[10px] text-text-secondary hover:text-text-primary transition-colors">
          <input
            type="checkbox"
            checked={instrumental}
            onChange={e => toggleInstrumental(e.target.checked)}
            className="accent-accent-blue"
          />
          Instrumental
        </label>
      </div>

      {/* Describe → let the LLM write it (optional — fields below are editable) */}
      <div className="space-y-2">
        <div>
          <label className="text-[11px] text-text-muted uppercase tracking-wider mb-1.5 block">Describe your song</label>
          <AutoGrowTextarea
            value={description}
            onChange={setDescription}
            placeholder="e.g. an upbeat synthwave track about late-night city driving — nostalgic but hopeful"
            extraClass="min-h-[4.5rem]"
          />
        </div>
        <button
          onClick={handleWriteSong}
          disabled={!description.trim() || writing}
          className={`w-full px-4 py-2 rounded-lg flex items-center justify-center gap-1.5 font-medium text-xs transition-all ${
            !description.trim() || writing
              ? 'bg-bg-tertiary text-text-muted cursor-not-allowed border border-border'
              : 'bg-cta hover:brightness-110 shadow-accent-glow text-white'
          }`}
        >
          {writing ? <Loader2 size={13} className="animate-spin" /> : <Sparkles size={13} />}
          {writing ? 'Writing…' : 'Write Song'}
        </button>
        {writeError && <p className="text-[10px] text-red-400 leading-snug">{writeError}</p>}
        <p className="text-[10px] text-text-muted leading-snug">
          Let the LLM write the Style{instrumental ? '' : ' + Lyrics'} from your description — or just fill them in yourself below. Either way, edit and Generate.
        </p>
      </div>

      {/* Style + Lyrics (editable, auto-sizing). Lyrics hidden when instrumental. */}
      <StyleField value={style} onChange={setStyle} />
      {!instrumental && <LyricsField value={lyrics} onChange={setLyrics} />}
    </div>
  )
}
