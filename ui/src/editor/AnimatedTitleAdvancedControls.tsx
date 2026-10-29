import { useState } from 'react'
import {
  newTitlePhrases,
  titleSubmissionId,
  titleFormState,
  titleSequenceTimeline,
  titleFrames,
  alignTitleSequence,
  setTitlePhraseSeconds,
  titleDurationChoices,
  titleFps,
  titleBackground,
  TITLE_EFFECTS,
  TITLE_EFFECT_LABELS,
  TITLE_PHRASE_ANIMATIONS,
} from './editorAnimatedTitles'
import type { AnimatedTitleSettings, TitlePhrase } from './editorAnimatedTitles'
import { TitleBackgroundControl } from './TitleBackgroundControl'

const FIELD = 'w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'
const MAX_PHRASES = 6
const MIN_PHRASES = 2
const MAX_PHRASE_LENGTH = 80

function averagePhraseSeconds(phrases: TitlePhrase[]) {
  if (phrases.length === 0) return 2.6
  const total = phrases.reduce((sum, phrase) => sum + (Number.isFinite(phrase.seconds) ? phrase.seconds : 0), 0)
  return Math.round(Math.max(1, Math.min(10, total / phrases.length)) * 10) / 10
}

function selectedSequenceFrames(settings: AnimatedTitleSettings, phrases: TitlePhrase[]) {
  const choices = titleDurationChoices(settings)
  const minimumFrames = phrases.length * titleFps(settings)
  let frames: number
  if (Number.isInteger(settings.sequenceFrames)) frames = settings.sequenceFrames!
  else {
    const total = phrases.reduce((sum, phrase) => sum + (Number.isFinite(phrase.seconds) ? phrase.seconds : 0), 0)
    try { frames = titleFrames(total, settings) } catch { frames = choices[0]?.frames ?? 124 }
  }
  if (frames >= minimumFrames && choices.some(option => option.frames === frames)) return frames
  const compatible = choices.find(option => option.frames >= Math.max(frames, minimumFrames))
  if (compatible) return compatible.frames
  return choices.at(-1)?.frames ?? frames
}

function alignPhrases(settings: AnimatedTitleSettings, phrases: TitlePhrase[], frames = selectedSequenceFrames(settings, settings.phrases ?? [])) {
  return alignTitleSequence({ ...settings, mode: 'sequence', phrases }, frames)
}

/** Shared title fields; placement and submission remain owned by each workspace. */
export function AnimatedTitleAdvancedControls({ settings, patch, busy, canvas, destination = 'editor', at = 0 }: {
  settings: AnimatedTitleSettings
  patch: (value: Partial<AnimatedTitleSettings>) => void
  busy: boolean
  canvas: { width: number; height: number }
  destination?: 'editor' | 'studio'
  at?: number
}) {
  const background = titleBackground(settings)
  const fps = titleFps(settings)
  const durationChoices = titleDurationChoices(settings)
  const sequenceMode = settings.mode === 'sequence'
  const phrases = settings.phrases ?? []
  const [pasteLines, setPasteLines] = useState('')
  const [pasteError, setPasteError] = useState('')
  const [modeError, setModeError] = useState('')
  const formState = titleFormState(settings, canvas)
  const sequenceTotal = phrases.reduce((sum, phrase) => sum + (Number.isFinite(phrase.seconds) ? phrase.seconds : 0), 0)
  const nativeFrames = selectedSequenceFrames(settings, phrases)
  const nativeDuration = durationChoices.find(option => option.frames === nativeFrames)
  const sequenceDurationOptions = durationChoices.filter(option => option.frames >= phrases.length * fps)
  const sequenceCharacters = phrases.map(phrase => phrase.text.trim()).join('\n').length
  const timeline = sequenceMode ? (() => {
    try { return titleSequenceTimeline(settings) } catch { return null }
  })() : null
  const timelineRows = timeline?.phrases ?? (() => {
    let start = 0
    return phrases.map(phrase => {
      const seconds = Number.isFinite(phrase.seconds) && phrase.seconds > 0 ? phrase.seconds : 0
      const row = { ...phrase, start, end: start + seconds }
      start += seconds
      return row
    })
  })()
  const { geometry, duration, prompt, validation } = formState

  function setMode(mode: 'single' | 'sequence') {
    if (mode === 'sequence') {
      try {
        const nextPhrases = settings.phrases?.length
          ? settings.phrases
          : newTitlePhrases(settings.text, settings.seconds)
        const frames = selectedSequenceFrames(settings, nextPhrases)
        patch(alignPhrases({ ...settings, mode }, nextPhrases, frames))
        setModeError('')
      } catch (reason) {
        setModeError(reason instanceof Error ? reason.message : 'Check the title text before switching to a sequence.')
      }
      return
    }
    patch({ mode })
    setModeError('')
  }

  function patchPhrase(id: string, changes: Partial<TitlePhrase>) {
    const nextPhrases = phrases.map(phrase => phrase.id === id ? { ...phrase, ...changes } : phrase)
    patch({ phrases: nextPhrases })
  }

  function reorderPhrase(index: number, direction: -1 | 1) {
    const destinationIndex = index + direction
    if (destinationIndex < 0 || destinationIndex >= phrases.length) return
    const nextPhrases = [...phrases]
    ;[nextPhrases[index], nextPhrases[destinationIndex]] = [nextPhrases[destinationIndex], nextPhrases[index]]
    patch({ phrases: nextPhrases })
  }

  function addPhrase() {
    if (phrases.length >= MAX_PHRASES) return
    const nextPhrases = [...phrases, {
      id: titleSubmissionId(), text: '', seconds: averagePhraseSeconds(phrases), animation: '',
    }]
    patch(alignPhrases(settings, nextPhrases))
  }

  function removePhrase(id: string) {
    if (phrases.length <= MIN_PHRASES) return
    const nextPhrases = phrases.filter(phrase => phrase.id !== id)
    patch(alignPhrases(settings, nextPhrases))
  }

  function applyPastedLines(value: string) {
    try {
      const nextPhrases = newTitlePhrases(value, nativeDuration?.seconds ?? (sequenceTotal || settings.seconds))
      patch(alignPhrases(settings, nextPhrases))
      setPasteError('')
    } catch (reason) {
      setPasteError(reason instanceof Error ? reason.message : 'Check the pasted phrase lines.')
    }
  }

  return <>
        <p className="text-xs leading-relaxed text-text-secondary">{settings.model?.name ?? 'H3 Fused 4-Step · Frames'}. Background: {background.label}.
          {settings.model?.startFrame === false ? ' The model follows the background instruction in the prompt.' : ' Matching boundary images are prepared for the model’s supported frame inputs.'}
          {destination === 'editor' && (background.chromaKey ? ' Chroma key is enabled on insertion.' : ' The background stays opaque.')}</p>

        <div role="group" aria-label="Title format" className="grid grid-cols-2 gap-2 rounded-xl border border-border bg-bg-tertiary p-1">
          <button type="button" aria-pressed={!sequenceMode} disabled={busy} onClick={() => setMode('single')}
            className={`min-h-9 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${!sequenceMode ? 'bg-bg-secondary text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'}`}>
            Single title
          </button>
          <button type="button" aria-pressed={sequenceMode} disabled={busy} onClick={() => setMode('sequence')}
            className={`min-h-9 rounded-lg px-3 py-2 text-xs font-medium transition-colors ${sequenceMode ? 'bg-bg-secondary text-text-primary shadow-sm' : 'text-text-secondary hover:text-text-primary'}`}>
            Phrase sequence
          </button>
        </div>
        {modeError && <p role="alert" className="text-xs text-red-400">{modeError}</p>}

        {!sequenceMode && <label className="block space-y-1.5 text-xs text-text-secondary">Title text
          <textarea aria-label="Animated title text" value={settings.text} maxLength={160} rows={2} disabled={busy}
            onChange={event => patch({ text: event.target.value })} placeholder="Create something extraordinary" className={FIELD} />
          <span className="block text-[11px] leading-relaxed text-text-muted">{settings.text.length}/160 characters</span>
        </label>}

        {sequenceMode && <section aria-label="Phrase sequence" className="space-y-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">Phrase sequence</h3>
            <p className="mt-1 text-[11px] leading-relaxed text-text-muted">Each phrase appears in order with its own timing and motion. Use 2–6 phrases.</p>
          </div>

          <label className="block space-y-1.5 text-xs text-text-secondary">Sequence duration
            <select aria-label="Sequence duration" value={nativeFrames} disabled={busy}
              onChange={event => patch(alignTitleSequence(settings, Number(event.target.value)))} className={FIELD}>
              {sequenceDurationOptions.map(option => <option key={option.frames} value={option.frames}>{option.label}</option>)}
            </select>
            <span className="block text-[11px] leading-relaxed text-text-muted">
              {nativeDuration ? `${nativeDuration.seconds.toFixed(3)} seconds · ${nativeDuration.frames} frames at ${fps} fps.` : `${(nativeFrames / fps).toFixed(3)} seconds · ${nativeFrames} frames at ${fps} fps.`} Changing a phrase time adjusts the other phrases to keep the selected clip length.
            </span>
          </label>

          <div className="space-y-3">
            {phrases.map((phrase, index) => {
              const row = timelineRows.find(item => item.id === phrase.id)
              const selectedPreset = Object.entries(TITLE_PHRASE_ANIMATIONS)
                .find(([, preset]) => preset.direction === phrase.animation)?.[0] ?? 'custom'
              const phraseLength = phrase.text.length
              return <section key={phrase.id} data-testid="title-phrase-card" role="group" aria-label={`Phrase ${index + 1}`}
                className="min-w-0 space-y-3 rounded-xl border border-border bg-bg-tertiary p-3">
                <div className="flex min-w-0 items-center justify-between gap-2">
                  <h4 className="text-xs font-semibold text-text-primary">Phrase {index + 1}</h4>
                  <div className="flex shrink-0 items-center gap-1">
                    <button type="button" aria-label={`Move phrase ${index + 1} up`} title="Move up" disabled={busy || index === 0}
                      onClick={() => reorderPhrase(index, -1)} className="min-h-9 min-w-9 rounded-md border border-border px-2 text-xs text-text-secondary disabled:opacity-40">↑</button>
                    <button type="button" aria-label={`Move phrase ${index + 1} down`} title="Move down" disabled={busy || index === phrases.length - 1}
                      onClick={() => reorderPhrase(index, 1)} className="min-h-9 min-w-9 rounded-md border border-border px-2 text-xs text-text-secondary disabled:opacity-40">↓</button>
                    <button type="button" aria-label={`Remove phrase ${index + 1}`} title="Remove phrase" disabled={busy || phrases.length <= MIN_PHRASES}
                      onClick={() => removePhrase(phrase.id)} className="min-h-9 rounded-md border border-border px-2 text-xs text-text-secondary disabled:opacity-40">Remove</button>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_8rem]">
                  <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Phrase text
                    <input aria-label={`Phrase ${index + 1} text`} type="text" value={phrase.text} maxLength={MAX_PHRASE_LENGTH} disabled={busy}
                      onChange={event => patchPhrase(phrase.id, { text: event.target.value })}
                      placeholder={index === 0 ? 'CREATE' : index === 1 ? 'EXPLORE' : 'INSPIRE'} className={FIELD} />
                    <span className={`block text-[11px] leading-relaxed ${phraseLength >= MAX_PHRASE_LENGTH ? 'text-amber-400' : 'text-text-muted'}`}>
                      {phraseLength}/{MAX_PHRASE_LENGTH} characters{phraseLength >= MAX_PHRASE_LENGTH ? ' · limit reached' : ''}
                    </span>
                  </label>

                  <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Time on screen (seconds)
                    <input aria-label={`Phrase ${index + 1} duration`} type="number" min={1} max={10} step="any"
                      value={Number.isFinite(phrase.seconds) ? Number(phrase.seconds.toFixed(3)) : ''} disabled={busy}
                      onChange={event => patchPhrase(phrase.id, { seconds: Number(event.target.value) })}
                      onBlur={event => {
                        const seconds = Number(event.currentTarget.value)
                        if (Number.isFinite(seconds)) patch(setTitlePhraseSeconds(settings, phrase.id, seconds))
                      }} className={FIELD} />
                  </label>
                </div>

                <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Describe this phrase’s motion
                  <textarea aria-label={`Phrase ${index + 1} animation`} value={phrase.animation} maxLength={800} rows={2} disabled={busy}
                    onChange={event => patchPhrase(phrase.id, { animation: event.target.value })}
                    placeholder="Describe how this phrase appears, holds, and clears before the next one." className={FIELD} />
                  <span className="block text-[11px] leading-relaxed text-text-muted">Optional. Leave blank for a simple fade in, readable hold, and complete fade out.</span>
                </label>

                <details className="rounded-lg border border-border p-2 text-xs text-text-secondary">
                  <summary className="cursor-pointer">Use a motion preset (optional)</summary>
                  <label className="mt-2 block min-w-0 space-y-1.5">Animation preset
                    <select aria-label={`Phrase ${index + 1} animation preset`} value={selectedPreset} disabled={busy}
                      onChange={event => {
                        const value = event.target.value as keyof typeof TITLE_PHRASE_ANIMATIONS | 'custom'
                        patchPhrase(phrase.id, { animation: value === 'custom' ? '' : TITLE_PHRASE_ANIMATIONS[value].direction })
                      }} className={FIELD}>
                      <option value="custom">Custom description</option>
                      {Object.entries(TITLE_PHRASE_ANIMATIONS).map(([value, preset]) => <option key={value} value={value}>{preset.label}</option>)}
                    </select>
                  </label>
                  <p className="mt-2 text-[11px] leading-relaxed text-text-muted">A preset fills the motion description; edit it freely.</p>
                </details>

                {row && <p className="text-[11px] text-text-muted">Approx. {row.start.toFixed(3)}–{row.end.toFixed(3)}s · {Math.round((row.end - row.start) * fps)} frames</p>}
                {!phrase.text.trim() && <p className="text-[11px] text-amber-400">Add wording for this phrase.</p>}
                {phraseLength > MAX_PHRASE_LENGTH && <p role="alert" className="text-[11px] text-red-400">Phrase text must be 80 characters or fewer.</p>}
                {(!Number.isFinite(phrase.seconds) || phrase.seconds < 1 || phrase.seconds > 10) && <p role="alert" className="text-[11px] text-red-400">Use a time between 1 and 10 seconds.</p>}
              </section>
            })}
          </div>

          {phrases.length < MAX_PHRASES && <button type="button" disabled={busy} onClick={addPhrase}
            className="min-h-10 w-full rounded-lg border border-dashed border-border px-3 py-2 text-xs font-medium text-text-secondary hover:border-accent-blue/60 hover:text-text-primary disabled:opacity-50">
            + Add phrase
          </button>}
          {phrases.length > MAX_PHRASES && <p role="alert" className="text-xs text-red-400">A sequence can contain at most 6 phrases.</p>}

          <details className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
            <summary className="cursor-pointer">Paste lines to make phrases</summary>
            <label className="mt-3 block space-y-1.5">One phrase per line
              <textarea aria-label="Paste phrases, one per line" rows={4} disabled={busy} value={pasteLines}
                onChange={event => { setPasteLines(event.target.value); setPasteError('') }}
                placeholder={'CREATE\nEXPLORE\nINSPIRE'} className={FIELD} />
              <span className="block text-[11px] text-text-muted">{pasteLines.split(/\r?\n/).map(line => line.trim()).filter(Boolean).length} non-empty lines · maximum 6</span>
            </label>
            {pasteError && <p role="alert" className="mt-2 text-[11px] text-red-400">{pasteError}</p>}
            <button type="button" disabled={busy} className="mt-3 min-h-9 rounded-lg border border-border px-3 py-2 text-xs text-text-secondary disabled:opacity-50"
              onClick={() => applyPastedLines(pasteLines)}>
              Replace phrases from lines
            </button>
          </details>

          <div aria-label="Approximate sequence timeline" className="space-y-2 rounded-lg border border-border bg-bg-tertiary p-3 text-[11px] leading-relaxed text-text-muted">
            <h4 className="text-xs font-medium text-text-secondary">Approximate timeline</h4>
            <ol className="space-y-1">
              {timelineRows.map((row, index) => <li key={row.id} className="flex min-w-0 justify-between gap-3">
                <span className="min-w-0 truncate">{index + 1}. {row.text.trim() || `Phrase ${index + 1}`}</span>
                <span className="shrink-0">{row.start.toFixed(3)}–{row.end.toFixed(3)}s · {Math.round((row.end - row.start) * fps)} frames</span>
              </li>)}
            </ol>
            <p>Selected clip: {nativeFrames} frames · {(nativeFrames / fps).toFixed(3)}s at {fps} fps. Phrase timings align to whole frames; The model can interpret the requested timing approximately.</p>
            <p>{sequenceCharacters}/160 total text characters</p>
            {phrases.length < MIN_PHRASES && <p role="alert" className="text-amber-400">Add at least 2 phrases.</p>}
            {sequenceCharacters > 160 && <p role="alert" className="text-amber-400">Shorten the phrases to stay within 160 characters total.</p>}
            <p>Each phrase clears before the next one; the last clears to the selected {background.label} background.</p>
          </div>
        </section>}

        <TitleBackgroundControl settings={settings} patch={patch} busy={busy} editor={destination === 'editor'} />
        <label className="block space-y-1.5 text-xs text-text-secondary">Lettering and visual style
          <input aria-label="Title visual style" value={settings.look} maxLength={1200} disabled={busy}
            onChange={event => patch({ look: event.target.value })} placeholder="Describe any style: anime, hand-drawn, metallic, minimal…" className={FIELD} />
        </label>
        <label className="block space-y-1.5 text-xs text-text-secondary">{sequenceMode ? 'Shared direction (optional)' : 'Describe the animation'}
          <textarea aria-label={sequenceMode ? 'Shared title direction' : 'Title animation description'} value={sequenceMode ? settings.sequenceDirection ?? '' : settings.direction} maxLength={1200} rows={3} disabled={busy}
            onChange={event => patch(sequenceMode ? { sequenceDirection: event.target.value } : { direction: event.target.value, effect: 'custom' })}
            placeholder={sequenceMode ? 'Optional direction shared across the phrase sequence.' : 'Describe how the title appears, moves and disappears, including any timing you want.'} className={FIELD} />
        </label>
        {!sequenceMode && <details className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
          <summary className="cursor-pointer">Use an animation preset (optional)</summary>
          <label className="mt-3 block space-y-1.5">Animation preset
            <select aria-label="Title animation preset" value={settings.effect} disabled={busy}
              onChange={event => {
                const effect = event.target.value as AnimatedTitleSettings['effect']
                patch({ effect, ...(effect !== 'custom' ? { direction: TITLE_EFFECTS[effect] } : {}) })
              }} className={FIELD}>
              <option value="custom">None — use my description</option>
              {Object.entries(TITLE_EFFECT_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select>
          </label>
          <p className="mt-2 text-[11px] leading-relaxed text-text-muted">A preset fills the description as a starting point. Edit it freely; only the description is sent as the animation direction.</p>
        </details>}
        <div className="grid grid-cols-2 gap-3">
          {destination === 'editor' && !sequenceMode && <label className="space-y-1.5 text-xs text-text-secondary">Duration (seconds)
            <input aria-label="Title duration" type="number" min={5} max={14.5} step={.1} value={settings.seconds} disabled={busy}
              onChange={event => patch({ seconds: Number(event.target.value) })} className={FIELD} />
          </label>}
          {destination === 'editor' && <label className="space-y-1.5 text-xs text-text-secondary">Aspect ratio
            <select aria-label="Title aspect ratio" value={settings.aspect} disabled={busy} onChange={event => patch({ aspect: event.target.value as AnimatedTitleSettings['aspect'] })} className={FIELD}>
              <option value="project">Match project</option><option value="21:9">21:9 · ultrawide</option><option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · portrait</option><option value="1:1">1:1 · square</option><option value="4:3">4:3</option><option value="3:4">3:4</option>
            </select>
          </label>}
          {destination === 'editor' && <label className="space-y-1.5 text-xs text-text-secondary">Resolution
            <select aria-label="Title resolution" value={settings.resolution} disabled={busy} onChange={event => patch({ resolution: event.target.value as AnimatedTitleSettings['resolution'] })} className={FIELD}>
              <option value="480p">480p · faster</option><option value="540p">540p</option><option value="720p">720p</option><option value="768p">768p · native H3</option><option value="1080p">1080p · experimental</option>
            </select>
          </label>}
        </div>
        {destination === 'editor' && <label className="flex items-center gap-2 text-xs text-text-secondary"><input type="checkbox" checked={settings.keepAudio} disabled={busy}
          onChange={event => patch({ keepAudio: event.target.checked })} className="accent-[var(--color-accent-blue)]" />Keep generated clip audio</label>}
        {destination === 'studio' && <p className="text-[11px] leading-relaxed text-text-muted">Generated video audio is retained. You can mute it in the Editor or another editing app.</p>}
        <div className="rounded-lg border border-border bg-bg-tertiary p-3 text-[11px] leading-relaxed text-text-muted">
          {geometry.width}×{geometry.height} · {duration ? sequenceMode ? `${nativeFrames} frames · ${(nativeFrames / fps).toFixed(3)}s at ${fps} fps` : `${duration.toFixed(2)}s at ${fps} fps` : sequenceMode ? `about ${sequenceTotal.toFixed(1)}s requested` : 'Check duration'}{destination === 'editor' ? ` · inserts at ${at.toFixed(2)}s.` : ' · saved to the gallery.'}<br />
          Background: {background.label}{background.chromaKey ? destination === 'studio' ? ', ready for keying.' : ', removed automatically.' : ', opaque.'}<br />
          No reference characters or Studio settings are reused. Review the wording, timing and edge quality; generated results can vary.
        </div>
        {prompt && <details className="text-xs text-text-secondary"><summary className="cursor-pointer">View prepared prompt</summary><pre className="mt-2 whitespace-pre-wrap rounded-lg bg-bg-tertiary p-3 text-[11px] leading-relaxed">{prompt}</pre></details>}
        {sequenceMode && validation && <p role="alert" className="text-xs text-red-400">{validation}</p>}
        {!sequenceMode && validation && (!geometry.width || !duration) && <p role="alert" className="text-xs text-red-400">{validation}</p>}
  </>
}
