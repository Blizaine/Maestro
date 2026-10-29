import {
  alignTitleSequence,
  titleBrief,
  titleFrames,
  titlePlanIsCurrent,
  titleSeconds,
  titleSequenceTimeline,
  TITLE_EFFECT_LABELS,
  TITLE_SEQUENCE_DURATIONS,
} from './editorAnimatedTitles'
import type { AnimatedTitleSettings } from './editorAnimatedTitles'
import { AnimatedTitleAdvancedControls } from './AnimatedTitleAdvancedControls'
import type { TitleComposerController } from './useTitleComposer'

const FIELD = 'w-full min-w-0 rounded-lg border border-border bg-bg-tertiary px-3 py-2 text-xs text-text-primary outline-none focus:border-accent-blue/60'

function framesFor(settings: AnimatedTitleSettings) {
  try { return titleFrames(titleSeconds(settings)) } catch { return TITLE_SEQUENCE_DURATIONS[0].frames }
}

/** The shared, natural-language title composer. Workspace-specific submission stays in its caller. */
export function AnimatedTitleControls({ settings, patch, busy, canvas, destination = 'editor', at = 0, composer }: {
  settings: AnimatedTitleSettings
  patch: (value: Partial<AnimatedTitleSettings>) => void
  busy: boolean
  canvas: { width: number; height: number }
  destination?: 'editor' | 'studio'
  at?: number
  composer: TitleComposerController
}) {
  const background = settings.background ?? 'chroma'
  const brief = titleBrief(settings)
  const selectedFrames = framesFor(settings)
  let currentPlan = false
  try { currentPlan = settings.brief !== undefined && Boolean(settings.plannedKey) && titlePlanIsCurrent(settings) }
  catch { /* Invalid manual duration and other malformed settings remain editable instead of breaking render. */ }
  const sequence = settings.mode === 'sequence'
  const phrases = settings.phrases ?? []
  let timeline: ReturnType<typeof titleSequenceTimeline> | null = null
  if (sequence) {
    try { timeline = titleSequenceTimeline(settings) } catch { /* Show the available wording while an invalid advanced plan is repaired. */ }
  }
  const reviewRows = sequence ? (timeline?.phrases ?? (() => {
    let start = 0
    return phrases.map(phrase => {
      const seconds = Number.isFinite(phrase.seconds) && phrase.seconds > 0 ? phrase.seconds : 0
      const row = { ...phrase, start, end: start + seconds }
      start += seconds
      return row
    })
  })()) : []

  function patchCreative(value: Partial<AnimatedTitleSettings>) {
    composer.clearFeedback()
    patch({ ...value, brief: titleBrief(settings), manualPlanEdited: false })
  }

  function changeBrief(value: string) {
    composer.clearFeedback()
    patch({ brief: value, manualPlanEdited: false })
  }

  function changeDuration(frames: number) {
    composer.clearFeedback()
    const base = { ...settings, brief: titleBrief(settings), manualPlanEdited: false, seconds: frames / 24, sequenceFrames: frames }
    if (base.mode === 'sequence' && (base.phrases?.length ?? 0) >= 2 && (base.phrases?.length ?? 0) <= 6) {
      try { patch(alignTitleSequence(base, frames)); return } catch { /* Keep the chosen whole-clip duration visible while the invalid sequence is repaired. */ }
    }
    patch(base)
  }

  function patchAdvanced(value: Partial<AnimatedTitleSettings>) {
    const creativeFields: (keyof AnimatedTitleSettings)[] = [
      'text', 'look', 'effect', 'direction', 'mode', 'phrases', 'sequenceDirection', 'sequenceFrames', 'seconds', 'background',
    ]
    const creativeChange = creativeFields.some(field => Object.hasOwn(value, field))
    if (creativeChange && settings.brief !== undefined) composer.clearFeedback()
    patch({ ...value, ...(creativeChange && settings.brief !== undefined ? { plannedKey: undefined, manualPlanEdited: true } : {}) })
  }

  return <div className="space-y-4">
    <label className="block space-y-1.5 text-xs text-text-secondary">Describe your title animation
      <textarea aria-label="Describe your title animation" value={brief} maxLength={4000} rows={5} disabled={busy}
        onChange={event => changeBrief(event.target.value)}
        placeholder="A silver title spelling ‘ORION’ forms from fine particles, holds clearly, then dissolves into a soft blue glow."
        className={FIELD} />
      <span className="block text-[11px] leading-relaxed text-text-muted">
        Write the words exactly as they should appear, including spelling, punctuation and language. The plan shows the detected wording, order and timing for review.
        <span className="block">{brief.length}/4,000 characters</span>
      </span>
    </label>

    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Title style preset <span className="font-normal text-text-muted">(optional)</span>
        <select aria-label="Title style preset" value={settings.stylePreset ?? 'custom'} disabled={busy}
          onChange={event => patchCreative({ stylePreset: event.target.value as AnimatedTitleSettings['stylePreset'] })}
          className={FIELD}>
          <option value="custom">Custom description</option>
          {Object.entries(TITLE_EFFECT_LABELS).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>

      <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Title clip duration
        <select aria-label="Title clip duration" value={selectedFrames} disabled={busy}
          onChange={event => changeDuration(Number(event.target.value))} className={FIELD}>
          {TITLE_SEQUENCE_DURATIONS.map(option => <option key={option.frames} value={option.frames}>{option.label}</option>)}
        </select>
        <span className="block text-[11px] leading-relaxed text-text-muted">Whole-frame H3 duration at 24 fps.</span>
      </label>
    </div>

    <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
      <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Title background
        <select aria-label="Title background" value={background} disabled={busy}
          onChange={event => patchCreative({ background: event.target.value as NonNullable<AnimatedTitleSettings['background']> })} className={FIELD}>
          <option value="chroma">Chroma key (green)</option><option value="black">Black background</option>
        </select>
        <span className="block text-[11px] leading-relaxed text-text-muted">{background === 'chroma'
          ? destination === 'studio' ? 'Studio keeps the green background in the saved clip, ready for keying.' : 'The Editor removes green automatically so the title shows over footage below.'
          : destination === 'studio' ? 'Studio keeps the opaque black background in the saved clip.' : 'Black stays opaque and covers footage below for the title clip.'}</span>
      </label>

      <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Aspect ratio
        <select aria-label="Title aspect ratio" value={settings.aspect} disabled={busy}
          onChange={event => patch({ aspect: event.target.value as AnimatedTitleSettings['aspect'] })} className={FIELD}>
          {destination === 'editor' && <option value="project">Match project</option>}
          <option value="16:9">16:9 · landscape</option><option value="9:16">9:16 · portrait</option><option value="1:1">1:1 · square</option>
        </select>
      </label>

      <label className="block min-w-0 space-y-1.5 text-xs text-text-secondary">Resolution
        <select aria-label="Title resolution" value={settings.resolution} disabled={busy}
          onChange={event => patch({ resolution: event.target.value as AnimatedTitleSettings['resolution'] })} className={FIELD}>
          <option value="480p">480p · faster</option><option value="720p">720p</option>
        </select>
      </label>
    </div>

    <div className="flex flex-wrap items-center gap-2">
      <button type="button" disabled={busy || !brief.trim() || brief.length > 4000} onClick={() => void composer.enhance()}
        className="flex min-h-10 items-center gap-2 rounded-lg border border-border px-3 py-2 text-xs font-semibold text-text-primary hover:border-accent-blue/60 disabled:opacity-40">
        {composer.busy ? <span aria-hidden="true" className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-r-transparent" /> : null}
        {composer.busy ? 'Enhancing…' : 'Enhance with AI'}
      </button>
      {composer.message && <p role="status" className="min-w-0 flex-1 text-xs text-text-secondary">{composer.message}</p>}
    </div>
    {composer.error && <p role="alert" className="rounded-lg border border-red-500/25 bg-red-500/5 p-3 text-xs text-red-400">{composer.error}</p>}

    {currentPlan && <details data-testid="title-plan-review" aria-label="Title plan review" open={settings.titleReviewRequired || undefined}
      className="rounded-xl border border-accent-blue/25 bg-accent-blue/5 p-3 text-xs text-text-secondary">
      <summary className="cursor-pointer select-none">
        <span className="block font-semibold text-text-primary">Review title plan · {sequence ? `${phrases.length} phrases` : 'one title'} · {(framesFor(settings) / 24).toFixed(3)}s</span>
        <span className="mt-1 block break-words text-[11px] text-text-muted">Detected wording: {sequence
          ? phrases.map(phrase => phrase.text.trim() || 'Untitled phrase').join(' → ') || 'No wording detected'
          : settings.text || 'No wording detected'}</span>
      </summary>
      <div className="mt-3 space-y-3">
        <p className="text-[11px] leading-relaxed text-text-secondary">Check the detected wording, order and timing before generating.</p>
        {sequence ? <ol className="space-y-2 text-xs text-text-secondary">
          {reviewRows.map((phrase, index) => <li key={phrase.id} className="flex flex-wrap justify-between gap-x-3 gap-y-1 rounded-lg bg-bg-secondary/70 px-3 py-2">
            <span className="min-w-0"><strong>{index + 1}.</strong> {phrase.text || `Phrase ${index + 1}`}
              {phrase.animation && <span className="mt-1 block text-[11px] text-text-muted">{phrase.animation}</span>}
            </span>
            <span className="shrink-0 text-text-muted">{phrase.start.toFixed(3)}–{phrase.end.toFixed(3)}s</span>
          </li>)}
        </ol> : <div className="rounded-lg bg-bg-secondary/70 px-3 py-2 text-xs text-text-secondary">
          <strong>Title wording:</strong> {settings.text || 'No wording detected'}
          <span className="mt-1 block text-[11px] text-text-muted">One title · {titleSeconds(settings).toFixed(3)}s at 24 fps</span>
          {settings.direction && <span className="mt-1 block text-[11px] text-text-muted">Motion: {settings.direction}</span>}
        </div>}
        <p className="text-[11px] text-text-muted">Clip: {framesFor(settings)} frames · {(framesFor(settings) / 24).toFixed(3)}s at 24 fps. Review the generated video before using it; results can vary.</p>
      </div>
    </details>}

    {settings.plannerWarnings?.length ? <ul aria-label="Title planner warnings" className="space-y-1 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-[11px] text-amber-400">
      {settings.plannerWarnings.map((warning, index) => <li key={`${warning}-${index}`}>{warning}</li>)}
    </ul> : null}

    {settings.rawTitleDraft && <details className="rounded-lg border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
      <summary className="cursor-pointer">Raw planner draft diagnostics</summary>
      {settings.titleDraftHistoryId && <a className="mt-2 inline-flex text-accent-blue underline" href={`/api/v1/llm/drafts/${encodeURIComponent(settings.titleDraftHistoryId)}`} target="_blank" rel="noreferrer">Open saved draft history</a>}
      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words text-[11px] leading-relaxed">{settings.rawTitleDraft}</pre>
    </details>}

    <details className="rounded-xl border border-border bg-bg-tertiary p-3 text-xs text-text-secondary">
      <summary className="cursor-pointer select-none font-medium">Advanced title controls</summary>
      <div className="mt-3 space-y-3">
        <AnimatedTitleAdvancedControls settings={settings} patch={patchAdvanced} busy={busy} canvas={canvas} destination={destination} at={at} />
        <button type="button" disabled={busy} onClick={composer.acceptManual}
          className="min-h-9 rounded-lg border border-border px-3 py-2 text-xs font-medium text-text-primary hover:border-accent-blue/60 disabled:opacity-40">
          Use manual title plan
        </button>
        <p className="text-[11px] leading-relaxed text-text-muted">{settings.manualPlanEdited
          ? 'Advanced edits are waiting for approval. Use this manual plan to keep them, or Enhance to replace them with a new AI plan.'
          : 'Use this when you want to set the exact title wording and motion yourself. The manual plan is validated before it can be generated.'}</p>
      </div>
    </details>
  </div>
}
